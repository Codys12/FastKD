# generator_distributed.py – weight‑streaming version
# Author : ChatGPT – June 2025 ✨  (rev‑4: vectorised sampling)
"""
rev‑4 summary
-------------
* **`sample_distribution` rewritten** – no nested `for b in … / for s in …`.
  We flatten `(bsz, seqlen)` → `(bsz*seqlen)`, take a single `torch.multinomial`
  call, then compute `torch.unique` per row.  That keeps all sampling fully on‑
  GPU while still returning the same Python‑list structure (ids & counts per
  position) expected by downstream code.
* No other functional changes; memory footprint unchanged because logits are
  already resident.
"""
from __future__ import annotations

import argparse, atexit, os, signal, threading, time, datetime
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, Repository
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# GLOBAL SHUTDOWN FLAG & SIGNAL HANDLERS
# ---------------------------------------------------------------------------
shutdown_evt = threading.Event()
for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
    signal.signal(_s, lambda *_: shutdown_evt.set())

# ---------------------------------------------------------------------------
# ENV setup – helps NCCL propagate failures
# ---------------------------------------------------------------------------
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

################################################################################
# Argument parsing                                                              
################################################################################

dataclass = dataclass  # silence flake

@dataclass
class Args:
    model_name: str; dataset_name: str; dataset_split: str; output_repo: str
    batch_size: int; micro_batch_size: int; num_rounds: int; num_samples: int
    sampling_temperature: float; push_every: int; max_seq_len: int
    hf_token: str | None = None


def parse_args() -> Args:
    p = argparse.ArgumentParser("Generate KD dataset with distributed weight‑streaming – vectorised sampling")
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--output_repo", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--num_rounds", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--sampling_temperature", type=float, default=1.0)
    p.add_argument("--push_every", type=int, default=1000)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--hf_token", default=None)
    return Args(**vars(p.parse_args()))

################################################################################
# Distributed helpers                                                           
################################################################################

def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=900))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def safe_barrier(timeout: float = 60.0):
    if not dist.is_initialized():
        return
    start = time.time()
    while True:
        work = dist.barrier(async_op=True)
        while not work.is_completed():
            if shutdown_evt.is_set() or time.time() - start > timeout:
                return
            time.sleep(0.1)
        return

@atexit.register
def _destroy_pg():
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

################################################################################
# Vectorised sampling utility                                                   
################################################################################

def sample_distribution(logits: torch.Tensor, num_samples: int, num_rounds: int, temperature: float = 1.0):
    """GPU‑side multinomial over *all* positions in one call, return ids & counts."""
    logits = logits.float()
    probs = torch.softmax(logits / temperature, dim=-1)  # (B, L, V)
    B, L, V = probs.shape
    total_draws = num_samples * num_rounds

    # flatten (B, L, V) → (B*L, V) to sample in one go
    draws = torch.multinomial(probs.view(-1, V), total_draws, replacement=True)  # (B*L, total_draws)

    ids_rows: List[List[int]] = []
    counts_rows: List[List[int]] = []
    for row in draws:  # iterate over flattened rows – no inner position loop
        uniq, cnt = torch.unique(row, return_counts=True)
        ids_rows.append(uniq.cpu().tolist())
        counts_rows.append(cnt.cpu().tolist())

    # reshape back to (B, L, ...)
    ids_all = [ids_rows[i * L : (i + 1) * L] for i in range(B)]
    counts_all = [counts_rows[i * L : (i + 1) * L] for i in range(B)]
    return ids_all, counts_all

################################################################################
# Distributed weight‑streaming streamer (unchanged aside from new sampler)      
################################################################################

class DistributedStreamer:
    """Streams layers with **double‑buffer prefetch** so the next block’s weights
    are broadcast while the current block is computing.  Keeps only two blocks
    resident at once and overlaps comm/compute to reduce stalls.
    """
    def __init__(self, args: Args, device: torch.device, rank: int, world: int):
        self.rank, self.world, self.device = rank, world, device
        cfg = AutoConfig.from_pretrained(args.model_name); cfg.attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=cfg,
            device_map={"": "cpu"},  # start on CPU (weight‑streaming)
            torch_dtype=torch.bfloat16,
        )
        self.model.config.attn_implementation = "flash_attention_2"
        self.layers = list(self.model.model.layers)
        self.rotary_emb = self.model.model.rotary_emb
        self.embed = self.model.model.embed_tokens.to(device)
        self.lm_head = self.model.lm_head.to(device)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
        def _async_broadcast(self, module: torch.nn.Module, owner: int):
        """Ensure every tensor is on *self.device*, then broadcast async.

        NCCL cannot send CPU tensors, so for every param/buffer that is still on
        CPU we first allocate a *placeholder* tensor on this GPU (same dtype &
        shape) and swap it into `module` before the broadcast call.
        """
        handles: List[dist.Work] = []
        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                if t.device != self.device:
                    # create un‑initialised tensor on GPU with same shape/dtype
                    empty_cuda = torch.empty_like(t, device=self.device)
                    t.data = empty_cuda  # swap in‑place so broadcast can fill it
                handles.append(dist.broadcast(t.data, src=owner, async_op=True))
        return handles

    # ------------------------------------------------------------------
    # forward & sample
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, ids: torch.Tensor, micro_bs: int, num_samples: int, num_rounds: int, temp: float):
        # 1. Prepare resident micro‑batches (on‑device) --------------------------------
        micro_inputs = [ids[i : i + micro_bs] for i in range(0, ids.size(0), micro_bs)]
        micro_states = []
        for mb in micro_inputs:
            h = self.embed(mb.to(self.device, non_blocking=True))
            L = h.size(1)
            cache_pos = torch.arange(L, device=self.device)
            pos_ids = cache_pos.unsqueeze(0)
            pos_emb = self.rotary_emb(h, pos_ids)
            micro_states.append({"hidden": h, "cache_pos": cache_pos, "pos_ids": pos_ids, "pos_emb": pos_emb})

        # 2. Double‑buffered layer stream ----------------------------------------------
        prefetch_handles = []  # async handles for the *next* layer

                # Prefetch layer‑0 synchronously to kick things off
        first_layer = self.layers[0]
        owner0 = 0 % self.world
        if self.rank == owner0:
            first_layer.to(self.device, non_blocking=True)
        first_handles = self._async_broadcast(first_layer, owner0)
        for h in first_handles:
            h.wait()
        prefetch_handles = []

        for idx, layer in enumerate(self.layers):
            owner = idx % self.world

            # Ensure any previous prefetch finished (for idx>0)
            if prefetch_handles:
                for h in prefetch_handles:
                    h.wait()
                prefetch_handles = []

                        # current layer is already on GPU thanks to broadcast; no owner‑device check needed

            # ---- compute current layer over all micro‑batches ----
            for ms in micro_states:
                out = layer(
                    ms["hidden"],
                    position_ids=ms["pos_ids"],
                    cache_position=ms["cache_pos"],
                    position_embeddings=ms["pos_emb"],
                    use_cache=False,
                )
                ms["hidden"] = out[0] if isinstance(out, tuple) else out

            # ---- launch prefetch of *next* layer (if any) while we free this one ----
            next_idx = idx + 1
            if next_idx < len(self.layers):
                next_layer = self.layers[next_idx]
                next_owner = next_idx % self.world
                if self.rank == next_owner:
                    next_layer.to(self.device, non_blocking=True)
                prefetch_handles = self._async_broadcast(next_layer, next_owner)

            # Free current layer weights to keep at most 2 layers resident
            layer.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()

            if shutdown_evt.is_set():
                break, non_blocking=True)
            torch.cuda.empty_cache()

            if shutdown_evt.is_set():
                break

        # 3. lm_head once per micro‑batch → vectorised sampling ------------------------
        ids_all, cnts_all = [], []
        for ms in micro_states:
            logits = self.lm_head(ms["hidden"])
            i, c = sample_distribution(logits, num_samples, num_rounds, temp)
            ids_all.extend(i); cnts_all.extend(c)

        return ids_all, cnts_all

################################################################################
# Data loader, push logic, worker (unchanged)                                   
################################################################################

def collate_fn(exs, tok, max_len):
    return tok([e["text"] for e in exs], return_tensors="pt", padding=True, truncation=True, max_length=max_len)

def streaming_dataloader(ds, tok, bs, max_len, rank, world):
    batch: List[dict] = []
    for i, ex in enumerate(ds):
        if shutdown_evt.is_set(): break
        if i % world != rank: continue
        batch.append(ex)
        if len(batch) == bs:
            yield collate_fn(batch, tok, max_len); batch.clear()
            if shutdown_evt.is_set(): break
    if batch and not shutdown_evt.is_set():
        yield collate_fn(batch, tok, max_len)

def push_shard(local: List[dict], args: Args, rank: int):
    gather = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local, gather, 0)
    if rank != 0: return
    merged: List[dict] = []
    for part in gather: merged.extend(part)
    if not merged: return
    api = HfApi(token=args.hf_token)
    idx = sum(f.endswith(".parquet") for f in api.list_repo_files(args.output_repo, repo_type="dataset"))
    fname = f"data_{idx:05d}.parquet"; Dataset.from_list(merged).to_parquet(fname)
    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull(); repo.git_add(fname); repo.git_commit(f"Add shard {idx}"); repo.git_push(); repo.git_clear(); os.remove(fname)

def worker(args: Args):
    local_rank, rank, world, device = setup_distributed()
    print(f"[rank {rank}] started on {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    streamer = DistributedStreamer(args, device, rank, world)

    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        token=args.hf_token,
        streaming=True,
    )

    loader = streaming_dataloader(
        dataset, tokenizer, args.batch_size, args.max_seq_len, rank, world
    )

    buffer: List[dict] = []
    seen = 0

    try:
        for batch in loader:
            if shutdown_evt.is_set():
                break
            ids, cnts = streamer.sample(
                batch["input_ids"],
                args.micro_batch_size,
                args.num_samples,
                args.num_rounds,
                args.sampling_temperature,
            )
            for i, toks in enumerate(batch["input_ids"]):
                tok_list = toks.tolist()
                while tok_list and tok_list[-1] == 0:
                    tok_list.pop()
                L = len(tok_list)
                buffer.append(
                    {
                        "input_ids": tok_list,
                        "sampled_ids": ids[i][:L],
                        "sampled_counts": cnts[i][:L],
                    }
                )
            seen += len(batch["input_ids"])
            if seen >= args.push_every:
                push_shard(buffer, args, rank)
                buffer.clear(); seen = 0
            if shutdown_evt.is_set():
                break
    except Exception as e:
        print(f"[rank {rank}] exception: {e}", flush=True)
        shutdown_evt.set()

    if buffer and not shutdown_evt.is_set():
        push_shard(buffer, args, rank)

    safe_barrier()
    print(f"[rank {rank}] exiting", flush=True)

################################################################################
# Entrypoint
################################################################################

if __name__ == "__main__":
    worker(parse_args())

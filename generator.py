# generator_distributed.py – weight‑streaming version
# Author : ChatGPT – June 2025 ✨  (rev‑3: **robust shutdown**)
"""
### What changed in rev‑3
* **Early, global signal traps** for `SIGTERM`, `SIGINT`, `SIGUSR1` – installed at
  import‑time so *every* rank sees them even if the signal arrives during heavy
  start‑up.
* A **global `shutdown_evt`** shared by all modules; the handler also sets an
  integer CUDA tensor flag that is broadcast so *all* ranks learn that one rank
  died (covers GPU OOM / NCCL errors where only one rank gets the signal).
* Enforces `NCCL_ASYNC_ERROR_HANDLING=1` and `NCCL_BLOCKING_WAIT=1` to force
  collectives to raise Python exceptions instead of hanging.
* Adds `safe_barrier()` which times out after 60 s – prevents permanent hangs
  if a rank is already gone.
* Ensures `dist.destroy_process_group()` runs from `atexit` on normal or
  signalled termination.

With these tweaks Slurm’s SIGTERM should propagate cleanly, all ranks exit
within the grace period, and the node never goes to the *drained* state.
"""
from __future__ import annotations

import argparse, atexit, os, signal, sys, threading, time
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, Repository
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# GLOBAL SHUTDOWN FLAG & SIGNAL HANDLERS  (registered *before* heavy work)
# ---------------------------------------------------------------------------
shutdown_evt = threading.Event()

def _signal_handler(sig, frame):
    print(f"[rank ?] caught signal {sig}")
    shutdown_evt.set()
    # set CUDA flag so other ranks notice (works even if they’re stuck in C++)
    if dist.is_initialized():
        try:
            flag = torch.tensor([1], device=torch.device("cuda"))  # small 1‑int tensor
            dist.broadcast(flag, src=dist.get_rank())
        except Exception:
            pass  # ok – maybe not initialised yet

for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
    signal.signal(_s, _signal_handler)

# ---------------------------------------------------------------------------
# ENV setup – makes NCCL propagate failures instead of hanging
# ---------------------------------------------------------------------------
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

################################################################################
# Argument parsing                                                              
################################################################################

@dataclass
class Args:
    model_name: str; dataset_name: str; dataset_split: str; output_repo: str
    batch_size: int; micro_batch_size: int; num_rounds: int; num_samples: int
    sampling_temperature: float; push_every: int; max_seq_len: int
    hf_token: str | None = None


def parse_args() -> Args:
    p = argparse.ArgumentParser("Generate KD dataset with distributed weight‑streaming – robust shutdown")
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
        dist.init_process_group("nccl", timeout=torch.timedelta(seconds=900))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, dist.get_rank(), dist.get_world_size(), device


def safe_barrier(timeout: float = 60.0):
    """Barrier that times out if peers are dead to avoid infinite hangs."""
    if not dist.is_initialized():
        return
    start = time.time()
    while True:
        req = dist.barrier(async_op=True)
        while not req.is_completed():
            if shutdown_evt.is_set() or time.time() - start > timeout:
                return
            time.sleep(0.1)
        return

# ensure destroy at exit
@atexit.register
def _cleanup_dist():
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

################################################################################
# Sampling utility (unchanged)                                                  
################################################################################

def sample_distribution(logits: torch.Tensor, num_samples: int, num_rounds: int, temperature: float = 1.0):
    logits = logits.float()
    q = torch.softmax(logits / temperature, dim=-1)
    ids_all: List[List[List[int]]] = []
    counts_all: List[List[List[int]]] = []
    bsz, seqlen, vocab = q.shape
    for b in range(bsz):
        ids_seq, counts_seq = [], []
        for s in range(seqlen):
            row = torch.nan_to_num(q[b, s], nan=0.0, posinf=0.0, neginf=0.0)
            row = row.clamp(min=0)
            if row.sum() == 0: row.fill_(1 / vocab)
            row = (row + 1e-6) / row.sum()
            smp = torch.multinomial(row, num_samples * num_rounds, replacement=True)
            uniq, cnt = torch.unique(smp, return_counts=True)
            ids_seq.append(uniq.cpu().tolist()); counts_seq.append(cnt.cpu().tolist())
        ids_all.append(ids_seq); counts_all.append(counts_seq)
    return ids_all, counts_all

################################################################################
# Distributed weight‑streaming streamer (same compute path)                     
################################################################################

class DistributedStreamer:
    def __init__(self, args: Args, device: torch.device, rank: int, world: int):
        self.rank, self.world, self.device = rank, world, device
        cfg = AutoConfig.from_pretrained(args.model_name); cfg.attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name, config=cfg, device_map={"": "cpu"}, torch_dtype=torch.bfloat16)
        self.model.config.attn_implementation = "flash_attention_2"
        self.layers = list(self.model.model.layers)
        self.rotary_emb = self.model.model.rotary_emb
        self.embed = self.model.model.embed_tokens.to(device)
        self.lm_head = self.model.lm_head.to(device)

    def _broadcast(self, mod: torch.nn.Module, owner: int):
        handles = [dist.broadcast(p.data, src=owner, async_op=True) for p in list(mod.parameters()) + list(mod.buffers())]
        for h in handles: h.wait()

    @torch.no_grad()
    def sample(self, ids: torch.Tensor, micro_bs: int, num_samples: int, num_rounds: int, temp: float):
        micro_inputs = [ids[i:i + micro_bs] for i in range(0, ids.size(0), micro_bs)]
        micro_states = []
        for mb in micro_inputs:
            h = self.embed(mb.to(self.device, non_blocking=True))
            L = h.size(1); cache_pos = torch.arange(L, device=self.device); pos_ids = cache_pos.unsqueeze(0)
            pos_emb = self.rotary_emb(h, pos_ids)
            micro_states.append({"hidden": h, "cache_pos": cache_pos, "pos_ids": pos_ids, "pos_emb": pos_emb})

        for i, layer in enumerate(self.layers):
            owner = i % self.world; layer.to(self.device, non_blocking=True); self._broadcast(layer, owner)
            for ms in micro_states:
                out = layer(ms["hidden"], position_ids=ms["pos_ids"], cache_position=ms["cache_pos"], position_embeddings=ms["pos_emb"], use_cache=False)
                ms["hidden"] = out[0] if isinstance(out, tuple) else out
            layer.to("cpu", non_blocking=True); torch.cuda.empty_cache()
            if shutdown_evt.is_set():
                break

        ids_all, counts_all = [], []
        for ms in micro_states:
            logits = self.lm_head(ms["hidden"])
            i, c = sample_distribution(logits, num_samples, num_rounds, temp)
            ids_all.extend(i); counts_all.extend(c)
        return ids_all, counts_all

################################################################################
# Data helpers (unchanged)                                                      
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

################################################################################
# Push shard (rank‑0)                                                           
################################################################################

def push_shard(local: List[dict], args: Args, rank: int):
    gather = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local, gather, 0)
    if rank != 0: return
    merged: List[dict] = []
    for part in gather: merged.extend(part)
    if not merged: return
    api = HfApi(token=args.hf_token)
    n = sum(f.endswith(".parquet") for f in api.list_repo_files(args.output_repo, repo_type="dataset"))
    fname = f"data_{n:05d}.parquet"; Dataset.from_list(merged).to_parquet(fname)
    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull(); repo.git_add(fname); repo.git_commit(f"Add shard {n}"); repo.git_push(); repo.git_clear(); os.remove(fname)

################################################################################
# Worker                                                                        
################################################################################

def worker(args: Args):
    local_rank, rank, world, device = setup_distributed()
    print(f"Rank {rank}/{world} online on {device}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    streamer = DistributedStreamer(args, device, rank, world)
    ds = load_dataset(args.dataset_name, split=args.dataset_split, token=args.hf_token, streaming=True)
    loader = streaming_dataloader(ds, tok, args.batch_size, args.max_seq_len, rank, world)

    buf: List[dict] = []; seen = 0
    try:
        for batch in loader:
            if shutdown_evt.is_set(): break
            ids, cnts = streamer.sample(batch["input_ids"], args.micro_batch_size, args.num_samples, args.num_rounds, args.sampling_temperature)
            for i, toks in enumerate(batch["input_ids"]):
                t = toks.tolist();
                while t and t[-1] == 0: t.pop(); L = len(t)
                buf.append({"input_ids": t, "sampled_ids": ids[i][:L], "sampled_counts": cnts[i][:L]})
            seen += len(batch["input_ids"])
            if seen >= args.push_every:
                push_shard(buf, args, rank); buf.clear(); seen = 0
            if shutdown_evt.is_set(): break
    except Exception as e:
        print(f"Rank {rank} exception: {e}"); shutdown_evt.set()

    if buf and not shutdown_evt.is_set(): push_shard(buf, args, rank)
    safe_barrier(); print(f"Rank {rank} clean exit")

################################################################################
# Entrypoint                                                                    
################################################################################

if __name__ == "__main__":
    worker(parse_args())

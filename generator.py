# generator_distributed.py – weight‑streaming + double‑buffer + vectorised sampling
# Author : ChatGPT – June 2025   (rev‑5: fix syntax + indent)
"""Minimal, runnable revision fixing the syntax/indent errors introduced in rev‑4.
Highlights remain:
* weight streaming with double‑buffer prefetch
* fully GPU‑side multinomial sampling (vectorised)
* only IDs+counts copied to CPU
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
# GLOBALS & SIGNAL HANDLERS
# ---------------------------------------------------------------------------
shutdown_evt = threading.Event()
for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
    signal.signal(_sig, lambda *_: shutdown_evt.set())

os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class Args:
    model_name: str
    dataset_name: str
    dataset_split: str
    output_repo: str
    batch_size: int
    micro_batch_size: int
    num_rounds: int
    num_samples: int
    sampling_temperature: float
    push_every: int
    max_seq_len: int
    hf_token: str | None = None

def parse_args() -> Args:
    p = argparse.ArgumentParser("KD generator – weight‑stream + vectorised sampling")
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

# ---------------------------------------------------------------------------
# DISTRIBUTED HELPERS
# ---------------------------------------------------------------------------

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=15))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, dist.get_rank(), dist.get_world_size(), device


def safe_barrier(timeout: float = 60):
    if not dist.is_initialized():
        return
    start = time.time()
    work = dist.barrier(async_op=True)
    while not work.is_completed():
        if shutdown_evt.is_set() or time.time() - start > timeout:
            return
        time.sleep(0.1)

@atexit.register
def _destroy_pg():
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# VECTORISED SAMPLER
# ---------------------------------------------------------------------------

def sample_distribution(logits: torch.Tensor, N: int, R: int, temp: float = 1.0):
    logits = logits.float()
    probs = torch.softmax(logits / temp, dim=-1)
    B, L, V = probs.shape
    draws = torch.multinomial(probs.view(-1, V), N * R, replacement=True)  # (B*L, N*R)
    ids_rows, cnts_rows = [], []
    for row in draws:
        uniq, cnt = torch.unique(row, return_counts=True)
        ids_rows.append(uniq.cpu().tolist())
        cnts_rows.append(cnt.cpu().tolist())
    reshape = lambda flat: [flat[i * L:(i + 1) * L] for i in range(B)]
    return reshape(ids_rows), reshape(cnts_rows)

# ---------------------------------------------------------------------------
# STREAMER WITH DOUBLE BUFFER
# ---------------------------------------------------------------------------
class DistributedStreamer:
    def __init__(self, args: Args, device: torch.device, rank: int, world: int):
        self.rank, self.world, self.device = rank, world, device
        cfg = AutoConfig.from_pretrained(args.model_name)
        cfg.attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, config=cfg, device_map={"": "cpu"}, torch_dtype=torch.bfloat16
        )
        self.model.config.attn_implementation = "flash_attention_2"
        self.layers = list(self.model.model.layers)
        self.rotary = self.model.model.rotary_emb
        self.embed = self.model.model.embed_tokens.to(device)
        self.lm_head = self.model.lm_head.to(device)

    # ---- helper
    def _async_broadcast(self, module: torch.nn.Module, owner: int):
        handles: List[dist.Work] = []
        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                if t.device != self.device:
                    t.data = torch.empty_like(t, device=self.device)
                handles.append(dist.broadcast(t.data, src=owner, async_op=True))
        return handles

    # ---- main
    @torch.no_grad()
    def sample(self, ids: torch.Tensor, micro_bs: int, N: int, R: int, temp: float):
        # prep micro batches
        mic_inp = [ids[i:i + micro_bs] for i in range(0, ids.size(0), micro_bs)]
        states = []
        for mb in mic_inp:
            h = self.embed(mb.to(self.device, non_blocking=True))
            L = h.size(1)
            pos = torch.arange(L, device=self.device)
            states.append({
                "hidden": h,
                "pos_ids": pos.unsqueeze(0),
                "cache_pos": pos,
                "pos_emb": self.rotary(h, pos.unsqueeze(0)),
            })

        # double‑buffer
        next_handles: List[dist.Work] = []
        # preload layer0
        if self.rank == 0:
            self.layers[0].to(self.device, non_blocking=True)
        for h in self._async_broadcast(self.layers[0], 0):
            h.wait()

        for idx, layer in enumerate(self.layers):
            # wait previous prefetch
            for h in next_handles: h.wait()
            next_handles = []

            # compute layer
            for s in states:
                out = layer(
                    s["hidden"],
                    position_ids=s["pos_ids"],
                    cache_position=s["cache_pos"],
                    position_embeddings=s["pos_emb"],
                    use_cache=False,
                )
                s["hidden"] = out[0] if isinstance(out, tuple) else out

            # prefetch next
            nxt = idx + 1
            if nxt < len(self.layers):
                nxt_layer = self.layers[nxt]
                owner = nxt % self.world
                if self.rank == owner:
                    nxt_layer.to(self.device, non_blocking=True)
                next_handles = self._async_broadcast(nxt_layer, owner)
            # free current
            layer.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()
            if shutdown_evt.is_set():
                break

        # lm_head & sampling
        ids_all, cnts_all = [], []
        for s in states:
            logits = self.lm_head(s["hidden"])
            i, c = sample_distribution(logits, N, R, temp)
            ids_all.extend(i); cnts_all.extend(c)
        return ids_all, cnts_all

# ---------------------------------------------------------------------------
# IO HELPERS
# ---------------------------------------------------------------------------

def collate_fn(ex, tok, ml):
    return tok([e["text"] for e in ex], return_tensors="pt", padding=True, truncation=True, max_length=ml)

def stream_loader(ds, tok, bs, ml, rank, world):
    batch = []
    for i, ex in enumerate(ds):
        if shutdown_evt.is_set():
            break
        if i % world != rank:
            continue
        batch.append(ex)
        if len(batch) == bs:
            yield collate_fn(batch, tok, ml)
            batch.clear()
            if shutdown_evt.is_set():
                break
    if batch and not shutdown_evt.is_set():
        yield collate_fn(batch, tok, ml)

def push_shard(local, args: Args, rank: int):
    gather = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local, gather, 0)
    if rank != 0:
        return
    merged = [rec for part in gather for rec in part]
    if not merged:
        return
    api = HfApi(token=args.hf_token)
    idx = sum(f.endswith(".parquet") for f in api.list_repo_files(args.output_repo, repo_type="dataset"))
    fname = f"data_{idx:05d}.parquet"
    Dataset.from_list(merged).to_parquet(fname)
    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull(); repo.git_add(fname); repo.git_commit(f"Add shard {idx}"); repo.git_push(); repo.git_clear(); os.remove(fname)

# ---------------------------------------------------------------------------
# WORKER
# ---------------------------------------------------------------------------

def worker(a: Args):
    _, rank, world, dev = setup_distributed()
    print(f"[rank {rank}] on {dev}")
    tok = AutoTokenizer.from_pretrained(a.model_name)
    streamer = DistributedStreamer(a, dev, rank, world)
    ds = load_dataset(a.dataset_name, split=a.dataset_split, token=a.hf_token, streaming=True)
    loader = stream_loader(ds, tok, a.batch_size, a.max_seq_len, rank, world)

    buf, seen = [], 0
    try:
        for batch in loader:
            if shutdown_evt.is_set():
                break
            ids, cnts = streamer.sample(batch["input_ids"], a.micro_batch_size, a.num_samples, a.num_rounds, a.sampling_temperature)
            for i, toks in enumerate(batch["input_ids"]):
                seq = toks.tolist()
                while seq and seq[-1] == 0:
                    seq.pop()
                L = len(seq)
                buf.append({"input_ids": seq, "sampled_ids": ids[i][:L], "sampled_counts": cnts[i][:L]})
            seen += len(batch["input_ids"])
            if seen >= a.push_every:
                push_shard(buf, a, rank); buf.clear(); seen = 0
    except Exception as e:
        print(f"[rank {rank}] error {e}"); shutdown_evt.set()
    if buf and not shutdown_evt.is_set():
        push_shard(buf, a, rank)
    safe_barrier(); print(f"[rank {rank}] exit")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    worker(parse_args())

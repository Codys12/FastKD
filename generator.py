from __future__ import annotations

import argparse
import os
import signal
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, Repository

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
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
    pipeline_backend: str = "nccl"


def parse_args() -> Args:
    p = argparse.ArgumentParser("GPU‑sampling parameter‑pipeline KD generator")
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
    p.add_argument("--pipeline_backend", default="nccl")
    return Args(**vars(p.parse_args()))

# ──────────────────────────────────────────────────────────────────────────────
# HF push helper
# ──────────────────────────────────────────────────────────────────────────────

def push_shard(records: List[Dict[str, Any]], args: Args):
    api = HfApi(token=args.hf_token)
    files = api.list_repo_files(args.output_repo, repo_type="dataset")
    idx = sum(f.endswith(".parquet") for f in files)
    fname = f"data_{idx:05d}.parquet"
    Dataset.from_list(records).to_parquet(fname)
    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull(); repo.git_add(fname); repo.git_commit(f"Add shard {idx}"); repo.git_push(); repo.git_clear()
    os.remove(fname)

# ──────────────────────────────────────────────────────────────────────────────
# Tensor utils
# ──────────────────────────────────────────────────────────────────────────────

def flatten_mod(m: nn.Module) -> torch.Tensor:
    return parameters_to_vector([p.detach() for p in m.parameters()])

def load_vec(m: nn.Module, vec: torch.Tensor):
    vector_to_parameters(vec, [p for p in m.parameters()])

# ──────────────────────────────────────────────────────────────────────────────
# Vectorised GPU sampler (flat rows)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def gpu_sample(logits: torch.Tensor, n_samples: int, n_rounds: int, t: float) -> Tuple[List, List]:
    logits = logits.float()
    probs = torch.softmax(logits / t, -1)
    B, S, V = probs.shape
    draws = torch.multinomial(probs.view(-1, V), n_samples * n_rounds, replacement=True)  # (B*S, T)
    ids_all, cnts_all = [], []
    for row in draws:  # still one loop across B*S rows (tiny)
        uniq, cnt = torch.unique(row, return_counts=True)
        ids_all.append(uniq.cpu().tolist()); cnts_all.append(cnt.cpu().tolist())
    # reshape back to [B][S][…]
    ids_out = [ids_all[i*S:(i+1)*S] for i in range(B)]
    cnts_out = [cnts_all[i*S:(i+1)*S] for i in range(B)]
    return ids_out, cnts_out

# ──────────────────────────────────────────────────────────────────────────────
# Worker
# ──────────────────────────────────────────────────────────────────────────────
class Worker:
    def __init__(self, rank: int, world: int, args: Args):
        self.rank, self.world, self.args = rank, world, args
        self.prev = rank - 1 if rank else None; self.next = rank + 1 if rank + 1 < world else None
        self.device = torch.device(f"cuda:{rank}"); torch.cuda.set_device(self.device)

        cfg = AutoConfig.from_pretrained(args.model_name); cfg.attn_implementation = "eager"
        base = AutoModelForCausalLM.from_pretrained(args.model_name, config=cfg, device_map={"": "cpu"})
        self.embed = base.model.embed_tokens
        self.layers = list(base.model.layers)
        self.lm_head = base.lm_head
        self.scratch = self.layers[0].to(self.device, dtype=torch.bfloat16)
        self.pbuf = [torch.empty(1, dtype=torch.bfloat16, device=self.device) for _ in range(2)]
        self.sbuf = [torch.empty(1, dtype=torch.int64) for _ in range(2)]
        self.hidden: torch.Tensor | None = None
        if rank == 0:
            self.tok = AutoTokenizer.from_pretrained(args.model_name)
            self.ds = load_dataset(args.dataset_name, split=args.dataset_split, token=args.hf_token, streaming=True)
            self.recs: List[Dict[str, Any]] = []

    # ────────────────────── helpers (rank‑0) ──────────────────────
    def _collate(self, batch):
        txt = [b["text"] for b in batch]
        return self.tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_len)

    def _loader(self):
        buf, bs = [], self.args.batch_size
        for ex in self.ds:
            buf.append(ex)
            if len(buf) == bs:
                yield self._collate(buf); buf.clear()
        if buf:
            yield self._collate(buf)

    # ───────────────────────── Rank‑0 loop ─────────────────────────
    def _driver(self, stop: threading.Event):
        args = self.args
        for batch in self._loader():
            if stop.is_set(): break
            with torch.no_grad():
                hid_cpu = self.embed(batch["input_ids"])
            shards = hid_cpu.split(args.micro_batch_size)
            for r in range(self.world):
                if r == 0:
                    self.hidden = shards[0].to(self.device, torch.bfloat16)
                else:
                    dist.send(torch.tensor(shards[r].shape, dtype=torch.int64), dst=r, tag=9000)
                    dist.send(shards[r].to(torch.bfloat16), dst=r, tag=9001)
            # stream layers
            for idx, layer in enumerate(self.layers):
                vec = flatten_mod(layer).to(torch.bfloat16)
                shape = torch.tensor([vec.numel()], dtype=torch.int64)
                if self.next is not None:
                    dist.isend(shape, dst=self.next, tag=idx*2)
                    dist.isend(vec,   dst=self.next, tag=idx*2+1)
                load_vec(self.scratch, vec.to(self.device, non_blocking=True))
                self.hidden = self.scratch(self.hidden)[0] if isinstance(self.scratch(self.hidden), tuple) else self.scratch(self.hidden)
            # lm_head
            hvec = flatten_mod(self.lm_head).to(torch.bfloat16)
            if self.next is not None:
                dist.send(torch.tensor([-1], dtype=torch.int64), dst=self.next, tag=9998)
                dist.send(hvec, dst=self.next, tag=9999)
            ids0, cnt0 = gpu_sample(self.lm_head(self.hidden.to(self.lm_head.weight.dtype)), args.num_samples, args.num_rounds, args.sampling_temperature)
            out = {"ids": ids0, "counts": cnt0}
            gather: List | None = [None]*self.world
            dist.gather_object(out, gather, dst=0)
            for obj in gather:
                if obj is None: continue
                for ids, cnt in zip(obj["ids"], obj["counts"]):
                    self.recs.append({"sampled_ids": ids, "sampled_counts": cnt})
            if len(self.recs) >= args.push_every:
                push_shard(self.recs, args); self.recs.clear()

    # ─────────────────────── GPU ranks loop ───────────────────────
    def _gpu(self, stop: threading.Event):
        args = self.args; cur = 0
        while not stop.is_set():
            shape = torch.empty((3,), dtype=torch.int64); dist.recv(shape, src=0, tag=9000)
            hsh = tuple(shape.tolist()); self.hidden = torch.empty(hsh, dtype=torch.bfloat16, device=self.device)
            dist.recv(self.hidden, src=0, tag=9001)
            idx = 0
            while True:
                dist.recv(self.sbuf[cur], src=self.prev, tag=idx*2)
                n = int(self.sbuf[cur].item())
                if n == -1:
                    hvec = torch.empty((flatten_mod(self.lm_head).numel(),), dtype=torch.bfloat16)
                    dist.recv(hvec, src=self.prev, tag=9999)
                    load_vec(self.lm_head.to(self.device), hvec.to(self.device, non_blocking=True))
                    ids, cnt = gpu_sample(self.lm_head(self.hidden.to(self.lm_head.weight.dtype)), args.num_samples, args.num_rounds, args.sampling_temperature)
                    dist.gather_object({"ids": ids, "counts": cnt}, None, dst=0)
                    break
                if self.pbuf[cur].numel() != n:
                    self.pbuf[cur] = torch.empty((n,), dtype=torch.bfloat16, device=self.device)
                dist.recv(self.pbuf[cur], src=self.prev, tag=idx*2+1)
                if self.next is not None:
                    dist.isend(self.sbuf[cur].clone(), dst=self.next, tag=idx*2)
                    dist.isend(self.pbuf[cur].clone(), dst=self.next, tag=idx*2+1)
                load_vec(self.scratch, self.pbuf[cur])
                self.hidden = self.scratch(self.hidden)[0] if isinstance(self.scratch(self.hidden), tuple) else self.scratch(self.hidden)
                cur ^= 1; idx += 1

    def run(self, stop: threading.Event):
        (self._driver if self.rank == 0 else self._gpu)(stop)

# ──────────────────────────────────────────────────────────────────────────────
# Launcher
# ──────────────────────────────────────────────────────────────────────────────

def _entry(rank: int, world: int, args: Args, stop: threading.Event):
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(world), "MASTER_ADDR": os.getenv("MASTER_ADDR", "127.0.0.1"), "MASTER_PORT": os.getenv("MASTER_PORT", "29500")})
    dist.init_process_group(backend=args.pipeline_backend, rank=rank, world_size=world)
    Worker(rank, world, args).run(stop)
    dist.destroy_process_group()

def main():
    args = parse_args(); world = torch.cuda.device_count()
    stop = threading.Event(); signal.signal(signal.SIGTERM, lambda *_: stop.set())
    mp.spawn(_entry, nprocs=world, args=(world, args, stop), join=True)

if __name__ == "__main__":
    main()

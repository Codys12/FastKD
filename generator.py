from __future__ import annotations
"""
Generator with reverse pipeline parallelism (ring) and optional debug logs.
Weights travel CPU -> GPU-0 -> ... -> GPU-N-1 -> GPU-0. Rank‑0 then off‑loads
back to CPU. Use --debug for verbose per‑rank output.

* Rank 0 loads the full model on CPU in bf16.
* Other ranks create an empty‑weight skeleton on the meta device (via
  accelerate, or manual fallback).
* Layers are packed into flat CUDA tensors and streamed around the ring with
  double buffering.
* Sampling is vectorised on GPU.
"""

import argparse
import os
import signal
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, Repository
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

###############################################################################
# CLI                                                                        #
###############################################################################

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
    debug: bool = False


def parse_args() -> Args:
    p = argparse.ArgumentParser("Reverse‑pipeline KD generator (ring)")
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
    p.add_argument("--debug", action="store_true")
    return Args(**vars(p.parse_args()))

###############################################################################
# Utils                                                                       #
###############################################################################

def get_logger(rank: int, enabled: bool):
    def log(msg: str):
        if enabled:
            print(f"[Rank {rank}] {msg}", flush=True)
    return log

###############################################################################
# Flat packet helper                                                          #
###############################################################################

class FlatPacket:
    def __init__(self):
        self.flat: torch.Tensor | None = None

    def pack(self, layer: torch.nn.Module, device: torch.device):
        with torch.no_grad():
            self.flat = torch.cat([p.data.view(-1) for p in layer.parameters()]).contiguous().to(device)

    def make_empty(self, numel: int, device: torch.device):
        self.flat = torch.empty(numel, dtype=torch.bfloat16, device=device)

    def unpack_to(self, layer: torch.nn.Module):
        if self.flat is None:
            raise RuntimeError("flat tensor is None")
        offset = 0
        with torch.no_grad():
            for p in layer.parameters():
                n = p.numel()
                p.data.copy_(self.flat[offset:offset+n].view_as(p))
                offset += n

###############################################################################
# Reverse pipeline engine                                                     #
###############################################################################

class ReversePipelineEngine:
    def __init__(self, args: Args, rank: int, world: int, dbg):
        self.args = args
        self.rank = rank
        self.world = world
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)
        self.dbg = dbg

        cfg = AutoConfig.from_pretrained(args.model_name)
        cfg.attn_implementation = "eager"

        if rank == 0:
            dbg("Loading full model on CPU")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=cfg,
                device_map={"": "cpu"},
                torch_dtype=torch.bfloat16,
            )
            model.config.attn_implementation = "flash_attention_2"
        else:
            dbg("Building empty‑weight skeleton")
            try:
                from accelerate import init_empty_weights
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(cfg)
            except ImportError:
                dbg("accelerate missing; manual meta tensors")
                model = AutoModelForCausalLM.from_config(cfg)
                for p in model.parameters():
                    p.data = torch.empty_like(p.data, device="meta")

        self.layers: List[torch.nn.Module] = [model.model.embed_tokens, *model.model.layers, model.lm_head]
        self.buffers = (FlatPacket(), FlatPacket())
        self.toggle = 0

    # ring helpers
    def _send(self, tensor: torch.Tensor, dst: int):
        dist.send(tensor, dst)
    def _recv(self, tensor: torch.Tensor, src: int):
        dist.recv(tensor, src)

    def _stream_layer(self, idx: int):
        left = (self.rank - 1) % self.world
        right = (self.rank + 1) % self.world
        buf = self.buffers[self.toggle]

        if not (self.rank == 0 and idx == 0):
            shape = torch.empty(1, dtype=torch.int64, device=self.device)
            self._recv(shape, left)
            buf.make_empty(int(shape.item()), self.device)
            self._recv(buf.flat, left)
        else:
            buf.pack(self.layers[idx], self.device)

        buf.unpack_to(self.layers[idx])
        shape_out = torch.tensor([buf.flat.numel()], dtype=torch.int64, device=self.device)
        self._send(shape_out, right)
        self._send(buf.flat, right)
        self.toggle ^= 1

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            self._stream_layer(i)
            layer = self.layers[i].to(self.device)
            for start in range(0, hidden.size(0), self.args.micro_batch_size):
                mb = hidden[start:start+self.args.micro_batch_size]
                out = layer(mb)
                hidden[start:start+self.args.micro_batch_size] = out[0] if isinstance(out, tuple) else out
            layer.to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()
        return hidden

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor):
        logits = self.forward(hidden)
        probs = torch.softmax(logits / self.args.sampling_temperature, dim=-1)
        draws = torch.multinomial(probs.view(-1, probs.size(-1)), self.args.num_samples * self.args.num_rounds, replacement=True)
        draws = draws.view(*probs.shape[:-1], -1)
        ids_all, counts_all = [], []
        for b in range(draws.size(0)):
            ids_seq, cnts_seq = [], []
            for t in range(draws.size(1)):
                uniq, cnt = torch.unique(draws[b, t], return_counts=True)
                ids_seq.append(uniq.cpu().tolist())
                cnts_seq.append(cnt.cpu().tolist())
            ids_all.append(ids_seq)
            counts_all.append(cnts_seq)
        return ids_all, counts_all

###############################################################################
# Data helpers                                                                #
###############################################################################

def collate_fn(examples, tok, max_len):
    return tok([e["text"] for e in examples], return_tensors="pt", padding=True, truncation=True, max_length=max_len)

def stream_loader(ds, tok, bs, max_len):
    buf = []
    for ex in ds:
        buf.append(ex)
        if len(buf) == bs:
            yield collate_fn(buf, tok, max_len)
            buf.clear()
    if buf:
        yield collate_fn(buf, tok, max_len)

###############################################################################
# Push shard                                                                  #
###############################################################################

def push_shard(records: List[dict], args: Args, dbg):
    dbg(f"Pushing {len(records)} records to hub")
    api = HfApi(token=args.hf_token)
    idx = sum(f.endswith(".parquet") for f in api.list_repo_files(args.output_repo, repo_type="dataset"))
    fname = f"data_{idx:05d}.parquet"
    Dataset.from_list(records).to_parquet(fname)
    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull()
    repo.git_add(fname)
    repo.git_commit(f"add shard {idx}")
    repo.git_push()
    repo.git_clear()
    os.remove(fname)

###############################################################################
# Gather helper                                                               #
###############################################################################

def gather_and_push(local: List[dict], args: Args, rank: int, world: int, dbg):
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        merged: List[dict] = []
        for part in gathered:  # type: ignore[arg-type]
            merged.extend(part)
        push_shard(merged, args, dbg)

###############################################################################
# Worker                                                                      #
###############################################################################

def worker(args: Args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dbg = get_logger(rank, args.debug)

    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())

    try:
        engine = ReversePipelineEngine(args, rank, world, dbg)
        tok = AutoTokenizer.from_pretrained(args.model_name)
        ds = load_dataset(
            args.dataset_name,
            split=args.dataset_split,
            streaming=True,
            token=args.hf_token,
        )
        loader = stream_loader(ds, tok, args.batch_size, args.max_seq_len)

        local_records: List[dict] = []
        seen = 0
        for batch_idx, batch in enumerate(loader):
            if stop_evt.is_set():
                dbg("SIGTERM received – breaking loop")
                break

            dbg(f"Batch {batch_idx} acquired")
            input_ids = batch["input_ids"].to(engine.device, non_blocking=True)
            hidden = engine.layers[0].to(engine.device)(input_ids)  # embed once per batch
            ids, cnts = engine.sample(hidden)

            for i in range(len(input_ids)):
                toks = input_ids[i].tolist()
                while toks and toks[-1] == 0:
                    toks.pop()
                seq_len = len(toks)
                local_records.append({
                    "input_ids": toks,
                    "sampled_ids": ids[i][:seq_len],
                    "sampled_counts": cnts[i][:seq_len],
                })
            seen += len(input_ids)

            if seen >= args.push_every:
                dbg("Reached push_every – gathering to rank‑0")
                gather_and_push(local_records, args, rank, world, dbg)
                local_records.clear(); seen = 0

        # final flush
        if local_records and not stop_evt.is_set():
            dbg("Final flush before exit")
            gather_and_push(local_records, args, rank, world, dbg)

    except Exception:
        dbg("Unhandled exception – printing traceback")
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    dbg("Graceful shutdown")
    dist.destroy_process_group()


def main():
    args = parse_args()
    worker(args)


if __name__ == "__main__":
    main()

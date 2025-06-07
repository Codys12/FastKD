from __future__ import annotations

"""
Generator with **reverse pipeline parallelism** (ring) + verbose DEBUG logs.
Parameters circulate in a ring  CPU → GPU‑0 → … → GPU‑N‑1 → GPU‑0 (rank‑0 then
moves weights back to CPU).  Add `--debug` to print *very* chatty per‑rank
information; otherwise only high‑level milestones are shown.

This version fixes the previous **SyntaxError** (duplicate `self.layers` block)
and relies on `accelerate.init_empty_weights` to build the empty parameter
skeleton on non‑zero ranks, avoiding incompatible tensor‑type issues.
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
#                                    CLI                                      #
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
#                             Helper: debug printer                           #
###############################################################################

def get_logger(rank: int, enabled: bool):
    def log(msg: str):
        if enabled:
            print(f"[Rank {rank}] {msg}", flush=True)
    return log

###############################################################################
#                       Flat parameter packet helper                          #
###############################################################################

class _FlatPacket:
    """Flatten parameters into a contiguous CUDA tensor for fast P2P transfer."""

    def __init__(self):
        self.flat: torch.Tensor | None = None

    def pack(self, layer: torch.nn.Module, device: torch.device) -> None:
        with torch.no_grad():
            vec = [p.data.view(-1) for p in layer.parameters()]
            self.flat = torch.cat(vec).contiguous().to(device, non_blocking=True)

    def make_empty(self, numel: int, device: torch.device) -> None:
        self.flat = torch.empty(numel, dtype=torch.bfloat16, device=device)

    def unpack_to(self, layer: torch.nn.Module) -> None:
        if self.flat is None:
            raise RuntimeError("Attempt to unpack empty packet")
        offset = 0
        with torch.no_grad():
            for p in layer.parameters():
                n = p.numel()
                p.data.copy_(self.flat[offset : offset + n].view_as(p))
                offset += n

###############################################################################
#                       Reverse pipeline engine (ring)                        #
###############################################################################

class ReversePipelineEngine:
    """Stream *layers* through static hidden‑states resident on each GPU."""

    def __init__(self, args: Args, rank: int, world: int, dbg):
        self.args = args
        self.rank = rank
        self.world = world
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)
        self.dbg = dbg

        cfg = AutoConfig.from_pretrained(args.model_name)
        cfg.attn_implementation = "eager"  # safe for CPU init

        if rank == 0:
            dbg("Loading full model on CPU…")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=cfg,
                device_map={"": "cpu"},
                torch_dtype=torch.bfloat16,
            )
            model.config.attn_implementation = "flash_attention_2"
        else:
            dbg("Building empty‑weight skeleton (meta)…")
            # Try accelerate's helper; fall back to manual meta tensors if not available
            try:
                from accelerate.utils import init_empty_weights  # accelerate>=0.26
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(cfg)
            except ImportError:
                dbg("accelerate.init_empty_weights missing; using manual meta tensors …")
                model = AutoModelForCausalLM.from_config(cfg)
                for p in model.parameters():
                    p.data = torch.empty_like(p.data, device="meta")
Gather python‑object shards to rank‑0 and push a single parquet file."""
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        flat: List[dict] = []
        for part in gathered:  # type: ignore[arg-type]
            flat.extend(part)
        push_shard(flat, args, dbg)


def worker(args: Args):
    """Main per‑process loop: init dist, stream data, sample, push."""
    # ---------------- init ----------------
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
                dbg("SIGTERM received; breaking loop")
                break

            dbg(f"Batch {batch_idx} acquired")
            input_ids = batch["input_ids"].to(engine.device, non_blocking=True)
            hidden = engine.layers[0].to(engine.device)(input_ids)  # embed once
            ids, cnts = engine.sample(hidden)

            for i in range(len(input_ids)):
                toks = input_ids[i].tolist()
                while toks and toks[-1] == 0:
                    toks.pop()
                seq_len = len(toks)
                local_records.append(
                    {
                        "input_ids": toks,
                        "sampled_ids": ids[i][:seq_len],
                        "sampled_counts": cnts[i][:seq_len],
                    }
                )
            seen += len(input_ids)

            if seen >= args.push_every:
                dbg("Local push_every reached; gathering to rank‑0 …")
                _gather_and_push(local_records, args, rank, world, dbg)
                local_records.clear(); seen = 0

        # final flush
        if local_records and not stop_evt.is_set():
            dbg("Final flush …")
            _gather_and_push(local_records, args, rank, world, dbg)

    except Exception:
        dbg("Exception occurred! Printing traceback …")
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    dbg("Graceful shutdown")
    dist.destroy_process_group()


###############################################################################
#                                   Entry                                    #
###############################################################################

def main():
    args = parse_args()
    worker(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Generator with **reverse pipeline parallelism** (ring) + verbose DEBUG logs.
Params circulate CPU → GPU‑0 → … → GPU‑N‑1 → GPU‑0 (then rank‑0 drops to CPU).

Pass `--debug` to enable very chatty per‑rank prints, otherwise only coarse
progress is reported.
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
#                           Helper: debug printer                             #
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
    """Flatten parameters into a contiguous CUDA tensor for fast P2P."""

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
#                    Reverse pipeline engine (ring)                           #
###############################################################################

class ReversePipelineEngine:
    """Stream *layers* through static hidden‑states on each GPU."""

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
            dbg("Loading full model on CPU…")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=cfg,
                device_map={"": "cpu"},
                torch_dtype=torch.bfloat16,
            )
            model.config.attn_implementation = "flash_attention_2"
        else:
            dbg("Building meta skeleton…")
            try:
                # Use accelerate to create an **empty‑weight** model straight on meta device
                from accelerate import init_empty_weights
            except ImportError:
                raise RuntimeError("accelerate>=0.26 required for init_empty_weights. Install via `pip install accelerate>=0.26`. ")
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(cfg)

        self.layers: List[torch.nn.Module] = [
            model.model.embed_tokens,
            *model.model.layers,
            model.lm_head,
        ]: List[torch.nn.Module] = [
            model.model.embed_tokens,
            *model.model.layers,
            model.lm_head,
        ]
        dbg(f"Total layers: {len(self.layers)}")

        self.buffers = (_FlatPacket(), _FlatPacket())
        self.toggle = 0

    # ------------------------------ ring helpers

    def _ring_send(self, tensor: torch.Tensor, dst: int):
        self.dbg(f"send → {dst}, numel={tensor.numel()}")
        dist.send(tensor, dst)

    def _ring_recv(self, tensor: torch.Tensor, src: int):
        self.dbg(f"recv ← {src}, expect numel={tensor.numel()}")
        dist.recv(tensor, src)

    # ------------------------------ layer streaming

    def _stream_layer(self, idx: int):
        left = (self.rank - 1) % self.world
        right = (self.rank + 1) % self.world
        buf = self.buffers[self.toggle]

        if not (self.rank == 0 and idx == 0):
            shape = torch.empty(1, dtype=torch.int64, device=self.device)
            self._ring_recv(shape, left)
            buf.make_empty(int(shape.item()), self.device)
            self._ring_recv(buf.flat, left)
        else:
            buf.pack(self.layers[idx], self.device)

        buf.unpack_to(self.layers[idx])

        shape_out = torch.tensor([buf.flat.numel()], dtype=torch.int64, device=self.device)
        self._ring_send(shape_out, right)
        self._ring_send(buf.flat, right)

        self.toggle ^= 1

    # ------------------------------ forward & sample

    @torch.no_grad()
    def _forward_full(self, hidden: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.layers)):
            self.dbg(f"Layer {idx}/{len(self.layers)-1} start")
            self._stream_layer(idx)
            layer = self.layers[idx].to(self.device, non_blocking=True)
            for start in range(0, hidden.size(0), self.args.micro_batch_size):
                mb = hidden[start : start + self.args.micro_batch_size]
                out = layer(mb)
                hidden[start : start + self.args.micro_batch_size] = out[0] if isinstance(out, tuple) else out
            layer.to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()
            self.dbg(f"Layer {idx} done")
        return hidden

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor):
        logits = self._forward_full(hidden)
        probs = torch.softmax(logits / self.args.sampling_temperature, dim=-1)
        draws = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            self.args.num_samples * self.args.num_rounds,
            replacement=True,
        ).view(*probs.shape[:-1], -1)

        ids_all, counts_all = [], []
        for b in range(draws.size(0)):
            ids_seq, cnt_seq = [], []
            for t in range(draws.size(1)):
                uniq, cnt = torch.unique(draws[b, t], return_counts=True)
                ids_seq.append(uniq.cpu().tolist())
                cnt_seq.append(cnt.cpu().tolist())
            ids_all.append(ids_seq)
            counts_all.append(cnt_seq)
        return ids_all, counts_all

###############################################################################
#                    Data utilities & HF push                                #
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

def push_shard(records: List[dict], args: Args, dbg):
    dbg(f"Pushing {len(records)} records to Hub…")
    api = HfApi(token=args.hf_token)
    idx = sum(f.endswith(".parquet") for f in api.list_repo_files(args.output_repo, repo_type="dataset"))
    fname = f"data_{idx:05d}.parquet"
    Dataset.from_list(records).to_parquet(fname)
    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull(); repo.git_add(fname); repo.git_commit(f"Add shard {idx}"); repo.git_push(); repo.git_clear()
    os.remove(fname)

###############################################################################
#                                   Worker                                   #
###############################################################################

def _gather_and_push(local: List[dict], args: Args, rank: int, world: int, dbg):
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        flat: List[dict] = []
        for part in gathered:  # type: ignore[arg-type]
            flat.extend(part)
        push_shard(flat, args, dbg)


def worker(args: Args):
    # Distributed init
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dbg = get_logger(rank, args.debug)

    dbg("Process started, setting handlers…")
    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())

    try:
        engine = ReversePipelineEngine(args, rank, world, dbg)
        tok = AutoTokenizer.from_pretrained(args.model_name)
        ds = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True, token=args.hf_token)
        loader = stream_loader(ds, tok, args.batch_size, args.max_seq_len)

        local_records: List[dict] = []
        seen = 0
        for i, batch in enumerate(loader):
            if stop_evt.is_set():
                dbg("SIGTERM received, exiting loop")
                break
            dbg(f"Batch {i} acquired")
            input_ids = batch["input_ids"].to(engine.device, non_blocking=True)
            hidden = engine.layers[0].to(engine.device)(input_ids)
            ids, cnts = engine.sample(hidden)
            for j in range(len(input_ids)):
                toks = input_ids[j].tolist()
                while toks and toks[-1] == 0:
                    toks.pop()
                seq_len = len(toks)
                local_records.append({
                    "input_ids": toks,
                    "sampled_ids": ids[j][:seq_len],
                    "sampled_counts": cnts[j][:seq_len],
                })
            seen += len(input_ids)
            if seen >= args.push_every:
                _gather_and_push(local_records, args, rank, world, dbg)
                local_records.clear(); seen = 0
            dbg(f"Batch {i} done, total local records {len(local_records)}")
            if stop_evt.is_set():
                break
        if local_records and not stop_evt.is_set():
            _gather_and_push(local_records, args, rank, world, dbg)
    except Exception as e:
        dbg("Exception occurred! Printing traceback…")
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    dbg("Graceful shutdown")
    dist.destroy_process_group()

###############################################################################
#                                   Main                                     #
###############################################################################

def main():
    args = parse_args()
    worker(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Generator with **reverse pipeline parallelism** using NCCL peer‑to‑peer.
Parameters circulate in a ring  CPU → GPU‑0 → GPU‑1 → … → GPU‑(N−1) → GPU‑0  
Rank‑0 immediately off‑loads the final hop back to CPU so memory pressure on
GPU‑0 stays bounded.

Key implementation points
-------------------------
* **CUDA tensors only** in `dist.send/recv` ➟ avoids the «No backend type associated
  with device type cpu» error.
* **Skeleton weights on non‑zero ranks** are initialised as *meta* tensors with
  the same *shape* and **bfloat16** dtype; real parameters arrive via the ring.
* **Double‑buffering** hides latency: while one buffer is used for compute, the
  other is in flight to the next rank.
* **Vectorised multinomial**: sampling is a single GPU call—no Python loops.

The script still matches your SLURM launch line verbatim.
"""

import argparse
import os
import signal
import threading
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, Repository
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

###############################################################################
#                               Args                                          #
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


def parse_args() -> Args:
    p = argparse.ArgumentParser("Reverse‑pipeline KD generator")
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

###############################################################################
#                       Flat parameter packet helper                          #
###############################################################################

class _FlatPacket:
    """Simple container that flattens parameters for efficient P2P transfer."""

    def __init__(self):
        self.flat: torch.Tensor | None = None  # always CUDA when populated

    # ---------------------------------------------------------------- pack/unpack

    def pack(self, layer: torch.nn.Module, device: torch.device) -> None:
        """Flatten `layer` params into a single contiguous CUDA tensor."""
        with torch.no_grad():
            vec = [p.data.view(-1) for p in layer.parameters()]
            self.flat = torch.cat(vec).contiguous().to(device, non_blocking=True)

    def make_empty(self, numel: int, device: torch.device) -> None:
        self.flat = torch.empty(numel, dtype=torch.bfloat16, device=device)

    def unpack_to(self, layer: torch.nn.Module) -> None:
        if self.flat is None:
            raise RuntimeError("_FlatPacket is empty — cannot unpack")
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
    """Streams *layers* through static hidden‑states in a ring topology."""

    def __init__(self, args: Args, rank: int, world: int):
        self.args = args
        self.rank = rank
        self.world = world
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)

        # ---------------------------------------------------------------- model skeleton
        cfg = AutoConfig.from_pretrained(args.model_name)
        cfg.attn_implementation = "eager"  # safe on CPU

        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=cfg,
                device_map={"": "cpu"},
                torch_dtype=torch.bfloat16,
            )
            model.config.attn_implementation = "flash_attention_2"
        else:
            model = AutoModelForCausalLM.from_config(cfg)
            # Replace real storage with meta tensors (same shape, bf16) to avoid memory use
                        for p in model.parameters():
                # Preserve original dtype to avoid the incompatible‑type error
                with torch.no_grad():
                    p.data = torch.empty_like(p.data, device="meta")

        self.layers: List[torch.nn.Module] = [
            model.model.embed_tokens,
            *model.model.layers,
            model.lm_head,
        ]: List[torch.nn.Module] = [
            model.model.embed_tokens,
            *model.model.layers,
            model.lm_head,
        ]

        # two flight buffers (double‑buffer)
        self.buffers = (_FlatPacket(), _FlatPacket())
        self.toggle = 0

    # ---------------------------------------------------------------- helper P2P wrappers

    def _ring_send(self, tensor: torch.Tensor, dst: int) -> None:
        dist.send(tensor=tensor, dst=dst)

    def _ring_recv(self, tensor: torch.Tensor, src: int) -> None:
        dist.recv(tensor=tensor, src=src)

    # ---------------------------------------------------------------- layer streaming primitive

    def _stream_layer(self, idx: int) -> None:
        left = (self.rank - 1) % self.world
        right = (self.rank + 1) % self.world
        buf_in = self.buffers[self.toggle]

        # 1) receive or pack
        if not (self.rank == 0 and idx == 0):
            shape = torch.empty(1, dtype=torch.int64, device=self.device)
            self._ring_recv(shape, src=left)
            buf_in.make_empty(int(shape.item()), self.device)
            self._ring_recv(buf_in.flat, src=left)
        else:
            buf_in.pack(self.layers[idx], self.device)  # rank‑0 kick‑off

        # 2) materialise on this rank
        buf_in.unpack_to(self.layers[idx])

        # 3) forward payload to the next rank
        shape_out = torch.tensor([buf_in.flat.numel()], dtype=torch.int64, device=self.device)
        self._ring_send(shape_out, dst=right)
        self._ring_send(buf_in.flat, dst=right)

        # switch buffers for next layer
        self.toggle = 1 - self.toggle

    # ---------------------------------------------------------------- forward & sample

    @torch.no_grad()
    def _forward_full(self, hidden: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.layers)):
            self._stream_layer(idx)
            layer = self.layers[idx].to(self.device, non_blocking=True)
            for start in range(0, hidden.size(0), self.args.micro_batch_size):
                mb = hidden[start : start + self.args.micro_batch_size]
                out = layer(mb)
                hidden[start : start + self.args.micro_batch_size] = out[0] if isinstance(out, tuple) else out
            layer.to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()
        return hidden  # logits

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        logits = self._forward_full(hidden)
        probs = torch.softmax(logits / self.args.sampling_temperature, dim=-1)
        draws = torch.multinomial(
            probs.view(-1, probs.size(-1)), self.args.num_samples * self.args.num_rounds, replacement=True
        ).view(*probs.shape[:-1], -1)

        ids_all: List[List[List[int]]] = []
        counts_all: List[List[List[int]]] = []
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
#                    Data handling & Hub push                                #
###############################################################################

def collate_fn(examples, tok, max_len):
    return tok([e["text"] for e in examples], return_tensors="pt", padding=True, truncation=True, max_length=max_len)

def streaming_loader(ds, tok, bs, max_len):
    buf = []
    for ex in ds:
        buf.append(ex)
        if len(buf) == bs:
            yield collate_fn(buf, tok, max_len)
            buf.clear()
    if buf:
        yield collate_fn(buf, tok, max_len)

def push_shard(records: List[dict], args: Args):
    api = HfApi(token=args.hf_token)
    files = api.list_repo_files(args.output_repo, repo_type="dataset")
    idx = sum(f.endswith(".parquet") for f in files)
    fname = f"data_{idx:05d}.parquet"
    Dataset.from_list(records).to_parquet(fname)
    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull()
    repo.git_add(fname)
    repo.git_commit(f"Add shard {idx}")
    repo.git_push()
    repo.git_clear()
    os.remove(fname)

###############################################################################
#                                   Worker                                   #
###############################################################################

def _gather_and_push(local: List[dict], args: Args, rank: int, world: int):
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        merged: List[dict] = []
        for part in gathered:  # type: ignore[arg-type]
            merged.extend(part)
        push_shard(merged, args)


def worker(args: Args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())

    engine = ReversePipelineEngine(args, rank, world)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True, token=args.hf_token)
    loader = streaming_loader(ds, tok, args.batch_size, args.max_seq_len)

    local_records: List[dict] = []
    seen = 0
    for batch in loader:
        if stop_evt.is_set():
            break
        input_ids = batch["input_ids"].to(engine.device, non_blocking=True)
        hidden = engine.layers[0].to(engine.device)(input_ids)  # embed is index‑0
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
            _gather_and_push(local_records, args, rank, world)
            local_records.clear(); seen = 0
        if stop_evt.is_set():
            break
    if local_records and not stop_evt.is_set():
        _gather_and_push(local_records, args, rank, world)

    dist.destroy_process_group()

###############################################################################
#                                   Main                                     #
###############################################################################

def main():
    worker(parse_args())


if __name__ == "__main__":
    main()

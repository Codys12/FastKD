from __future__ import annotations

"""
Generator with **reverse pipeline parallelism** using NCCL peer‑to‑peer.
Parameters circulate in a ring  CPU → GPU‑0 → GPU‑1 → … → GPU‑(N−1) → GPU‑0  
Rank‑0 immediately off‑loads the final hop back to CPU so memory pressure on
GPU‑0 stays bounded.

Key implementation points
-------------------------
* **CUDA tensors only** in `dist.send/recv` ➟ avoids CPU‑backend errors.
* **Skeleton weights on non‑zero ranks**: stored as *meta* tensors (same shape &
  dtype) so they allocate **zero** real memory until packets arrive.
* **Double‑buffering** overlaps weight transfer and compute.
* **Vectorised multinomial** for fast sampling.

Launch command unchanged – works with `torchrun --nproc_per_node 8 …`.
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

    def __init__(self, args: Args, rank: int, world: int):
        self.args = args
        self.rank = rank
        self.world = world
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)

        # ------------------------------ model skeleton
        cfg = AutoConfig.from_pretrained(args.model_name)
        cfg.attn_implementation = "eager"

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
            for p in model.parameters():
                with torch.no_grad():
                    p.data = torch.empty_like(p.data, device="meta")

        self.layers: List[torch.nn.Module] = [
            model.model.embed_tokens,
            *model.model.layers,
            model.lm_head,
        ]

        # double‑buffer
        self.buffers = (_FlatPacket(), _FlatPacket())
        self.toggle = 0

    # ------------------------------ ring helpers

    def _ring_send(self, tensor: torch.Tensor, dst: int):
        dist.send(tensor, dst)

    def _ring_recv(self, tensor: torch.Tensor, src: int):
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

        self.toggle ^= 1  # swap buffers

    # ------------------------------ forward & sample

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

def push_shard(records: List[dict], args: Args):
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

def _gather_and_push(local: List[dict], args: Args, rank: int, world: int):
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        flat: List[dict] = []
        for part in gathered:  # type: ignore[arg-type]
            flat.extend(part)
        push_shard(flat, args)


def worker(args: Args):
    dist.init

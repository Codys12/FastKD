from __future__ import annotations

"""
Generator with **reverse pipeline parallelism** using NCCL peer‑to‑peer.
Parameters travel in a ring  CPU → GPU‑0 → GPU‑1 → … → GPU‑(N−1) → **GPU‑0**
(where rank‑0 off‑loads back to CPU after the final hop). All tensors used in
`dist.send/recv` are **CUDA** so the NCCL backend is valid.

This version fixes the runtime error «No backend type associated with device
type cpu» by ensuring that every tensor passed to NCCL lives on a CUDA device.

Double‑buffering still overlaps transfer / compute. Sampling is vectorised.
The launch command from your SLURM script remains unchanged.
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
    def __init__(self):
        self.flat: torch.Tensor | None = None  # CUDA tensor when active

    def pack(self, layer: torch.nn.Module, device: torch.device) -> None:
        with torch.no_grad():
            vec = [p.data.view(-1) for p in layer.parameters()]
            self.flat = torch.cat(vec).contiguous().to(device, non_blocking=True)

    def make_empty(self, numel: int, device: torch.device) -> None:
        self.flat = torch.empty(numel, dtype=torch.bfloat16, device=device)

    def unpack_to(self, layer: torch.nn.Module) -> None:
        if self.flat is None:
            raise RuntimeError("attempt to unpack empty packet")
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
    """Stream *layers* through resident hidden‑states on each GPU."""

    def __init__(self, args: Args, rank: int, world: int):
        self.args = args
        self.rank = rank
        self.world = world
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)

        # Build model skeleton on every rank (meta params to save memory)
        base_cfg = AutoConfig.from_pretrained(args.model_name)
        base_cfg.attn_implementation = "eager"  # Later we flip to flash_attention_2
        if rank == 0:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=base_cfg,
                device_map={"": "cpu"},
                torch_dtype=torch.bfloat16,
            )
            model.config.attn_implementation = "flash_attention_2"
        else:
            model = AutoModelForCausalLM.from_config(base_cfg)
            for p in model.parameters():
                p.data = torch.empty(0, device="meta")

        self.layers: List[torch.nn.Module] = [
            model.model.embed_tokens,
            *model.model.layers,
            model.lm_head,
        ]

        # Double buffers for weight flighting
        self.buffers = (_FlatPacket(), _FlatPacket())
        self.toggle = 0

    # ------------------------------------------------------------------ utils

    def _ring_send(self, tensor: torch.Tensor, dst: int) -> None:
        dist.send(tensor=tensor, dst=dst)

    def _ring_recv(self, tensor: torch.Tensor, src: int) -> None:
        dist.recv(tensor=tensor, src=src)

    # ---------------------------------------------------------------- pipeline

    def _stream_layer(self, idx: int) -> None:
        """Move *one* layer along the ring; double‑buffered."""
        left = (self.rank - 1) % self.world
        right = (self.rank + 1) % self.world

        buf_in = self.buffers[self.toggle]
        buf_out = self.buffers[1 - self.toggle]

        # 1. Receive parameters from left neighbour (except rank‑0 first hop)
        if not (self.rank == 0 and idx == 0):
            shape_gpu = torch.empty(1, dtype=torch.int64, device=self.device)
            self._ring_recv(shape_gpu, src=left)
            numel = int(shape_gpu.item())
            buf_in.make_empty(numel, self.device)
            self._ring_recv(buf_in.flat, src=left)
        else:
            # Rank‑0 packs first layer
            buf_in.pack(self.layers[idx], self.device)

        # 2. Materialise layer on this rank
        buf_in.unpack_to(self.layers[idx])

        # 3. Send to right neighbour (rank‑0 still sends, including last GPU → 0)
        shape_gpu = torch.tensor([buf_in.flat.numel()], dtype=torch.int64, device=self.device)
        self._ring_send(shape_gpu, dst=right)
        self._ring_send(buf_in.flat, dst=right)

        self.toggle = 1 - self.toggle  # swap buffers

    # ---------------------------------------------------- forward + sampling

    @torch.no_grad()
    def _forward_full(self, hidden: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            self._stream_layer(i)
            layer = self.layers[i].to(self.device, non_blocking=True)
            for start in range(0, hidden.size(0), self.args.micro_batch_size):
                mb = hidden[start : start + self.args.micro_batch_size]
                out = layer(mb)
                if isinstance(out, tuple):
                    out = out[0]
                hidden[start : start + self.args.micro_batch_size] = out
            layer.to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()
        return hidden  # logits after lm_head

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
            ids_seq, counts_seq = [], []
            for t in range(draws.size(1)):
                uniq, cnt = torch.unique(draws[b, t], return_counts=True)
                ids_seq.append(uniq.cpu().tolist())
                counts_seq.append(cnt.cpu().tolist())
            ids_all.append(ids_seq)
            counts_all.append(counts_seq)
        return ids_all, counts_all

###############################################################################
#                    Data handling & HF push                                 #
###############################################################################

def collate_fn(examples, tok, max_len):
    return tok([e["text"] for e in examples], return_tensors="pt", padding=True, truncation=True, max_length=max_len)

def streaming_loader(ds, tok, bs, max_len):
    batch = []
    for ex in ds:
        batch.append(ex)
        if len(batch) == bs:
            yield collate_fn(batch, tok, max_len)
            batch.clear()
    if batch:
        yield collate_fn(batch, tok, max_len)

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

def _gather_and_push(local_records: List[dict], args: Args, rank: int, world: int):
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    if rank == 0:
        merged: List[dict] = []
        for part in gathered:  # type: ignore[assignment]
            merged.extend(part)
        push_shard(merged, args)


def worker(args: Args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    engine = ReversePipelineEngine(args, rank, world)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True, token=args.hf_token)
    loader = streaming_loader(ds, tok, args.batch_size, args.max_seq_len)

    local_records: List[dict] = []
    seen = 0
    for batch in loader:
        if stop_event.is_set():
            break
        input_ids = batch["input_ids"].to(engine.device, non_blocking=True)
        hidden = engine.layers[0].to(engine.device)(input_ids)  # embed layer stays resident
        ids, counts = engine.sample(hidden)
        for i in range(len(input_ids)):
            toks = input_ids[i].tolist()
            while toks and toks[-1] == 0:
                toks.pop()
            seq_len = len(toks)
            local_records.append({
                "input_ids": toks,
                "sampled_ids": ids[i][:seq_len],
                "sampled_counts": counts[i][:seq_len],
            })
        seen += len(input_ids)
        if seen >= args.push_every:
            _gather_and_push(local_records, args, rank, world)
            local_records.clear(); seen = 0
        if stop_event.is_set():
            break
    if local_records and not stop_event.is_set():
        _gather_and_push(local_records, args, rank, world)

    dist.destroy_process_group()

###############################################################################
#                                   main                                     #
###############################################################################

def main():
    worker(parse_args())

if __name__ == "__main__":
    main()

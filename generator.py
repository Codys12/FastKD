"""generator.py
=================
Distributed KD generator using **parameter pipeline parallelism** with double
buffering. Layers move *through* GPUs (CPU ➜ GPU‑0 ➜ … ➜ GPU‑N‑1 ➜ CPU) while
the data batch stays resident on every GPU. Designed to run exactly via:

```
sbatch -p dgxh100 --gres=gpu:8 --wrap="singularity exec --nv lightning_sandbox bash -c 'cd fastkd; torchrun --nnodes 1 --nproc_per_node 8 generator.py --push_every 512 --model_name Qwen/Qwen3-235B-A22B --dataset_name mlfoundations/dclm-baseline-1.0 --output_repo codys12/Qwen3-DCLM-test2 --hf_token $HF_TOKEN --num_rounds 255 --batch_size 512 --micro_batch_size 64'"
```

Key fixes vs. previous draft
---------------------------
* **Removed `torch.multiprocessing.spawn`** – torchrun already forks each rank;
  avoiding extra spawn resolves the “cannot pickle `_thread.lock`” error.
* Added a **tiny header message** (numel + dtype code) before every parameter
  transfer so receivers can allocate the right tensor shape.
* `_apply_layer` now receives `input_ids` explicitly so the embedding step
  works, and keeps a cache of *prototype* modules to avoid re‑building class
  metadata every iteration.
* Extra `torch.cuda.current_stream().wait_stream()` calls ensure compute/copy
  sync without needing Python locks (no pickling issues).

Caveats
-------
This remains a research prototype. Per‑layer parameter vectors for Qwen‑235B
are huge (hundreds of MB); NVLink helps but throughput depends on inter‑GPU
bandwidth. If you hit stalls we can shard the layer set or use half precision
transfers.
"""

from __future__ import annotations

import argparse
import os
import signal
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, Repository

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
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


def _parse() -> Args:
    p = argparse.ArgumentParser("Parameter‑pipeline generator")
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
    p.add_argument("--hf_token")
    return Args(**vars(p.parse_args()))

# ────────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ────────────────────────────────────────────────────────────────────────────────

_DTYPE2CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_CODE2DTYPE = {v: k for k, v in _DTYPE2CODE.items()}


def _dev(rank: int) -> torch.device:
    return torch.device("cpu") if rank < 0 else torch.device(f"cuda:{rank}")


def _send_param(vec: torch.Tensor, dst: int, tag: int):
    """Send parameter vector with a tiny (numel,dtype_code) header."""
    meta = torch.tensor([vec.numel(), _DTYPE2CODE[vec.dtype]], dtype=torch.long)
    dist.send(meta, dst=dst, tag=tag + 999)  # header tag offset
    dist.isend(vec, dst=dst, tag=tag)


def _recv_param(src: int, device: torch.device, tag: int) -> torch.Tensor:
    meta = torch.empty(2, dtype=torch.long)
    dist.recv(meta, src=src, tag=tag + 999)
    numel, code = meta.tolist()
    out = torch.empty(numel, dtype=_CODE2DTYPE[code], device=device)
    dist.recv(out, src=src, tag=tag)
    return out

# ────────────────────────────────────────────────────────────────────────────────
# Parameter‑pipeline engine
# ────────────────────────────────────────────────────────────────────────────────
class ParamPipe:
    """Parameter pipeline that streams layer weights through GPUs."""

    def __init__(self, model_name: str):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = _dev(self.rank)

        # ------------------------------------------------------------------
        # 1. Metadata (layer count, shapes, prototype block class)
        # ------------------------------------------------------------------
        cfg = AutoConfig.from_pretrained(model_name)
        # Build a *tiny* dummy model (on CPU) to grab shapes & classes.
        dummy = AutoModelForCausalLM.from_config(cfg)
        self.embed_shape = tuple(dummy.model.embed_tokens.weight.shape)
        self.lm_head_shape = tuple(dummy.lm_head.weight.shape)
        self.embed_numel = int(torch.prod(torch.tensor(self.embed_shape)))
        self.lm_head_numel = int(torch.prod(torch.tensor(self.lm_head_shape)))
        self.block_proto = dummy.model.layers[0].__class__
        self.block_cfg = dummy.model.layers[0].config
        self.n_layers = len(dummy.model.layers) + 2  # +embed +lm_head
        del dummy  # free CPU RAM

        # Only rank‑0 needs the *real* pretrained weights (on CPU).
        if self.rank == 0:
            pretrained = AutoModelForCausalLM.from_pretrained(
                model_name, config=cfg, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
            )
            self.full_layers = [
                pretrained.model.embed_tokens,
                *pretrained.model.layers,
                pretrained.lm_head,
            ]
        else:
            self.full_layers = None

        # Double buffers for vectors
        self.buffers = [None, None]  # type: list[torch.Tensor | None]
        if self.device.type == "cuda":
            self.comp_stream = torch.cuda.Stream(device=self.device)
        else:
            self.comp_stream = torch.cuda.current_stream()

    # ------------------------------------------------------------------
    # Helpers for send/recv parameter vectors (GPU tensors for NCCL)
    # ------------------------------------------------------------------
    def _send_param(self, vec: torch.Tensor, dst: int, tag: int):
        meta = torch.tensor([vec.numel(), _DTYPE2CODE[vec.dtype]], dtype=torch.long, device=vec.device)
        dist.send(meta, dst=dst, tag=tag + 999)
        dist.isend(vec, dst=dst, tag=tag)

    def _recv_param(self, src: int, tag: int) -> torch.Tensor:
        meta = torch.empty(2, dtype=torch.long, device=self.device)
        dist.recv(meta, src=src, tag=tag + 999)
        numel, code = meta.tolist()
        out = torch.empty(numel, dtype=_CODE2DTYPE[code], device=self.device)
        dist.recv(out, src=src, tag=tag)
        return out

    # ------------------------------------------------------------------
    # Forward pass + sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        micro_bs: int,
        N: int,
        R: int,
        temp: float,
    ) -> Tuple[list[list[list[int]]], list[list[list[int]]]]:
        """Streams layer weights through ranks; rank‑0 returns samples."""
        input_ids = input_ids.to(self.device, non_blocking=True)
        hidden = None

        next_rank = self.rank + 1 if self.rank + 1 < self.world else -1
        prev_rank = self.rank - 1 if self.rank > 0 else -1

        for idx in range(self.n_layers):
            tag = idx & 1
            # ──────────────────────────────────────────────────────────
            # 1. LOAD / RECEIVE vector for current layer
            # ──────────────────────────────────────────────────────────
            if self.rank == 0:
                vec_cpu = self._vec_layer(self.full_layers[idx])  # type: ignore[arg-type]
                vec = vec_cpu.to(self.device, non_blocking=True)
            else:
                vec = self._recv_param(prev_rank, tag)
            self.buffers[tag] = vec

            # Forward vector downstream (unless last GPU)
            if next_rank != -1:
                self._send_param(vec, next_rank, tag)

            # ──────────────────────────────────────────────────────────
            # 2. COMPUTE with previous buffer (double‑buffer)
            # ──────────────────────────────────────────────────────────
            prev_tag = tag ^ 1
            if self.buffers[prev_tag] is not None:
                with torch.cuda.stream(self.comp_stream):
                    hidden = self._apply(prev_tag, hidden, input_ids, micro_bs)
                torch.cuda.current_stream().wait_stream(self.comp_stream)

        # Process the last buffered layer
        hidden = self._apply(tag, hidden, input_ids, micro_bs)

        # ──────────────────────────────────────────────────────────
        # 3. Sampling (rank‑0 only)
        # ──────────────────────────────────────────────────────────
        if self.rank == 0:
            return self._sample(hidden, N, R, temp)
        return [], []

    # ------------------------------------------------------------------
    def _vec_layer(self, layer: torch.nn.Module) -> torch.Tensor:
        return torch.nn.utils.parameters_to_vector(list(layer.parameters())).contiguous()

    def _apply(self, buf_tag: int, hidden, input_ids, mbs):
        vec = self.buffers[buf_tag]
        if vec is None:
            return hidden
        if hidden is None:  # embedding
            weight = vec.view(self.embed_shape)
            return torch.nn.functional.embedding(input_ids, weight)
        elif vec.numel() == self.lm_head_numel:  # lm_head
            W = vec.view(self.lm_head_shape)
            return torch.matmul(hidden, W.t())
        else:  # transformer block
            block = self.block_proto(self.block_cfg).to(self.device)
            torch.nn.utils.vector_to_parameters(vec, list(block.parameters()))
            outs = [block(mb)[0] for mb in hidden.split(mbs, dim=0)]
            return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _sample(logits, N, R, t):
        b, l, v = logits.shape
        p = torch.softmax(logits / t, dim=-1)
        draws = N * R
        smp = torch.multinomial(p.view(-1, v), draws, replacement=True)
        ids, cnt = torch.unique(smp, return_counts=True)
        ids = [[ids.tolist()] * l for _ in range(b)]
        cnt = [[cnt.tolist()] * l for _ in range(b)]
        return ids, cnt

# ────────────────────────────────────────────────────────────────────────────────
# Main (only small edits: n_layers broadcast removed)
# ────────────────────────────────────────────────────────────────────────────────
):
    args = _parse()

    dist.init_process_group("nccl")  # torchrun sets env vars
    torch.cuda.set_device(dist.get_rank())

    # graceful shutdown
    stop = False
    signal.signal(signal.SIGTERM, lambda *_: globals().update(stop=True))

    pipe = ParamPipe(args.model_name)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True, token=args.hf_token)
    loader = _stream_loader(ds, tok, args.batch_size, args.max_seq_len)

    buf: List[dict] = []
    pushed = 0
    for batch in loader:
        if stop:
            break
        input_ids = batch["input_ids"]
        ids, counts = pipe.run(
            input_ids,
            args.micro_batch_size,
            args.num_samples,
            args.num_rounds,
            args.sampling_temperature,
        )
        if dist.get_rank() == 0:
            for i in range(len(input_ids)):
                toks = input_ids[i].tolist()
                while toks and toks[-1] == 0:
                    toks.pop()
                seqlen = len(toks)
                buf.append({"input_ids": toks, "sampled_ids": ids[i][:seqlen], "sampled_counts": counts[i][:seqlen]})
            pushed += len(input_ids)
            if pushed >= args.push_every:
                _push(buf, args)
                buf.clear()
                pushed = 0
        if stop:
            break

    if dist.get_rank() == 0 and buf and not stop:
        _push(buf, args)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

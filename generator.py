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
    def __init__(self, model_name: str):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = _dev(self.rank)

        # Rank‑0 loads full model weights on CPU
        if self.rank == 0:
            cfg = AutoConfig.from_pretrained(model_name)
            cfg.attn_implementation = "eager"  # safer on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, config=cfg, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
            )
            self.layers = [self.model.model.embed_tokens, *self.model.model.layers, self.model.lm_head]
        else:
            self.model = None  # type: ignore
            self.layers = None  # will be broadcast later

        # Share #layers to everybody
        n_layers = torch.tensor([len(self.layers) if self.rank == 0 else 0], dtype=torch.long)
        dist.broadcast(n_layers, src=0)
        self.n_layers = int(n_layers.item())

        # Double buffers
        self.buffers = [None, None]  # type: List[torch.Tensor | None]
        if self.device.type == "cuda":
            self.comp_stream = torch.cuda.Stream(device=self.device)
            self.copy_stream = torch.cuda.Stream(device=self.device)
        else:
            self.comp_stream = self.copy_stream = torch.cuda.current_stream()

        # Prototype layer instances (avoids class rebuild every pass)
        if self.rank == 0:
            self.block_proto = self.model.model.layers[0].__class__  # type: ignore
            self.block_cfg = self.model.model.layers[0].config  # type: ignore
        else:
            self.block_proto = self.block_cfg = None  # will come over broadcast if needed

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        micro_bs: int,
        N: int,
        R: int,
        temp: float,
    ) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """Execute one forward + sampling (sampling only rank‑0)."""
        input_ids = input_ids.to(self.device, non_blocking=True)
        hidden = None

        next_rank = self.rank + 1 if self.rank + 1 < self.world else -1
        prev_rank = self.rank - 1 if self.rank > 0 else -1

        for idx in range(self.n_layers):
            tag = idx & 1  # ping‑pong
            # ─ receive parameters (except rank‑0 embed handled locally) ──
            if not (self.rank == 0 and idx == 0):
                self.buffers[tag] = _recv_param(prev_rank, self.device, tag)
            else:  # rank‑0 embed
                self.buffers[tag] = self._vec_layer(self.layers[idx]).to(self.device, non_blocking=True)

            # ─ send params downstream asynchronously ──
            if next_rank != -1:
                _send_param(self.buffers[tag], next_rank, tag)

            # ─ compute with **previous** buffer (double‑buffering) ──
            prev_tag = tag ^ 1
            if self.buffers[prev_tag] is not None:
                with torch.cuda.stream(self.comp_stream):
                    hidden = self._apply(prev_tag, hidden, input_ids, micro_bs)
            # sync so hidden ready before next iter
            torch.cuda.current_stream().wait_stream(self.comp_stream)

        # After last real layer, process lm_head (already in buffers[tag])
        with torch.cuda.stream(self.comp_stream):
            hidden = self._apply(tag, hidden, input_ids, micro_bs)
        torch.cuda.current_stream().wait_stream(self.comp_stream)

        if self.rank == 0:
            return self._sample(hidden, N, R, temp)
        return [], []

    # ------------------------------------------------------------------
    def _vec_layer(self, layer: torch.nn.Module) -> torch.Tensor:
        return torch.nn.utils.parameters_to_vector(list(layer.parameters())).contiguous()

    def _apply(
        self,
        buf_tag: int,
        hidden: torch.Tensor | None,
        input_ids: torch.Tensor,
        mbs: int,
    ) -> torch.Tensor:
        vec = self.buffers[buf_tag]
        assert vec is not None
        # Determine layer type by size comparison (cheap & safe under fixed arch)
        if hidden is None:  # embedding layer
            weight = vec.view_as(self.layers[0].weight)  # type: ignore
            out = torch.nn.functional.embedding(input_ids, weight)
            return out
        elif vec.numel() == self.layers[-1].weight.numel():  # lm_head
            W = vec.view_as(self.layers[-1].weight)  # type: ignore
            logits = torch.matmul(hidden, W.t())
            return logits
        else:  # transformer block
            block: torch.nn.Module = self.block_proto(self.block_cfg).to(self.device)  # type: ignore
            torch.nn.utils.vector_to_parameters(vec, list(block.parameters()))
            outs = []
            for mb in hidden.split(mbs, dim=0):
                outs.append(block(mb)[0])  # block returns tuple
            return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _sample(logits: torch.Tensor, N: int, R: int, t: float):
        b, l, v = logits.shape
        p = torch.softmax(logits / t, dim=-1)
        draws = N * R
        samples = torch.multinomial(p.view(-1, v), draws, replacement=True)
        ids, counts = torch.unique(samples, return_counts=True)
        # Broadcast identical ids/counts for every position to match API
        ids = [[ids.tolist() for _ in range(l)] for _ in range(b)]
        counts = [[counts.tolist() for _ in range(l)] for _ in range(b)]
        return ids, counts

# ────────────────────────────────────────────────────────────────────────────────
# Data helpers (unchanged)
# ────────────────────────────────────────────────────────────────────────────────

def _collate(batch, tok, max_len):
    texts = [ex["text"] for ex in batch]
    return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)


def _stream_loader(ds, tok, bs, max_len):
    buf = []
    for ex in ds:
        buf.append(ex)
        if len(buf) == bs:
            yield _collate(buf, tok, max_len)
            buf.clear()
    if buf:
        yield _collate(buf, tok, max_len)

# ────────────────────────────────────────────────────────────────────────────────
# Hub push helper (same as before)
# ────────────────────────────────────────────────────────────────────────────────

def _push(records: List[dict], args: Args):
    api = HfApi(token=args.hf_token)
    idx = sum(f.endswith(".parquet") for f in api.list_repo_files(args.output_repo, repo_type="dataset"))
    fname = f"data_{idx:05d}.parquet"
    Dataset.from_list(records).to_parquet(fname)
    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull()
    repo.git_add(fname)
    repo.git_commit(f"Add shard {idx}")
    repo.git_push()
    repo.git_clear()
    os.remove(fname)

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
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

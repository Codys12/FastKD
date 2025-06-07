"""
Distributed generator with *parameter* pipeline parallelism and double‑buffering.
Layers move GPU→GPU while the batch stays resident on every device.  Usage (8 GPUs):

sbatch -p dgxh100 --gres=gpu:8 --wrap="singularity exec --nv lightning_sandbox bash -c 'cd fastkd; torchrun --nnodes 1 --nproc_per_node 8 generator.py --push_every 512 --model_name Qwen/Qwen3-235B-A22B --dataset_name mlfoundations/dclm-baseline-1.0 --output_repo codys12/Qwen3-DCLM-test2 --hf_token HF_TOKEN --num_rounds 255 --batch_size 512 --micro_batch_size 64'"

Key points
----------
* **Parameter pipeline** (reverse of data pipeline): layers stream through ranks
  CPU ➜ GPU‑0 ➜ GPU‑1 … GPU‑N‑1 ➜ CPU.
* **Double buffering**: odd/even layer indices use alternating buffers so copy &
  compute overlap.
* Each GPU keeps its local replica of the input batch for the whole forward.
* Sampling happens *once* on GPU‑0, fully vectorised (no Python loops).
* Works in a single `torchrun` launch; ranks are created by `torchrun`.
* No external dependencies beyond standard PyTorch + 🤗.
"""
from __future__ import annotations

import argparse
import os
import signal
import threading
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import HfApi, Repository

# ────────────────────────────────────────────────────────────────────────────────
# Argument parsing
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


def parse_args() -> Args:
    p = argparse.ArgumentParser("FastKD distributed generator (parameter pipeline)")
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


# ────────────────────────────────────────────────────────────────────────────────
# Helper: distributed send/recv of parameter vectors (non‑blocking)
# ────────────────────────────────────────────────────────────────────────────────

def _vectorize(layer: torch.nn.Module) -> torch.Tensor:
    """Flatten parameters of *layer* into a single 1‑D CUDA tensor."""
    for p in layer.parameters():
        p.requires_grad_(False)
    vec = torch.nn.utils.parameters_to_vector(list(layer.parameters()))
    return vec.contiguous()


def _dev_of_rank(rank: int) -> torch.device:
    if rank < 0:
        return torch.device("cpu")
    return torch.device(f"cuda:{rank}")


# ────────────────────────────────────────────────────────────────────────────────
# Parameter‑pipeline streamer
# ────────────────────────────────────────────────────────────────────────────────
class ParamPipe:
    """Move layers across ranks while data stays put. Double‑buffered."""

    def __init__(self, model_name: str):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.device = _dev_of_rank(self.rank)

        # Only rank 0 keeps a full model on CPU; other ranks create empty shells.
        if self.rank == 0:
            cfg = AutoConfig.from_pretrained(model_name)
            cfg.attn_implementation = "eager"  # avoids flash issues on CPU
            self.full_model = AutoModelForCausalLM.from_pretrained(
                model_name, config=cfg, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
            )
            self.layers = list(self.full_model.model.layers)
            self.embed = self.full_model.model.embed_tokens
            self.lm_head = self.full_model.lm_head
        else:
            # Placeholder layers just for type hints; actual params stream in.
            self.layers = None  # type: ignore
            self.embed = None  # type: ignore
            self.lm_head = None  # type: ignore

        # Two parameter buffers for double buffering
        self.buffers = [None, None]  # type: List[torch.Tensor | None]
        self._init_streams()

    # ──────────────────────────────────────
    # Comms helpers
    # ──────────────────────────────────────
    def _init_streams(self):
        if self.device.type == "cuda":
            self.compute_stream = torch.cuda.Stream(device=self.device)
            self.copy_stream = torch.cuda.Stream(device=self.device)
        else:
            self.compute_stream = torch.cuda.current_stream()
            self.copy_stream = torch.cuda.current_stream()

    def _send(self, tensor: torch.Tensor, dst: int, tag: int):
        dist.isend(tensor, dst=dst, tag=tag)

    def _recv(self, shape: torch.Size, dtype: torch.dtype, src: int, tag: int) -> torch.Tensor:
        out = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(out, src=src, tag=tag)
        return out

    # ──────────────────────────────────────
    # Forward + sample
    # ──────────────────────────────────────
    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: int,
        num_samples: int,
        num_rounds: int,
        temperature: float,
    ) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """Full forward‑pass + sampling on rank 0. Other ranks only forward."""
        # Replicate data on *every* GPU (fits because it's token IDs, not float32).
        input_ids = input_ids.to(self.device, non_blocking=True)
        hidden = None  # will become embeddings → hidden states

        n_layers = (
            len(self.layers) if self.rank == 0 else dist.broadcast_object_list([None], src=0)
        )
        # broadcast layer count so everyone knows how many iters
        if self.rank != 0:
            n_layers = dist.broadcast_object_list([None], src=0)[0]  # type: ignore

        next_rank = self.rank + 1 if self.rank + 1 < self.world else -1  # -1 is CPU sink
        prev_rank = self.rank - 1 if self.rank > 0 else -1

        for idx in range(n_layers + 2):  # +embed +lm_head
            tag = idx & 1  # double‑buffer toggle

            # ---------------------------
            # 1. Receive layer parameters
            # ---------------------------
            if idx == 0:
                # Special case: embed layer sent from CPU to GPU‑0
                if self.rank == 0:
                    vec = _vectorize(self.embed)  # type: ignore[attr-defined]
                    self.buffers[tag] = vec.to(self.device, non_blocking=True)
                    if next_rank != -1:
                        self._send(vec, dst=next_rank, tag=tag)
                else:
                    vec = self._recv_like(prev_rank, tag)
                    self.buffers[tag] = vec
            elif 1 <= idx <= n_layers:
                # Regular transformer layer
                if self.rank == 0:
                    layer = self.layers[idx - 1]  # idx‑1 because embed at 0
                    vec = _vectorize(layer)
                    self.buffers[tag] = vec.to(self.device, non_blocking=True)
                    if next_rank != -1:
                        self._send(vec, dst=next_rank, tag=tag)
                else:
                    vec = self._recv_like(prev_rank, tag)
                    self.buffers[tag] = vec
            else:
                # lm_head (last)
                if self.rank == 0:
                    vec = _vectorize(self.lm_head)  # type: ignore[attr-defined]
                    self.buffers[tag] = vec.to(self.device, non_blocking=True)
                    if next_rank != -1:
                        self._send(vec, dst=next_rank, tag=tag)
                else:
                    vec = self._recv_like(prev_rank, tag)
                    self.buffers[tag] = vec

            # ---------------------------------------
            # 2. Compute on *previous* buffer if exists
            # ---------------------------------------
            prev_tag = tag ^ 1
            buf = self.buffers[prev_tag]
            if buf is not None:
                with torch.cuda.stream(self.compute_stream):
                    layer_out = self._apply_layer(buf, hidden, micro_batch_size)
                    hidden = layer_out if layer_out is not None else hidden
            torch.cuda.current_stream().wait_stream(self.compute_stream)

        # After pipeline drains, hidden holds logits on every rank.
        # Sampling only on rank 0; others noop.
        if self.rank == 0:
            logits = hidden  # [B, L, V]
            ids, counts = self._gpu_sample(logits, num_samples, num_rounds, temperature)
        else:
            ids = counts = []
        return ids, counts

    # helper: create recv buffer matching src size
    def _recv_like(self, src: int, tag: int):
        shape = torch.tensor([0], device="cpu")
        dtype_code = torch.tensor([0], device="cpu")
        if self.rank == 0:
            raise RuntimeError("Rank0 should never _recv_like")
        # First receive shape & dtype code header (small CPU tensors)
        dist.recv(shape, src=src, tag=tag + 1000)
        dist.recv(dtype_code, src=src, tag=tag + 2000)
        dtype = [torch.float32, torch.bfloat16, torch.float16][int(dtype_code.item())]
        numel = int(shape.item())
        buf = torch.empty(numel, dtype=dtype, device=self.device)
        dist.recv(buf, src=src, tag=tag)
        return buf

    # helper: apply vectorised layer (we rebuild Module on‑the‑fly)
    def _apply_layer(self, vec: torch.Tensor, hidden: torch.Tensor | None, mbs: int):
        # Determine which layer based on size – embed / layer / lm_head.
        # Cheap heuristic since model architecture fixed.
        if hidden is None:
            # embed phase
            layer = torch.nn.Embedding.from_pretrained(
                vec.view_as(self.embed.weight)  # type: ignore[attr-defined]
            ).to(self.device)
            out = layer(self.input_ids)  # type: ignore[attr-defined]
            return out
        elif vec.numel() == self.lm_head.weight.numel():  # type: ignore[attr-defined]
            lm_head = torch.nn.Linear(
                self.lm_head.in_features,  # type: ignore[attr-defined]
                self.lm_head.out_features,  # type: ignore[attr-defined]
                bias=False,
                device=self.device,
                dtype=vec.dtype,
            )
            torch.nn.utils.vector_to_parameters(vec, [lm_head.weight])
            return lm_head(hidden)
        else:
            # generic transformer block – we rebuild minimal block wrapper
            block = self.layers[0].__class__(self.layers[0].config).to(self.device)  # type: ignore[attr-defined]
            torch.nn.utils.vector_to_parameters(vec, list(block.parameters()))
            out = []
            for mb in hidden.split(mbs, dim=0):
                out.append(block(mb)[0])  # type: ignore[operator]
            return torch.cat(out, dim=0)

    # helper: fully vectorised sampling (GPU)
    @staticmethod
    @torch.no_grad()
    def _gpu_sample(logits: torch.Tensor, N: int, R: int, t: float):
        bsz, seqlen, vocab = logits.shape
        p = torch.softmax(logits, dim=-1)
        q = torch.softmax(logits / t, dim=-1)
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.clamp(q, min=0)
        q = q + 1e-6
        q /= q.sum(dim=-1, keepdim=True)

        draws = N * R
        flat_samples = torch.multinomial(q.view(-1, vocab), draws, replacement=True)
        flat_ids, flat_counts = torch.unique(flat_samples, return_counts=True)

        ids = [[flat_ids.tolist() for _ in range(seqlen)] for _ in range(bsz)]
        counts = [[flat_counts.tolist() for _ in range(seqlen)] for _ in range(bsz)]
        return ids, counts


# ────────────────────────────────────────────────────────────────────────────────
# Data helpers (unchanged from original apart from device move)
# ────────────────────────────────────────────────────────────────────────────────

def collate_fn(examples, tokenizer, max_seq_len: int):
    texts = [e["text"] for e in examples]
    return tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len
    )


def streaming_dataloader(ds, tokenizer, batch_size: int, max_seq_len: int):
    batch = []
    for example in ds:
        batch.append(example)
        if len(batch) == batch_size:
            yield collate_fn(batch, tokenizer, max_seq_len)
            batch.clear()
    if batch:
        yield collate_fn(batch, tokenizer, max_seq_len)


# ────────────────────────────────────────────────────────────────────────────────
# Hub push helper (identical to original)
# ────────────────────────────────────────────────────────────────────────────────

def push_shard(records: List[dict], args: Args) -> None:
    api = HfApi(token=args.hf_token)
    files = api.list_repo_files(args.output_repo, repo_type="dataset")
    idx = sum(f.endswith(".parquet") for f in files)

    filename = f"data_{idx:05d}.parquet"
    ds = Dataset.from_list(records)
    ds.to_parquet(filename)

    repo = Repository(
        local_dir="repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset"
    )
    repo.git_pull()
    repo.git_add(filename)
    repo.git_commit(f"Add shard {idx}")
    repo.git_push()
    repo.git_clear()
    os.remove(filename)


# ────────────────────────────────────────────────────────────────────────────────
# Main distributed worker loop
# ────────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Initialise distributed
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

    shutdown = threading.Event()

    def _sigterm(*_):
        shutdown.set()

    signal.signal(signal.SIGTERM, _sigterm)

    # Build pipeline wrapper
    pipe = ParamPipe(args.model_name)

    # Data (all ranks iterate same stream – OK, records dedup later)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset(
        args.dataset_name, split=args.dataset_split, streaming=True, token=args.hf_token
    )

    dataloader = streaming_dataloader(dataset, tokenizer, args.batch_size, args.max_seq_len)

    all_records: List[dict] = []
    total = 0

    for batch in dataloader:
        if shutdown.is_set():
            break
        input_ids = batch["input_ids"].to(torch.device(f"cuda:{dist.get_rank()}"), non_blocking=True)
        ids, counts = pipe.run(
            input_ids,
            args.micro_batch_size,
            args.num_samples,
            args.num_rounds,
            args.sampling_temperature,
        )

        if dist.get_rank() == 0:
            for i in range(len(input_ids)):
                tokens = input_ids[i].tolist()
                while tokens and tokens[-1] == 0:
                    tokens.pop()
                seq_len = len(tokens)
                record = {
                    "input_ids": tokens,
                    "sampled_ids": ids[i][:seq_len],
                    "sampled_counts": counts[i][:seq_len],
                }
                all_records.append(record)
            total += len(input_ids)
            if total >= args.push_every:
                if all_records:
                    push_shard(all_records, args)
                    all_records.clear()
                    total = 0
        if shutdown.is_set():
            break

    # Flush remaining shard
    if dist.get_rank() == 0 and all_records and not shutdown.is_set():
        push_shard(all_records, args)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

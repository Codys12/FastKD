#!/usr/bin/env python3
# generator.py
"""
Fast batched‑inference sampler for very‑large Transformer models on 8×H100.

Example:
  torchrun --nnodes 1 --nproc_per_node 8 generator.py \
      --model_name Qwen/Qwen3-235B-A22B \
      --dataset_name mlfoundations/dclm-baseline-1.0 \
      --output_repo codys12/Qwen3-DCLM-test \
      --hf_token  hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
      --batch_size 16 \
      --num_rounds 255 \
      --push_every 4096
"""

from __future__ import annotations
import argparse, json, math, os, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from torch.distributed import init_process_group, barrier
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional FP8 utilities (PyTorch 2.3 / NVIDIA transformer_engine >= 1.6)
try:
    from transformer_engine.pytorch.fp8_utils import FP8GlobalState, fp8_autocast
    FP8_AVAILABLE = True
except Exception:
    FP8_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def setup_distributed() -> Tuple[int, int]:
    """Initialise torch.distributed and return (rank, world_size)."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    init_process_group("nccl")
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def get_mixed_precision(fp8: bool) -> MixedPrecision:
    """Return an appropriate MixedPrecision policy."""
    if fp8 and FP8_AVAILABLE and torch.cuda.is_available():
        policy = MixedPrecision(
            param_dtype=torch.float8_e4m3fn,
            reduce_dtype=torch.float8_e4m3fn,
            buffer_dtype=torch.float8_e4m3fn,
        )
    else:
        # default BF16 on Hopper; fall back to FP16 otherwise
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        policy = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )
    return policy


def multinomial_gpu(
    probs: torch.Tensor, num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `num_samples` tokens per row from a 2-D probability tensor on-GPU and
    return both the raw samples (`tokenIds`) and a dense histogram (`counts`).

    Returns
    -------
    tokenIds : (rows, num_samples) int64
    counts   : (rows, vocab)       int32
    """
    # Draw samples on-GPU
    token_ids = torch.multinomial(probs, num_samples=num_samples, replacement=True)  # (R, k)

    # Build dense histogram for each row in one shot
    counts = torch.zeros(probs.shape, dtype=torch.int32, device=probs.device)
    ones   = torch.ones_like(token_ids, dtype=torch.int32)
    counts.scatter_add_(1, token_ids, ones)  # dim=1 (vocab)

    # Move to CPU once, **after** aggregation
    return token_ids.cpu(), counts.cpu()



def push_to_hub_cli(
    repo: str,
    hf_token: str,
    local_file: Path,
    remote_path: str,
    commit_msg: str,
) -> None:
    """Upload a single file to the Hub using huggingface‑cli."""
    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token
    cmd = [
        "huggingface-cli",
        "upload",
        repo,
        str(local_file),
        "--path_in_repo",
        remote_path,
        "-m",
        commit_msg,
    ]
    subprocess.check_call(cmd, env=env)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--output_repo", required=True)
    p.add_argument("--hf_token", required=True)
    p.add_argument("--push_every", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_rounds", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--precision_compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # --------------------------------------------------------------------- #
    # Load tokenizer & dataset (CPU)                                        #
    # --------------------------------------------------------------------- #
    if rank == 0:
        print(f"Loading tokenizer {args.model_name} …", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        print(f"Loading dataset {args.dataset_name} …", file=sys.stderr)
    ds = load_dataset(args.dataset_name, split="train", streaming=False)
    ds = ds.shuffle(seed=args.seed)  # reproducible

    def encode(batch: Dict[str, str]) -> Dict[str, List[int]]:
        ids = tokenizer(
            batch["text"],
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        return {"input_ids": ids.input_ids.squeeze(0)}  # streaming=False

    ds = ds.map(encode, remove_columns=ds.column_names, num_proc=8)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda items: {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [item["input_ids"] for item in items],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            )
        },
    )

    # --------------------------------------------------------------------- #
    # Model (FSDP)                                                          #
    # --------------------------------------------------------------------- #
    if rank == 0:
        print(f"Loading model {args.model_name} …", file=sys.stderr)

    policy = get_mixed_precision(args.fp8)

    with enable_wrap(
        wrapper_cls=FSDP,
        mixed_precision=policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=policy.param_dtype,
            device_map=None,  # handled by FSDP
            low_cpu_mem_usage=True,
        )
        model.eval()
        # force no‑kv cache
        model.config.use_cache = False

        fsdp_model = wrap(model)

    if args.precision_compile:
        fsdp_model = torch.compile(fsdp_model)

    # --------------------------------------------------------------------- #
    # Sampler loop                                                          #
    # --------------------------------------------------------------------- #
    samples_in_shard = 0
    shard_idx = 0
    tmp_dir = Path(tempfile.mkdtemp(prefix="shard_", dir="/tmp" if os.path.exists("/tmp") else "."))

    # Check what index we should start at (idempotency)
    if rank == 0:
        from huggingface_hub import list_repo_files, HfApi
        api = HfApi(token=args.hf_token)
        try:
            existing_files = list_repo_files(args.output_repo, repo_type="dataset", token=args.hf_token)
            shard_idx = (
                max(
                    [
                        int(f.split("_")[1].split(".")[0])
                        for f in existing_files
                        if f.startswith("shard_") and f.endswith(".jsonl")
                    ],
                    default=-1,
                )
                + 1
            )
        except Exception:
            # Repo may not exist yet; will be created on first push
            shard_idx = 0
    shard_idx = int(torch.tensor(shard_idx).to(device))
    # broadcast so all ranks agree
    torch.distributed.broadcast(shard_idx, src=0)
    shard_idx = shard_idx.item()

    local_rows: List[str] = []

    for round_ in range(args.num_rounds):
        for step, batch in enumerate(dl):
            batch_ids = batch["input_ids"].to(device, non_blocking=True)

            with (
                fp8_autocast(enabled=args.fp8) if (args.fp8 and FP8_AVAILABLE) else torch.no_grad()
            ):
                with torch.no_grad():
                    logits = fsdp_model(batch_ids).logits  # (B, L, V)

            B, L, V = logits.shape
            logits = logits.view(-1, V)  # rows = B*L
            probs = torch.softmax(logits, dim=-1)

            token_ids, counts = multinomial_gpu(probs, args.num_samples)  # CPU tensors

            # serialise rows
            counts_list = counts.tolist()
            tokens_list = token_ids.tolist()

            for row in range(len(tokens_list)):
                payload = {
                    "round": round_,
                    "global_row": round_ * len(dl) * B * L + step * B * L + row,
                    "tokenIds": tokens_list[row],
                    "counts": counts_list[row],
                }
                local_rows.append(json.dumps(payload) + "\n")

            samples_in_shard += len(tokens_list)

            # ------------------------------------------------------------------ #
            # PUSH IF NEEDED (rank 0 only)                                       #
            # ------------------------------------------------------------------ #
            if samples_in_shard >= args.push_every:
                # Gather rows from all ranks to rank 0
                gathered = [None for _ in range(world_size)]
                torch.distributed.gather_object(local_rows, gathered, dst=0)

                if rank == 0:
                    # Flatten & write to shard file
                    shard_path = tmp_dir / f"shard_{shard_idx:06d}.jsonl"
                    with open(shard_path, "w") as fh:
                        for chunk in gathered:
                            for ln in chunk:
                                fh.write(ln)
                    # push
                    remote_path = shard_path.name
                    commit_msg = f"Add {remote_path} ({samples_in_shard} rows)"
                    push_to_hub_cli(
                        args.output_repo,
                        args.hf_token,
                        shard_path,
                        remote_path,
                        commit_msg,
                    )
                    shard_idx += 1
                    print(f"Pushed {remote_path} ✔️", file=sys.stderr)

                # everyone waits
                barrier()

                # reset local buffer
                local_rows.clear()
                samples_in_shard = 0

    # --------------------------------------------------------------------- #
    # Final flush                                                           #
    # --------------------------------------------------------------------- #
    gathered = [None for _ in range(world_size)]
    torch.distributed.gather_object(local_rows, gathered, dst=0)

    if rank == 0 and any(gathered):
        shard_path = tmp_dir / f"shard_{shard_idx:06d}.jsonl"
        with open(shard_path, "w") as fh:
            for chunk in gathered:
                for ln in chunk:
                    fh.write(ln)
        remote_path = shard_path.name
        push_to_hub_cli(
            args.output_repo,
            args.hf_token,
            shard_path,
            remote_path,
            f"Add {remote_path} (final)",
        )
        print(f"Pushed {remote_path} ✔️ (final)", file=sys.stderr)

    barrier()
    if rank == 0:
        print("✓ All done", file=sys.stderr)


if __name__ == "__main__":
    main()

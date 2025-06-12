#!/usr/bin/env python
# generator.py
#
# ======================================================================
# FastKD Logit Sampler
# ----------------------------------------------------------------------
#  * 8‑GPU distributed, no‑KV causal‑LM inference
#  * multinomial sampling of logits → counts + tokenIds
#  * streaming dataset
#  * incremental pushes to the Hub with huggingface‑cli
# ======================================================================

import os
import time
import json
import uuid
import argparse
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from accelerate import Accelerator

# ----------------------------------------------------------------------
# Helper: upload a file to Hub with huggingface‑cli (no git clone needed)
# ----------------------------------------------------------------------
def hf_upload(local_path: Path, repo: str, token: str, commit_msg: str):
    """
    Parameters
    ----------
    local_path : Path
        File to be uploaded.
    repo : str
        Repo slug, e.g. "codys12/Qwen3-DCLM-test".
    token : str
        HF user / org token.
    commit_msg : str
        Commit message (shown on the dataset page).
    """
    cmd = [
        "huggingface-cli",
        "upload",
        repo,
        str(local_path),
        "--repo-type", "dataset",
        "--commit-message", commit_msg,
        "--token", token,
    ]
    # `upload` exits with return‑code 0 on success; we enforce that:
    subprocess.run(cmd, check=True)
    # Optional: remove the temp file once upload completes
    local_path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# Collate: pure tensor batch (already tokenised) → device
# ----------------------------------------------------------------------
def collate_fn(batch, pad_id):
    """
    batch : list of dicts with keys "input_ids" and "attention_mask"
    Pads to the longest in batch.
    """
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids  = []
    attn_masks = []
    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]
        pad_needed = max_len - len(ids)
        if pad_needed:
            ids  = ids  + [pad_id] * pad_needed
            mask = mask + [0]      * pad_needed
        input_ids.append(ids)
        attn_masks.append(mask)
    return {
        "input_ids":      torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FastKD logit sampler")
    # core
    parser.add_argument("--model_name",    type=str, required=True)
    parser.add_argument("--dataset_name",  type=str, required=True)
    parser.add_argument("--output_repo",   type=str, required=True)
    parser.add_argument("--hf_token",      type=str, required=True)
    # run control
    parser.add_argument("--batch_size",    type=int, default=16)
    parser.add_argument("--num_rounds",    type=int, default=1_000_000)
    parser.add_argument("--push_every",    type=int, default=4_096,
                        help="push after this many *samples* (not steps)")
    # sampling
    parser.add_argument("--num_samples",   type=int, default=64,
                        help="multinomial draws per token")
    parser.add_argument("--temperature",   type=float, default=1.0)
    # misc
    parser.add_argument("--max_seq_len",   type=int, default=2_048)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    # -----------------  Accelerator handles torchrun/SLURM env ----------
    accelerator = Accelerator()
    rank      = accelerator.process_index
    world     = accelerator.num_processes
    main_proc = accelerator.is_main_process

    # -----------------  NCCL tweaks for H100 NVLink ----------------------
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")    # enable NVLink P2P
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")

    # -----------------  Tokeniser (CPU; sent to GPU after collation) ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    pad_id    = tokenizer.pad_token_id or tokenizer.eos_token_id

    # -----------------  Model (sharded over 8×H100) ---------------------
    torch_dtype = torch.bfloat16  # H100 bf16 is fastest
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="balanced_low_0",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval().requires_grad_(False)

    # Optional (PyTorch 2): compile once for all
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    # Accelerate wraps model and any DataLoader we create later
    model = accelerator.prepare(model)

    # -----------------  Streaming dataset + deterministic sharding ------
    stream = load_dataset(
        args.dataset_name,
        split="train",
        streaming=True,
    )

    # Every GPU sees only its shard; avoids overlap without communication
    stream = stream.shuffle(buffer_size=10_000, seed=42)
    stream = stream.shard(num_shards=world, index=rank)

    # We add a simple map → tokenise as PyTorch tensors immediately
    def tok_fn(batch):
        # Each `batch` is a single example in streaming mode
        txt = batch["text"] if isinstance(batch, dict) else batch
        enc = tokenizer(
            txt,
            truncation=True,
            max_length=args.max_seq_len,
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    stream = stream.map(tok_fn)

    loader = DataLoader(
        stream,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,              # streaming ⇒ workers don’t help
        pin_memory=False,
    )
    loader = accelerator.prepare(loader)

    # -----------------  Buffers & book‑keeping --------------------------
    push_buffer = []                   # will hold JSON‑serialisable dicts
    samples_seen = 0                   # samples processed *by this rank*
    global_step  = 0                   # increments once per batch

    # -----------------  MAIN LOOP ---------------------------------------
    start = time.time()
    for _ in range(args.num_rounds):
        for batch in loader:
            global_step += 1
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch_dtype):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                # logits shape: [B, L, V]
                logits = out.logits / args.temperature

            # ------------------------------------------------------------
            # GPU multinomial sampling
            # ------------------------------------------------------------
            B, L, V = logits.shape
            flat_p  = torch.softmax(
                logits.view(-1, V), dim=-1
            )                           # shape [B*L, V]
            draws  = torch.multinomial(
                flat_p, args.num_samples, replacement=True
            )                           # [B*L, S]
            # Convert to counts with efficient scatter‑add
            eye     = torch.ones_like(draws, dtype=torch.int32)
            counts  = torch.zeros(
                flat_p.size(0), V, dtype=torch.int32, device=logits.device
            )
            # scatter: counts[row_idx, token_id] += 1
            row_idx = torch.arange(
                flat_p.size(0), device=logits.device
            ).unsqueeze(1).expand_as(draws)
            counts = counts.index_put_(
                (row_idx, draws), eye, accumulate=True
            )
            counts = counts.view(B, L, V)
            draws  = draws.view(B, L, args.num_samples)

            # ------------------------------------------------------------
            # Gather across GPUs → rank 0
            # ------------------------------------------------------------
            gathered_counts = accelerator.gather(counts)
            gathered_draws  = accelerator.gather(draws)

            if main_proc:
                for c, d in zip(gathered_counts, gathered_draws):
                    push_buffer.append(
                        {
                            "counts":   c.cpu().tolist(),
                            "tokenIds": d.cpu().tolist(),
                        }
                    )

            samples_seen += batch["input_ids"].size(0)

            # ------------------------------------------------------------
            # Periodic push
            # ------------------------------------------------------------
            if main_proc and samples_seen >= args.push_every:
                fname = (
                    Path("./")
                    / f"chunk_{uuid.uuid4().hex}.jsonl"
                )
                with fname.open("w") as f:
                    for item in push_buffer:
                        f.write(json.dumps(item) + "\n")

                commit_msg = (
                    f"Add {len(push_buffer)} samples "
                    f"(processed={samples_seen}, step={global_step})"
                )
                hf_upload(
                    local_path=fname,
                    repo=args.output_repo,
                    token=args.hf_token,
                    commit_msg=commit_msg,
                )
                push_buffer.clear()
                samples_seen = 0  # reset local counter

        # ----------------------------------------------------------------
        # End of one full DataLoader sweep (rare with `streaming=True`)
        # ----------------------------------------------------------------

    # -----------------  Final flush  ------------------------------------
    if main_proc and push_buffer:
        fname = Path("./") / f"chunk_{uuid.uuid4().hex}.jsonl"
        with fname.open("w") as f:
            for item in push_buffer:
                f.write(json.dumps(item) + "\n")
        commit_msg = f"Final flush: {len(push_buffer)} samples"
        hf_upload(
            local_path=fname,
            repo=args.output_repo,
            token=args.hf_token,
            commit_msg=commit_msg,
        )

    if main_proc:
        elapsed = time.time() - start
        print(f"Finished. Total time: {elapsed/60:.2f} min")


# ----------------------------------------------------------------------
# Entry‑point guard (makes script torchrun‑safe)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

#!/usr/bin/env python
# generator_pippy.py

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
from accelerate.inference import prepare_pippy


def hf_upload(local_path: Path, repo: str, token: str, commit_msg: str):
    cmd = [
        "huggingface-cli", "upload", repo, str(local_path),
        "--repo-type", "dataset",
        "--commit-message", commit_msg,
        "--token", token,
    ]
    subprocess.run(cmd, check=True)
    local_path.unlink(missing_ok=True)


def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attn_masks = []
    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]
        pad_needed = max_len - len(ids)
        if pad_needed:
            ids += [pad_id] * pad_needed
            mask += [0] * pad_needed
        input_ids.append(ids)
        attn_masks.append(mask)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser(description="FastKD logit sampler with PiPPy")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_repo", type=str, required=True)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_rounds", type=int, default=1_000_000)
    parser.add_argument("--push_every", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    accelerator = Accelerator()

    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval().requires_grad_(False)

    dummy_input_ids = torch.zeros((1, min(8, args.max_seq_len)), dtype=torch.long)
    dummy_attn_mask = torch.ones_like(dummy_input_ids)

    model = prepare_pippy(
        model,
        split_points="auto",
        example_kwargs={
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attn_mask,
        },
        gather_output=True,
    )

    stream = load_dataset(args.dataset_name, split="train", streaming=True)
    stream = stream.shuffle(buffer_size=10000, seed=42)
    stream = stream.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)

    def tok_fn(batch):
        txt = batch["text"] if isinstance(batch, dict) else batch
        enc = tokenizer(txt, truncation=True, max_length=args.max_seq_len)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    stream = stream.map(tok_fn)

    loader = DataLoader(
        stream,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,
        pin_memory=False,
    )
    loader = accelerator.prepare(loader)

    push_buffer = []
    samples_seen = 0
    global_step = 0

    start = time.time()
    for _ in range(args.num_rounds):
        for batch in loader:
            global_step += 1

            if accelerator.is_main_process:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
            else:
                input_ids = attention_mask = None

            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch_dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits / args.temperature

            B, L, V = logits.shape
            flat_p = torch.softmax(logits.view(-1, V), dim=-1)
            draws = torch.multinomial(flat_p, args.num_samples, replacement=True)
            eye = torch.ones_like(draws, dtype=torch.int32)
            counts = torch.zeros(flat_p.size(0), V, dtype=torch.int32, device=logits.device)
            row_idx = torch.arange(flat_p.size(0), device=logits.device)[:, None].expand_as(draws)
            counts.index_put_((row_idx, draws), eye, accumulate=True)
            counts = counts.view(B, L, V)
            draws = draws.view(B, L, args.num_samples)

            if accelerator.is_main_process:
                for c, d in zip(counts, draws):
                    push_buffer.append({"counts": c.cpu().tolist(), "tokenIds": d.cpu().tolist()})

                samples_seen += input_ids.size(0)

                if samples_seen >= args.push_every:
                    fname = Path("./") / f"chunk_{uuid.uuid4().hex}.jsonl"
                    with fname.open("w") as f:
                        for item in push_buffer:
                            f.write(json.dumps(item) + "\n")
                    commit_msg = f"Add {len(push_buffer)} samples (processed={samples_seen}, step={global_step})"
                    hf_upload(fname, args.output_repo, args.hf_token, commit_msg)
                    push_buffer.clear()
                    samples_seen = 0

    if accelerator.is_main_process and push_buffer:
        fname = Path("./") / f"chunk_{uuid.uuid4().hex}.jsonl"
        with fname.open("w") as f:
            for item in push_buffer:
                f.write(json.dumps(item) + "\n")
        commit_msg = f"Final flush: {len(push_buffer)} samples"
        hf_upload(fname, args.output_repo, args.hf_token, commit_msg)

    if accelerator.is_main_process:
        elapsed = time.time() - start
        print(f"Finished. Total time: {elapsed/60:.2f} min")


if __name__ == "__main__":
    main()

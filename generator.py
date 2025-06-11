#!/usr/bin/env python3
# ring_kd_generator.py  –  June 2025, pipeline‑parallel rewrite
#
#   • True model sharding across GPUs (no weight movement)
#   • 2‑way micro‑batch double buffering for steady‑state utilisation
#   • Same CLI and sampling / Hub‑push semantics as the v4 “ring” script
#
# launch example:
#   torchrun --nproc_per_node=8 ring_kd_generator.py \
#            --model_name meta‑llama/Meta‑Llama‑3‑70B \
#            --dataset_name c4 --output_repo my_user/c4‑llama‑kd
#
# NOTE: --offload_strategy is now a no‑op and kept for backwards compatibility
# ---------------------------------------------------------------------------

from __future__ import annotations
import argparse, os, signal, sys, threading, traceback, math, inspect
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer)

###############################################################################
# CLI                                                                         #
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
    offload_strategy: str          # <- kept but ignored
    hf_token: Optional[str] = None
    debug: bool = False


def parse_args() -> Args:
    p = argparse.ArgumentParser("Pipeline‑parallel KD generator (v5)")
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
    p.add_argument("--offload_strategy", default="none",
                   help="NO‑OP – kept for script compatibility")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--debug", action="store_true")
    return Args(**vars(p.parse_args()))

###############################################################################
# Utils                                                                       #
###############################################################################

def get_logger(rank: int, enabled: bool):
    def log(msg: str):
        if enabled:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts} | r{rank}] {msg}", flush=True)
    return log


def _pin_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.pin_memory()
    return batch


def collate_fn(examples, tok, max_len):
    enc = tok([e["text"] for e in examples],
              return_tensors="pt",
              padding=True,
              truncation=True,
              max_length=max_len)
    return _pin_batch(enc)


def stream_loader(ds, tok, bs, max_len):
    buf = []
    for ex in ds:
        buf.append(ex)
        if len(buf) == bs:
            yield collate_fn(buf, tok, max_len)
            buf.clear()
    if buf:
        yield collate_fn(buf, tok, max_len)

###############################################################################
# Model partitioning helper                                                   #
###############################################################################

def build_sharded_model(args: Args,
                        rank: int,
                        world: int,
                        device: torch.device,
                        dbg):
    """
    Loads only the layer subset required for *this* rank on its GPU.
    All other layers stay as *meta* tensors so memory stays tiny.
    """
    dbg("Loading model config")
    cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    cfg.attn_implementation = "eager"         # safe everywhere

    # How many transformer blocks does the model have?
    n_layers = getattr(cfg, "num_hidden_layers",
                       getattr(cfg, "n_layer", None))
    if n_layers is None:
        raise ValueError("Could not determine the number of layers – "
                         "please file an issue for this model.")

    # ------------------------------------------------------------------ #
    # 1) Create an empty-weight skeleton (all params on 'meta')          #
    # ------------------------------------------------------------------ #
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from huggingface_hub import snapshot_download

    dbg("Building empty-weight skeleton")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)

    # ------------------------------------------------------------------ #
    # 2) Build per-rank device_map                                       #
    # ------------------------------------------------------------------ #
    per_stage = math.ceil(n_layers / world)
    start = rank * per_stage
    end   = min((rank + 1) * per_stage, n_layers)

    # Default: keep weights on 'meta' (i.e. do not load here)
    device_map: Dict[str, torch.device | str] = {"": "meta"}
    if rank == 0:
        device_map["model.embed_tokens"] = device
    if rank == world - 1:
        device_map["lm_head"] = device
    for i in range(start, end):
        device_map[f"model.layers.{i}"] = device

    # ------------------------------------------------------------------ #
    # 3) Ensure checkpoint is local, then load only the shards we need   #
    # ------------------------------------------------------------------ #
    dbg("Resolving checkpoint locally (snapshot_download)")
    ckpt_dir = snapshot_download(
        repo_id=args.model_name,
        token=args.hf_token,
        local_files_only=False,
        # we only care about weight files; skip big tokenizer artefacts
        ignore_patterns=["*.json", "*.txt", "*.md", "tokenizer.*"]
    )

    dbg(f"Loading checkpoint shards for layers [{start}, {end}) on {device}")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=ckpt_dir,
        device_map=device_map,
        dtype=torch.bfloat16,
        no_split_module_classes=[
            "LlamaDecoderLayer", "OPTDecoderLayer", "GPTNeoXLayer",
            "MistralDecoderLayer", "BloomBlock"
        ],
    )

    model.eval()
    return model, start, end

###############################################################################
# Push helpers (unchanged)                                                    #
###############################################################################

def push_shard(records: List[dict], args: Args, dbg):
    dbg(f"Pushing {len(records):,d} records to hub")
    api = HfApi(token=args.hf_token)
    existing = api.list_repo_files(args.output_repo, repo_type="dataset")
    idx = sum(f.endswith(".parquet") for f in existing)

    fname = f"data_{idx:05d}.parquet"
    Dataset.from_list(records).to_parquet(fname)

    from huggingface_hub import CommitOperationAdd
    api.create_repo(args.output_repo, repo_type="dataset", exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fname)]
    api.create_commit(args.output_repo, repo_type="dataset",
                      operations=ops,
                      commit_message=f"add shard {idx}")
    os.remove(fname)


def gather_and_push(local: List[dict], args: Args,
                    rank: int, world: int, dbg):
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        merged: List[dict] = []
        for part in gathered:            # type: ignore[arg-type]
            merged.extend(part)
        push_shard(merged, args, dbg)

###############################################################################
# Worker                                                                      #
###############################################################################

def worker(args: Args):
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dbg = get_logger(rank, args.debug)

    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())

    try:
        engine = PipelineEngine(args, rank, world, dbg)

        tok = AutoTokenizer.from_pretrained(args.model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        # Only rank‑0 reads the dataset; all others just run the pipeline
        if rank == 0:
            ds = load_dataset(args.dataset_name,
                              split=args.dataset_split,
                              streaming=True,
                              token=args.hf_token)
            loader = stream_loader(ds, tok, args.batch_size, args.max_seq_len)
        else:
            loader = None  # type: ignore[assignment]

        local_records: List[dict] = []
        seen = 0

        batch_idx = 0
        while True:
            if stop_evt.is_set():
                dbg("SIGTERM received – breaking loop")
                break

            # Rank 0 fetches the next batch; other ranks pass dummy
            batch = next(loader, None) if rank == 0 else None
            # Broadcast a “continue / stop” flag – 1 if more data, 0 otherwise
            more_flag = torch.tensor(1 if batch is not None else 0,
                                     dtype=torch.int8,
                                     device=engine.device)
            dist.broadcast(more_flag, src=0)
            if more_flag.item() == 0:
                break

            dbg(f"Batch {batch_idx}")
            ids, cnts = engine.pipeline_forward(batch)
            if rank == 0:
                input_ids = batch["input_ids"]
                for i in range(len(input_ids)):
                    toks = input_ids[i].tolist()
                    while toks and toks[-1] == tok.pad_token_id:
                        toks.pop()
                    seq_len = len(toks)
                    local_records.append({
                        "input_ids": toks,
                        "sampled_ids": ids[i][:seq_len],
                        "sampled_counts": cnts[i][:seq_len],
                    })
                seen += len(input_ids)

                if seen >= args.push_every:
                    dbg("push_every reached – gathering to rank‑0")
                    gather_and_push(local_records, args, rank, world, dbg)
                    local_records.clear()
                    seen = 0

            batch_idx += 1

        # Final flush
        if rank == 0 and local_records:
            dbg("Final flush")
            gather_and_push(local_records, args, rank, world, dbg)

    except Exception:
        dbg("Unhandled exception – printing traceback")
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    dbg("Graceful shutdown")
    dist.destroy_process_group()

###############################################################################
# Entry‑point                                                                 #
###############################################################################

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    worker(args)

if __name__ == "__main__":
    main()

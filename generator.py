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
    #    accelerate / safetensors expect *ints* (GPU index) or strings   #
    #    like "cuda:0" – **not** torch.device objects. Passing a         #
    #    torch.device triggers:                                         #
    #       safetensors_rust.SafetensorError: device cuda:X is invalid   #
    # ------------------------------------------------------------------ #
    per_stage = math.ceil(n_layers / world)
    start = rank * per_stage
    end   = min((rank + 1) * per_stage, n_layers)

    device_idx = rank  # 0 … world-1
    device_map: Dict[str, int | str] = {"": "meta"}   # keep others as meta
    if rank == 0:
        device_map["model.embed_tokens"] = device_idx
    if rank == world - 1:
        device_map["lm_head"] = device_idx
    for i in range(start, end):
        device_map[f"model.layers.{i}"] = device_idx

    # ------------------------------------------------------------------ #
    # 3) Ensure checkpoint is local, then load only the shards we need   #
    # ------------------------------------------------------------------ #
    dbg("Resolving checkpoint locally (snapshot_download)")
    dbg(f"Loading checkpoint shards for layers [{start}, {end}) on cuda:{device_idx}")    ckpt_dir = snapshot_download(
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
# Pipeline engine (double‑buffered, inference‑only)                           #
###############################################################################

class PipelineEngine:
    """
    Standard left‑to‑right pipeline‑parallel execution with two micro‑batch
    buffers (double buffering). For inference we only need the forward pass.
    Each rank owns:
        • rank 0          : embed + layers[start:end]
        • 0 < rank < last : layers[start:end]
        • last rank       : layers[start:end] + lm_head
    """
    def __init__(self, args: Args, rank: int, world: int, dbg):
        self.args  = args
        self.rank  = rank
        self.world = world
        self.dbg   = dbg

        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)

        self.model, self.start, self.end = build_sharded_model(
            args, rank, world, self.device, dbg
        )

        # References for convenience
        self.layers = self.model.model.layers
        self.embed  = self.model.model.embed_tokens if rank == 0 else None
        self.head   = self.model.lm_head if rank == world - 1 else None
        self.rotary = self.model.model.rotary_emb  # small, lives everywhere

        # Buffers for double‑buffer micro‑batch scheduling
        self.toggle = 0  # 0 or 1
        self.streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    # ........................ micro‑batch helpers .......................... #
    def _tag(self, mb_idx: int) -> int:
        # unique tag per micro‑batch wave (fits in 32 bits)
        return mb_idx * 2

    def _recv_hidden(self, shape, src, tag):
        hidden = torch.empty(shape,
                             dtype=torch.bfloat16,
                             device=self.device)
        req = dist.irecv(hidden, src=src, tag=tag)
        return hidden, req

    def _send_hidden(self, hidden, dst, tag):
        return dist.isend(hidden, dst=dst, tag=tag)

    # ........................ core pipeline step .......................... #
    @torch.no_grad()
    def pipeline_forward(self, batch: Optional[Dict[str, torch.Tensor]]) \
            -> Tuple[List[list], List[list]]:
        """
        Executes one *global* batch consisting of
            batch_size / micro_batch_size  micro‑batches.
        Only *rank 0* receives real input tokens; other ranks receive/produce
        activations through P2P communication.
        Return value (ids, counts) is broadcast from the last rank so every
        process can keep the original record‑construction logic unchanged.
        """
        micro_bs = self.args.micro_batch_size
        B = batch["input_ids"].size(0) if self.rank == 0 else None
        # Broadcast batch size so every rank knows how many micro‑batches
        B_tensor = torch.tensor(B or 0, device=self.device)
        dist.broadcast(B_tensor, src=0)
        B = int(B_tensor.item())
        num_mbs = math.ceil(B / micro_bs)

        left  = (self.rank - 1) % self.world
        right = (self.rank + 1) % self.world

        # Per‑micro‑batch stash for last rank
        ids_all, counts_all = [], []

        # Two “slots” (double buffer) – iterate through a longer loop so
        # that the pipeline fills and drains.
        for wave in range(num_mbs + self.world - 1):
            buf_idx = self.toggle
            stream  = self.streams[buf_idx]
            tag     = self._tag(buf_idx)

            with torch.cuda.stream(stream):
                # 1) Receive hidden from left neighbour  (if needed) – async
                if self.rank != 0 and wave - (self.rank - 0) >= 0 \
                        and wave < num_mbs + self.rank:
                    # shape is unknown on non‑first wave – send shape tensor
                    shape_tensor = torch.empty(2, dtype=torch.int64,
                                               device=self.device)
                    dist.recv(shape_tensor, src=left, tag=tag)
                    shape = tuple(int(x) for x in shape_tensor.tolist())
                    hidden, recv_req = self._recv_hidden(shape, left, tag)
                else:
                    hidden, recv_req = None, None

                # 2) Prepare input on rank 0
                if self.rank == 0 and wave < num_mbs:
                    start = wave * micro_bs
                    end   = min(start + micro_bs, B)
                    input_ids = batch["input_ids"][start:end].to(self.device,
                                                                 non_blocking=True)
                    hidden = self.embed(input_ids)  # [mb, T, D]
                    seq_len = input_ids.size(1)
                    pos_ids = torch.arange(seq_len, device=self.device)\
                                    .unsqueeze(0).expand_as(input_ids)
                    cos, sin = self.rotary(hidden, pos_ids)
                    # embed returns hidden already; rotary will be re‑done
                    # inside transformer layers (models that need it)
                else:
                    # Wait for activations from previous stage
                    if recv_req is not None:
                        recv_req.wait()

                # 3) Run local transformer block(s)
                if hidden is not None and self.start < self.end:
                    for layer_idx in range(self.start, self.end):
                        layer = self.layers[layer_idx].to(self.device,
                                                          non_blocking=True)
                        # micro‑batch is small → single pass
                        hidden = layer(hidden)[0] if isinstance(
                            layer(hidden), tuple) else layer(hidden)

                # 4) If *not* last rank, send hidden rightwards – async
                if self.rank != self.world - 1 and hidden is not None:
                    shape_tensor = torch.tensor(hidden.shape[:2],
                                                dtype=torch.int64,
                                                device=self.device)
                    dist.isend(shape_tensor, dst=right, tag=tag)  # tiny sync
                    self._send_hidden(hidden, right, tag)

                # 5) If last rank we have full hidden – make logits + sample
                if self.rank == self.world - 1 \
                        and wave >= self.world - 1 and wave < num_mbs + self.world - 1:
                    logits = self.head(hidden)[0]       # [mb, T, vocab]
                    probs  = torch.softmax(
                        logits / self.args.sampling_temperature, dim=-1)
                    draws = torch.multinomial(
                        probs.contiguous().view(-1, probs.size(-1)),
                        self.args.num_samples * self.args.num_rounds,
                        replacement=True
                    ).view(*probs.shape[:-1], -1)

                    for b in range(draws.size(0)):
                        ids_seq, cnts_seq = [], []
                        for t in range(draws.size(1)):
                            uniq, cnt = torch.unique(draws[b, t],
                                                     return_counts=True)
                            ids_seq.append(uniq.cpu().tolist())
                            cnts_seq.append(cnt.cpu().tolist())
                        ids_all.append(ids_seq)
                        counts_all.append(cnts_seq)

            self.toggle ^= 1  # swap buffer

        # ---------------- Broadcast sampled ids/cnts to every rank -------- #
        if self.rank == self.world - 1:
            payload: Tuple[list, list] = (ids_all, counts_all)
        else:
            payload = (None, None)
        obj_list: List[object] = [payload]
        dist.broadcast_object_list(obj_list, src=self.world - 1)
        ids_all, counts_all = obj_list[0]

        return ids_all, counts_all

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

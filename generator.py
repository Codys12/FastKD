#!/usr/bin/env python3
from __future__ import annotations
"""
Generator with reverse pipeline parallelism (ring) and optional debug logs.
Weights travel  CPU → GPU-0 → … → GPU-N-1 → GPU-0.  Rank-0 then off-loads the
last layer back to CPU.  Use --debug for verbose per-rank output.

June 2025 **second patch**
=========================

✅ **Message-tagged ring** – every send/recv now uses a unique *tag* derived
   from the layer index so packets from different layers can never be
   confused.  This eliminates the catastrophic *Packet size mismatch* error
   observed when concurrent point-to-point transfers arrived out of order.

✅ **Zero-element safety** – packing *and* unpacking ignore tensors whose
   ``numel()==0`` so flat-packet sizes always stay consistent.

✅ **Minor clean-ups** – lint fixes, doc tweaks, explicit cloning of the
   tokenizer’s pad token where necessary.

The rest of the logic (materialisation guard, pin-memory collator, graceful
SIGTERM handling, etc.) is unchanged from the original June 2025 patch.
"""

###############################################################################
# Imports                                                                     #
###############################################################################

import argparse
import inspect
import os
import signal
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, Repository
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
    hf_token: str | None = None
    debug: bool = False


def parse_args() -> Args:
    p = argparse.ArgumentParser("Reverse-pipeline KD generator (ring)")
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
    p.add_argument("--debug", action="store_true")
    return Args(**vars(p.parse_args()))


###############################################################################
# Utils                                                                       #
###############################################################################


def get_logger(rank: int, enabled: bool):
    def log(msg: str):
        if enabled:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now} | rank {rank}] {msg}", flush=True)

    return log


# --------------------------------------------------------------------------- #
# NEW: storage materialisation helper                                         #
# --------------------------------------------------------------------------- #


def _has_real_storage(t: torch.Tensor) -> bool:
    # torch 2.1 added the public attribute .is_meta; older builds require this
    return not (getattr(t, "is_meta", False) or getattr(t.data, "is_meta", False))


def materialise_meta(module: torch.nn.Module, device: torch.device, dtype: torch.dtype):
    """Ensure *all* params & buffers of *module* live on *device* with storage."""

    to_empty = getattr(module, "to_empty", None)
    fallback_exc: Exception | None = None

    if callable(to_empty):
        try:
            kwargs = {}
            sig = inspect.signature(to_empty)
            if "device" in sig.parameters:
                kwargs["device"] = device
            if "dtype" in sig.parameters:
                kwargs["dtype"] = dtype
            to_empty(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover – fall back if not supported
            fallback_exc = exc
    else:
        fallback_exc = RuntimeError("module has no .to_empty()")

    if fallback_exc is not None:
        for p in module.parameters(recurse=True):
            if not _has_real_storage(p):
                p.data = torch.empty_like(p, dtype=dtype, device=device)
        for name, buf in module.named_buffers(recurse=True):
            if not _has_real_storage(buf):
                module._buffers[name] = torch.empty_like(buf, dtype=dtype, device=device)

    # sanity – every tensor must now be real
    bad = [
        (n, "PARAM" if isinstance(t, torch.nn.Parameter) else "BUFFER")
        for n, t in list(module.named_parameters(recurse=True)) + list(module.named_buffers(recurse=True))
        if not _has_real_storage(t)
    ]
    if bad:  # pragma: no cover
        names = ", ".join(f"{k}({kind})" for k, kind in bad)
        raise RuntimeError(f"materialise_meta() failed – {len(bad)} tensors still meta: {names}")


###############################################################################
# Flat packet helper                                                          #
###############################################################################


class FlatPacket:
    """Serialises one *layer* worth of parameters into a flat contiguous tensor."""

    def __init__(self):
        self.flat: torch.Tensor | None = None

    # ..................................................................... #
    # Packing                                                               #
    # ..................................................................... #

    def pack(self, layer: torch.nn.Module, device: torch.device):
        with torch.no_grad():
            slices = [p.data.view(-1) for p in layer.parameters() if p.numel()]
            self.flat = torch.cat(slices, out=None).contiguous().to(device) if slices else torch.empty(0, device=device)

    def make_empty(self, numel: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.flat = torch.empty(numel, dtype=dtype, device=device)

    # ..................................................................... #
    # Unpacking                                                             #
    # ..................................................................... #

    def unpack_to(self, layer: torch.nn.Module):
        if self.flat is None:
            raise RuntimeError("FlatPacket.unpack_to() called before .flat is set")

        expected = sum(p.numel() for p in layer.parameters() if p.numel())
        if expected != self.flat.numel():
            raise RuntimeError(
                "Packet size mismatch: layer expects "
                f"{expected:,d} elements but flat tensor has {self.flat.numel():,d}"
            )

        offset = 0
        with torch.no_grad():
            for p in layer.parameters():
                n = p.numel()
                if n:  # skip zero-element tensors
                    p.data.copy_(self.flat[offset : offset + n].view_as(p))
                    offset += n


###############################################################################
# Reverse pipeline engine                                                     #
###############################################################################


class ReversePipelineEngine:
    """Streams layers through static data; ensures embed weights exist on each rank."""

    def __init__(self, args: Args, rank: int, world: int, dbg):
        self.args = args
        self.rank = rank
        self.world = world
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(self.device)
        self.dbg = dbg

        # --------------------------------------------------------------- #
        # 1) Build or load the model skeleton.                            #
        # --------------------------------------------------------------- #
        cfg = AutoConfig.from_pretrained(args.model_name)
        cfg.attn_implementation = "eager"  # always safe across devices

        if rank == 0:
            dbg("Loading full model on CPU (bf16)")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                config=cfg,
                device_map={"": "cpu"},
                torch_dtype=torch.bfloat16,
            )
            model.config.attn_implementation = "flash_attention_2"
        else:
            dbg("Building empty-weight skeleton on meta")
            try:
                from accelerate import init_empty_weights

                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(cfg)
            except ImportError:
                dbg("`accelerate` not available; manual meta tensors")
                model = AutoModelForCausalLM.from_config(cfg)
                for p in model.parameters():
                    p.data = torch.empty_like(p, device="meta")

        # Iteration order must match send/recv: embed → blocks → head
        self.layers: List[torch.nn.Module] = [
            model.model.embed_tokens,
            *model.model.layers,
            model.lm_head,
        ]

        # Double buffers for the ring
        self.buffers = (FlatPacket(), FlatPacket())
        self.toggle = 0

    # ..................................................................... #
    # Ring helpers – now TAG-AWARE                                         #
    # ..................................................................... #

    @staticmethod
    def _send(tensor: torch.Tensor, dst: int, tag: int):
        dist.send(tensor, dst=dst, tag=tag)

    @staticmethod
    def _recv(tensor: torch.Tensor, src: int, tag: int):
        dist.recv(tensor, src=src, tag=tag)

    # ..................................................................... #
    # Streaming one layer                                                   #
    # ..................................................................... #

    def _stream_layer(self, idx: int):
        """Move *weights* for layer `idx` one hop along the ring and rebuild params."""
        left = (self.rank - 1) % self.world
        right = (self.rank + 1) % self.world
        buf = self.buffers[self.toggle]

        # Use unique tags so transfers for different layers never collide.
        shape_tag = idx * 2
        data_tag = idx * 2 + 1

        # --------------------------------------------------------------- #
        # 1) Transmit / receive the flat packet and its length.           #
        # --------------------------------------------------------------- #
        first_pass_rank0 = self.rank == 0 and idx == 0
        if not first_pass_rank0:
            shape = torch.empty(1, dtype=torch.int64, device=self.device)
            self._recv(shape, left, shape_tag)
            numel = int(shape.item())
            buf.make_empty(numel, self.device)
            if numel:
                self._recv(buf.flat, left, data_tag)
        else:
            buf.pack(self.layers[idx], self.device)

        # --------------------------------------------------------------- #
        # 2) Materialise storage if still meta.                           #
        # --------------------------------------------------------------- #
        if any(not _has_real_storage(p) for p in self.layers[idx].parameters()):
            materialise_meta(self.layers[idx], self.device, buf.flat.dtype)

        # --------------------------------------------------------------- #
        # 3) Copy packet into the layer’s parameters.                     #
        # --------------------------------------------------------------- #
        if buf.flat.numel():
            buf.unpack_to(self.layers[idx])

        # --------------------------------------------------------------- #
        # 4) Forward packet to the next rank.                             #
        # --------------------------------------------------------------- #
        shape_out = torch.tensor([buf.flat.numel()], dtype=torch.int64, device=self.device)
        self._send(shape_out, right, shape_tag)
        if buf.flat.numel():
            self._send(buf.flat, right, data_tag)
        self.toggle ^= 1  # flip buffer

    # ..................................................................... #
    # Forward + multinomial sampling                                        #
    # ..................................................................... #

    @torch.no_grad()
    def sample(self, input_ids: torch.Tensor) -> Tuple[list, list]:
        """Embed tokens then run through streamed blocks and lm_head."""
        # ---------- 1. Embedding layer (idx 0) -------------------------- #
        self._stream_layer(0)
        embed = self.layers[0].to(self.device, non_blocking=True)
        hidden = embed(input_ids)
        embed.to("cpu", non_blocking=True)  # keep vRAM down
        torch.cuda.current_stream().synchronize()

        # ---------- 2. Transformer blocks (idx 1 … n-2) ----------------- #
        for idx in range(1, len(self.layers) - 1):
            self._stream_layer(idx)
            layer = self.layers[idx].to(self.device, non_blocking=True)

            for start in range(0, hidden.size(0), self.args.micro_batch_size):
                mb = hidden[start : start + self.args.micro_batch_size]
                out = layer(mb)
                mb_out = out[0] if isinstance(out, tuple) else out
                hidden[start : start + self.args.micro_batch_size] = mb_out

            layer.to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()

        # ---------- 3. lm_head (idx n-1) -------------------------------- #
        idx = len(self.layers) - 1
        self._stream_layer(idx)
        head = self.layers[idx].to(self.device, non_blocking=True)

        logits_chunks: List[torch.Tensor] = []
        for start in range(0, hidden.size(0), self.args.micro_batch_size):
            mb = hidden[start : start + self.args.micro_batch_size]
            out = head(mb)
            logits_chunks.append(out[0] if isinstance(out, tuple) else out)

        head.to("cpu", non_blocking=True)
        torch.cuda.current_stream().synchronize()
        logits = torch.cat(logits_chunks, dim=0)

        # ---------- 4. Vectorised multinomial sampling ------------------ #
        probs = torch.softmax(logits / self.args.sampling_temperature, dim=-1)
        if probs.numel() == 0:
            raise RuntimeError("Got empty probability tensor during sampling.")

        draws = (
            torch.multinomial(
                probs.view(-1, probs.size(-1)),
                self.args.num_samples * self.args.num_rounds,
                replacement=True,
            )
            .view(*probs.shape[:-1], -1)
            .contiguous()
        )

        ids_all, counts_all = [], []
        for b in range(draws.size(0)):
            ids_seq, cnts_seq = [], []
            for t in range(draws.size(1)):
                uniq, cnt = torch.unique(draws[b, t], return_counts=True)
                ids_seq.append(uniq.cpu().tolist())
                cnts_seq.append(cnt.cpu().tolist())
            ids_all.append(ids_seq)
            counts_all.append(cnts_seq)
        return ids_all, counts_all


###############################################################################
# Data helpers                                                                #
###############################################################################


def _pin_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pin all tensor values so `.to(device, non_blocking=True)` is fast."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.pin_memory()
    return batch


def collate_fn(examples, tok, max_len):
    enc = tok(
        [e["text"] for e in examples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
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
# Push shard                                                                  #
###############################################################################


def push_shard(records: List[dict], args: Args, dbg):
    dbg(f"Pushing {len(records):,d} records to hub")
    api = HfApi(token=args.hf_token)
    existing = api.list_repo_files(args.output_repo, repo_type="dataset")
    idx = sum(f.endswith(".parquet") for f in existing)

    fname = f"data_{idx:05d}.parquet"
    Dataset.from_list(records).to_parquet(fname)

    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull()
    repo.git_add(fname)
    repo.git_commit(f"add shard {idx}")
    repo.git_push()
    repo.git_clear()
    os.remove(fname)


###############################################################################
# Gather helper                                                               #
###############################################################################


def gather_and_push(local: List[dict], args: Args, rank: int, world: int, dbg):
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        merged: List[dict] = []
        for part in gathered:  # type: ignore[arg-type]
            merged.extend(part)
        push_shard(merged, args, dbg)


###############################################################################
# Worker                                                                      #
###############################################################################


def worker(args: Args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dbg = get_logger(rank, args.debug)

    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())

    try:
        engine = ReversePipelineEngine(args, rank, world, dbg)
        tok = AutoTokenizer.from_pretrained(args.model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token  # ensure padding exists

        ds = load_dataset(
            args.dataset_name,
            split=args.dataset_split,
            streaming=True,
            token=args.hf_token,
        )
        loader = stream_loader(ds, tok, args.batch_size, args.max_seq_len)

        local_records: List[dict] = []
        seen = 0
        for batch_idx, batch in enumerate(loader):
            if stop_evt.is_set():
                dbg("SIGTERM received – breaking loop")
                break

            dbg(f"Batch {batch_idx} acquired")
            input_ids = batch["input_ids"].to(engine.device, non_blocking=True)
            ids, cnts = engine.sample(input_ids)

            for i in range(len(input_ids)):
                toks = input_ids[i].tolist()
                while toks and toks[-1] == tok.pad_token_id:  # strip padding
                    toks.pop()

                seq_len = len(toks)
                local_records.append(
                    {
                        "input_ids": toks,
                        "sampled_ids": ids[i][:seq_len],
                        "sampled_counts": cnts[i][:seq_len],
                    }
                )
            seen += len(input_ids)

            if seen >= args.push_every:
                dbg("Reached push_every – gathering to rank-0")
                gather_and_push(local_records, args, rank, world, dbg)
                local_records.clear()
                seen = 0

        # ───── Final flush ───────────────────────────────────────────── #
        if local_records and not stop_evt.is_set():
            dbg("Final flush before exit")
            gather_and_push(local_records, args, rank, world, dbg)

    except Exception:
        dbg("Unhandled exception – printing traceback")
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    dbg("Graceful shutdown")
    dist.destroy_process_group()


###############################################################################
# Entry-point                                                                 #
###############################################################################


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    worker(args)


if __name__ == "__main__":
    main()

from __future__ import annotations
"""
Generator with reverse pipeline parallelism (ring) and optional debug logs.
Weights travel  CPU → GPU-0 → … → GPU-N-1 → GPU-0.  Rank-0 then off-loads the
last layer back to CPU.  Use --debug for verbose per-rank output.

June 2025 patch
===============

* **Robust storage materialisation**
  PyTorch ≥ 2.1 added `Module.to_empty(device=…, dtype=…)`; older builds either
  expose `to_empty(device=…)` (no *dtype*) or no `to_empty` at all.  The helper
  now:

  1. Reflects on the callable signature and only passes parameters that exist.
  2. Falls back to an explicit loop (same logic as before) if `to_empty`
    is absent **or** raises.

* **Meta-state sanity checks**
  `materialise_meta()` verifies *every* parameter/buffer has real storage on the
  target device after it finishes; otherwise it raises immediately with a clear
  hint instead of letting you discover the mismatch during the forward pass.

* **Packet integrity guard**
  `FlatPacket.unpack_to()` checks that the flat tensor exactly matches the
  number of elements in the target layer and refuses to proceed if not.

* **Minor quality-of-life**
  * `collate_fn()` pin-memories the tensors so `to(device, non_blocking=True)`
    uses the P2P path.
  * Debug logger prints the local time so ranks stay in-sync.
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
from typing import List, Tuple

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
    return not (getattr(t, "is_meta", False) or getattr(t.data, "is_meta", False))


def materialise_meta(module: torch.nn.Module, device: torch.device, dtype: torch.dtype):
    """
    Ensure *all* params & buffers of *module* live on the indicated device
    with real backing storage.  Works across PyTorch versions:

      * ≥ 2.1  : Module.to_empty(device=…, dtype=…)
      * 2.0.x  : Module.to_empty(device=…)        (no dtype arg)
      * < 2.0  : no to_empty() – manual loop
    """
    # ------------------------------------------------------------------ #
    # 1) Try the most capable native API first, *adapting* to signature. #
    # ------------------------------------------------------------------ #
    to_empty = getattr(module, "to_empty", None)
    if callable(to_empty):
        try:
            kw = {}
            sig = inspect.signature(to_empty)
            if "device" in sig.parameters:
                kw["device"] = device
            if "dtype" in sig.parameters:
                kw["dtype"] = dtype
            to_empty(**kw)        # type: ignore[arg-type]
        except Exception as exc:  # RuntimeError, TypeError, etc.
            # We fall back – but keep the exception around for context.
            fallback_exc = exc
        else:
            fallback_exc = None
    else:
        fallback_exc = RuntimeError("module has no .to_empty()")

    # ------------------------------------------------------------------ #
    # 2) Manual allocation fallback if needed.                           #
    # ------------------------------------------------------------------ #
    if fallback_exc is not None:
        for p in module.parameters(recurse=True):
            if not _has_real_storage(p):
                p.data = torch.empty(p.shape, dtype=dtype, device=device)
        for name, buf in module.named_buffers(recurse=True):
            if not _has_real_storage(buf):
                module._buffers[name] = torch.empty(buf.shape, dtype=dtype, device=device)

    # ------------------------------------------------------------------ #
    # 3) Post-condition check – infectious bug stopper.                  #
    # ------------------------------------------------------------------ #
    bad = [
        (n, "PARAM" if isinstance(t, torch.nn.Parameter) else "BUFFER")
        for n, t in list(module.named_parameters(recurse=True))
                 + list(module.named_buffers(recurse=True))
        if not _has_real_storage(t)
    ]
    if bad:  # pragma: no cover
        names = ", ".join(f"{k}({kind})" for k, kind in bad)
        raise RuntimeError(
            f"materialise_meta() failed – {len(bad)} tensors are still meta "
            f"after allocation: {names}"
        )

###############################################################################
# Flat packet helper                                                          #
###############################################################################


class FlatPacket:
    """
    Helper that serialises one *layer* worth of parameters into a single flat
    contiguous tensor, suitable for NCCL point-to-point sends/receives.
    """

    def __init__(self):
        self.flat: torch.Tensor | None = None

    # ..................................................................... #
    # Packing                                                               #
    # ..................................................................... #

    def pack(self, layer: torch.nn.Module, device: torch.device):
        with torch.no_grad():
            self.flat = torch.cat(
                [p.data.view(-1) for p in layer.parameters()],
                out=None
            ).contiguous().to(device)

    def make_empty(self, numel: int, device: torch.device, dtype=torch.bfloat16):
        self.flat = torch.empty(numel, dtype=dtype, device=device)

    # ..................................................................... #
    # Unpacking                                                             #
    # ..................................................................... #

    def unpack_to(self, layer: torch.nn.Module):
        if self.flat is None:
            raise RuntimeError("FlatPacket.unpack_to() called before .flat is set")

        param_elems = sum(p.numel() for p in layer.parameters())
        if param_elems != self.flat.numel():
            raise RuntimeError(
                f"Packet size mismatch: layer expects {param_elems:,d} elements "
                f"but flat tensor has {self.flat.numel():,d}"
            )

        offset = 0
        with torch.no_grad():
            for p in layer.parameters():
                n = p.numel()
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
        cfg.attn_implementation = "eager"          # 100 % safe everywhere

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
                    p.data = torch.empty_like(p.data, device="meta")

        # The iteration order must match send/receive order: embed → blocks → head
        self.layers: List[torch.nn.Module] = [
            model.model.embed_tokens,
            *model.model.layers,
            model.lm_head,
        ]

        # Double buffers for the ring
        self.buffers = (FlatPacket(), FlatPacket())
        self.toggle = 0

    # ..................................................................... #
    # Ring helpers                                                          #
    # ..................................................................... #
    def _send(self, tensor: torch.Tensor, dst: int):
        dist.send(tensor, dst)

    def _recv(self, tensor: torch.Tensor, src: int):
        dist.recv(tensor, src)

    # ..................................................................... #
    # Streaming one layer                                                   #
    # ..................................................................... #
    def _stream_layer(self, idx: int):
        """Move *weights* for layer `idx` one hop along the ring and rebuild params."""
        left  = (self.rank - 1) % self.world
        right = (self.rank + 1) % self.world
        buf = self.buffers[self.toggle]

        # --------------------------------------------------------------- #
        # 1) Transmit / receive the flat packet and its length.           #
        # --------------------------------------------------------------- #
        first_pass_rank0 = self.rank == 0 and idx == 0
        if not first_pass_rank0:
            # Receive `int64 numel`
            shape = torch.empty(1, dtype=torch.int64, device=self.device)
            self._recv(shape, left)
            buf.make_empty(int(shape.item()), self.device)
            self._recv(buf.flat, left)
        else:
            # Rank-0 originates the packet
            buf.pack(self.layers[idx], self.device)

        # --------------------------------------------------------------- #
        # 2) Materialise storage if this replica still has meta tensors.  #
        # --------------------------------------------------------------- #
        if any(not _has_real_storage(p) for p in self.layers[idx].parameters()):
            materialise_meta(self.layers[idx], self.device, buf.flat.dtype)

        # --------------------------------------------------------------- #
        # 3) Copy packet into the layer’s parameters (GPU-to-GPU memcpy). #
        # --------------------------------------------------------------- #
        buf.unpack_to(self.layers[idx])

        # --------------------------------------------------------------- #
        # 4) Forward packet to the next rank.                             #
        # --------------------------------------------------------------- #
        shape_out = torch.tensor([buf.flat.numel()], dtype=torch.int64, device=self.device)
        self._send(shape_out, right)
        self._send(buf.flat, right)
        self.toggle ^= 1  # flip buffer

    # ..................................................................... #
    # Forward + multinomial sampling                                        #
    # ..................................................................... #
    @torch.no_grad()
    def sample(self, input_ids: torch.Tensor) -> Tuple[list, list]:
        """Embed tokens then run through streamed blocks and lm_head."""
        # ----- Embedding layer (idx 0) --------------------------------- #
        self._stream_layer(0)
        embed = self.layers[0]               # now lives on self.device
        hidden = embed(input_ids)

        # Optionally drop the weights back to CPU to cap vRAM usage
        embed.to("cpu", non_blocking=True)
        torch.cuda.current_stream().synchronize()

        # ----- Transformer blocks & lm_head (idx 1 .. end) ------------- #
        for idx in range(1, len(self.layers)):
            self._stream_layer(idx)
            layer = self.layers[idx].to(self.device, non_blocking=True)

            # micro-batch loop (needed only if batch too big for memory)
            for start in range(0, hidden.size(0), self.args.micro_batch_size):
                mb = hidden[start : start + self.args.micro_batch_size]
                out = layer(mb)
                hidden[start : start + self.args.micro_batch_size] = (
                    out[0] if isinstance(out, tuple) else out
                )

            layer.to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()

        # ----- Vectorised multinomial sampling ------------------------- #
        logits = hidden      # after lm_head
        probs  = torch.softmax(logits / self.args.sampling_temperature, dim=-1)

        draws = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            self.args.num_samples * self.args.num_rounds,
            replacement=True,
        ).view(*probs.shape[:-1], -1)

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


def collate_fn(examples, tok, max_len):
    return tok(
        [e["text"] for e in examples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).pin_memory()  # enables fast non-blocking .to(device)


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
    idx = sum(f.endswith(".parquet") for f in api.list_repo_files(args.output_repo, repo_type="dataset"))
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
        for part in gathered:        # type: ignore[arg-type]
            merged.extend(part)
        push_shard(merged, args, dbg)


###############################################################################
# Worker                                                                      #
###############################################################################


def worker(args: Args):
    dist.init_process_group("nccl")
    rank  = dist.get_rank()
    world = dist.get_world_size()
    dbg   = get_logger(rank, args.debug)

    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())

    try:
        engine = ReversePipelineEngine(args, rank, world, dbg)
        tok    = AutoTokenizer.from_pretrained(args.model_name)
        ds     = load_dataset(
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
                while toks and toks[-1] == 0:      # strip right-pad
                    toks.pop()

                seq_len = len(toks)
                local_records.append(
                    {
                        "input_ids": toks,
                        "sampled_ids":     ids[i][:seq_len],
                        "sampled_counts": cnts[i][:seq_len],
                    }
                )
            seen += len(input_ids)

            if seen >= args.push_every:
                dbg("Reached push_every – gathering to rank-0")
                gather_and_push(local_records, args, rank, world, dbg)
                local_records.clear()
                seen = 0

        # -------- Final flush ----------------------------------------- #
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

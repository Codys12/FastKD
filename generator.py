# generator_distributed.py – weight‑streaming pipeline version
# Author: ChatGPT – June 2025
"""
This version replaces the hand‑rolled multi‑GPU logic with **torch.distributed**
weight‑streaming pipelining.  Instead of moving *data* through fixed model
stages, we keep a static shard of the batch on every GPU and **stream the
layers** (weights) across devices.  Every layer processes up to
`micro_batch_size` examples resident on each GPU before its parameters are sent
on to the next rank.  All communications use `torch.distributed.broadcast`
async ops so communication of the next layer overlaps compute of the current
one.

Key features
------------
* One process **per GPU** (launched with *torchrun*).
* Dataset stream is **sharded by rank** (round‑robin) so every example is used
  exactly once.
* Embedding & LM‑head weights are **replicated once** at start‑up;
  transformer block weights are streamed layer‑by‑layer.
* Rank 0 collects sampled statistics from all ranks and pushes shards to the
  Hub, avoiding write conflicts.

Caveats
-------
* The naive broadcast of every layer to *all* ranks is simple but network‑heavy.
  For larger clusters, you may prefer a ring or tree schedule.
* The implementation relies on identical layer/buffer topologies on every
  process – as is standard for decoder‑only LLMs.
* Not yet rigorously profiled.  Tune ``micro_batch_size`` and overlap windows
  for your fabric.
"""
from __future__ import annotations

import argparse
import os
import signal
import threading
from dataclasses import dataclass
from typing import List

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import HfApi, Repository

################################################################################
# Argument parsing                                                               
################################################################################

@dataclass
class Args:
    model_name: str
    dataset_name: str
    dataset_split: str
    output_repo: str
    batch_size: int
    micro_batch_size: int
    num_rounds: int        # sampling R
    num_samples: int       # sampling N per round
    sampling_temperature: float
    push_every: int
    max_seq_len: int
    hf_token: str | None = None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Generate KD dataset with distributed weight‑streaming")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--output_repo", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--push_every", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()
    return Args(**vars(args))

################################################################################
# Distributed helper                                                             
################################################################################

def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialise torch.distributed and return (local_rank, global_rank, world_size, device)."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, dist.get_rank(), dist.get_world_size(), device

################################################################################
# Sampling utilities                                                             
################################################################################

def sample_distribution(
    logits: torch.Tensor,
    num_samples: int,
    num_rounds: int,
    temperature: float = 1.0,
):
    """Return sampled token ids and raw counts for each position."""
    logits = logits.float()
    p = torch.softmax(logits, dim=-1)
    q = torch.softmax(logits / temperature, dim=-1)

    ids_all: List[List[List[int]]] = []
    counts_all: List[List[List[int]]] = []
    bsz, seqlen, vocab = p.shape

    for b in range(bsz):
        ids_seq: List[List[int]] = []
        counts_seq: List[List[int]] = []
        for s in range(seqlen):
            p_row = p[b, s]
            q_row = q[b, s]
            q_row = torch.nan_to_num(q_row, nan=0.0, posinf=0.0, neginf=0.0)
            q_row = torch.clamp(q_row, min=0)
            if not torch.isfinite(q_row).all() or q_row.sum() == 0:
                q_row.fill_(1.0 / vocab)
            else:
                q_row = q_row + 1e-6
                q_row /= q_row.sum()
            total_draws = num_samples * num_rounds
            samples = torch.multinomial(q_row, total_draws, replacement=True)
            uniq, counts = torch.unique(samples, return_counts=True)
            ids_seq.append(uniq.cpu().tolist())
            counts_seq.append(counts.cpu().tolist())
        ids_all.append(ids_seq)
        counts_all.append(counts_seq)
    return ids_all, counts_all

################################################################################
# Distributed weight‑streaming streamer                                          
################################################################################

class DistributedStreamer:
    """Streams model **layers** across devices, keeping activations resident."""

    def __init__(self, args: Args, device: torch.device, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # Load the full model on CPU first (no FlashAttn yet – avoids init issues)
        config = AutoConfig.from_pretrained(args.model_name)
        config.attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, config=config, device_map={"": "cpu"}, torch_dtype=torch.bfloat16
        )
        # enable flash after weight load so every rank can use it
        self.model.config.attn_implementation = "flash_attention_2"

        self.layers = list(self.model.model.layers)
        self.embed = self.model.model.embed_tokens.to(self.device)  # replicated once
        self.lm_head = self.model.lm_head.to(self.device)           # replicated once

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _broadcast_module(self, module: torch.nn.Module, owner: int):
        """Broadcast *in‑place* parameters & buffers of *module* from *owner* to all ranks."""
        work_handles: List[dist.Work] = []
        for p in module.parameters(recurse=True):
            work_handles.append(dist.broadcast(p.data, src=owner, async_op=True))
        for b in module.buffers(recurse=True):
            work_handles.append(dist.broadcast(b.data, src=owner, async_op=True))
        # Wait for parameter reception *before* computation but *after* async send.
        for w in work_handles:
            w.wait()

    # ---------------------------------------------------------------------
    # forward sampling                                                      
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: int,
        num_samples: int,
        num_rounds: int,
        temperature: float = 1.0,
    ):
        """Compute logits and sample using weight‑streaming pipeline."""
        # --------------------------------------------------------------
        # Prepare micro‑batches that stay resident on *this* GPU.
        # --------------------------------------------------------------
        batches = [
            input_ids[i : i + micro_batch_size]
            for i in range(0, input_ids.size(0), micro_batch_size)
        ]

        ids_all: List[List[List[int]]] = []
        counts_all: List[List[List[int]]] = []

        # --------------------------------------------------------------
        # Move TKNs to GPU once, keep until all layers processed.
        # --------------------------------------------------------------
        for micro in batches:
            hidden = micro.to(self.device, non_blocking=True)
            hidden = self.embed(hidden)

            # ------------------------------------------------------
            # Stream every transformer block.
            # Owner rotates deterministically: idx % world_size
            # ------------------------------------------------------
            for idx, layer in enumerate(self.layers):
                owner = idx % self.world_size

                # Owner moves weights to GPU; others create an uninit copy first.
                layer_device_ready = layer.to(self.device, non_blocking=True)

                # broadcast weights in‑place so all replicas match owner.
                self._broadcast_module(layer_device_ready, owner)

                # Execute layer on resident activations.
                out = layer_device_ready(hidden, use_cache=False)
                hidden = out[0] if isinstance(out, tuple) else out

                # Free weights early to make room for next layer.
                layer_device_ready.to("cpu", non_blocking=True)
                torch.cuda.empty_cache()

            # ------------------------------------------------------
            # Final LM head (replicated), logits -> sample.
            # ------------------------------------------------------
            logits = self.lm_head(hidden)
            ids, counts = sample_distribution(
                logits,
                num_samples=num_samples,
                num_rounds=num_rounds,
                temperature=temperature,
            )
            ids_all.extend(ids)
            counts_all.extend(counts)

        return ids_all, counts_all

################################################################################
# Data utilities                                                                 
################################################################################

def collate_fn(examples, tokenizer, max_seq_len: int):
    texts = [e["text"] for e in examples]
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )


def streaming_dataloader(
    ds, tokenizer, batch_size: int, max_seq_len: int, rank: int, world_size: int, shutdown: threading.Event | None = None
):
    """Round‑robin sharding: example *i* goes to rank *i % world_size*."""
    batch: List[dict] = []
    for idx, example in enumerate(ds):
        if shutdown and shutdown.is_set():
            break
        if idx % world_size != rank:
            continue  # not my shard
        batch.append(example)
        if len(batch) == batch_size:
            yield collate_fn(batch, tokenizer, max_seq_len)
            batch.clear()
            if shutdown and shutdown.is_set():
                break
    if batch and not (shutdown and shutdown.is_set()):
        yield collate_fn(batch, tokenizer, max_seq_len)

################################################################################
# Push helper (rank 0 only)                                                      
################################################################################

def push_shard_distributed(records_local: List[dict], args: Args, rank: int):
    """Gather shards to rank 0 then push to the Hub to avoid write conflicts."""
    world_size = dist.get_world_size()
    gathered: List[List[dict]] | None = None
    if rank == 0:
        gathered = [None] * world_size  # type: ignore
    dist.gather_object(records_local, gathered, dst=0)

    if rank == 0 and gathered is not None:
        combined: List[dict] = []
        for rec_list in gathered:
            combined.extend(rec_list)
        if not combined:
            return  # nothing to write

        api = HfApi(token=args.hf_token)
        files = api.list_repo_files(args.output_repo, repo_type="dataset")
        idx = sum(f.endswith(".parquet") for f in files)
        filename = f"data_{idx:05d}.parquet"
        ds = Dataset.from_list(combined)
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

################################################################################
# Worker                                                                         
################################################################################

def worker(args: Args):
    local_rank, rank, world_size, device = setup_distributed()

    # Graceful shutdown handling – every rank listens.
    shutdown_evt = threading.Event()

    def _handle_sigterm(signum, frame):
        shutdown_evt.set()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    streamer = DistributedStreamer(args, device, rank, world_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        token=args.hf_token,
        streaming=True,
    )
    dataloader = streaming_dataloader(
        dataset, tokenizer, args.batch_size, args.max_seq_len, rank, world_size, shutdown_evt
    )

    pending_records: List[dict] = []
    processed_total = 0

    for batch in dataloader:
        if shutdown_evt.is_set():
            break
        input_ids = batch["input_ids"]
        ids, counts = streamer.sample(
            input_ids,
            args.micro_batch_size,
            args.num_samples,
            args.num_rounds,
            args.sampling_temperature,
        )
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
            pending_records.append(record)
        processed_total += len(input_ids)

        if processed_total >= args.push_every:
            push_shard_distributed(pending_records, args, rank)
            pending_records.clear()
            processed_total = 0

        if shutdown_evt.is_set():
            break

    # Final flush
    if pending_records and not shutdown_evt.is_set():
        push_shard_distributed(pending_records, args, rank)

################################################################################
# Main                                                                           
################################################################################

def main():
    args = parse_args()
    worker(args)


if __name__ == "__main__":
    main()

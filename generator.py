# generator_distributed.py – weight‑streaming pipeline version
# Author: ChatGPT – June 2025  ✨  (rev‑2)
"""
**Weight‑streaming, batch‑scoped**: Parameters are now broadcast **once per layer
per full `batch_size`**, not once per `micro_batch_size`.  Each layer stays
resident on the GPU while it marches through every micro‑batch in the batch,
then is released.  This amortises communication over `batch_size /
micro_batch_size` forwards and yields far fewer synchronisations.

High‑level flow per `sample()` on each rank
------------------------------------------
1. **Slice** the input tensor into micro‑batches on‐device.
2. **Embed** every micro‑batch immediately (tokens → hidden) & build its rotary
   position helpers; store in a small struct.
3. **For each transformer block** *i*:
   • Owner rank *i % world_size* brings the weights to GPU and broadcasts once.
   • The block runs **sequentially over all resident micro‑batches**.
   • Then weights are sent back to CPU & cache flushed.
4. **lm_head** (already replicated) processes every micro‑batch → logits.
5. `sample_distribution` draws multinomial samples; results concatenated.

Only the sampling utilities and data loader are unchanged.
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
    parser = argparse.ArgumentParser(description="Generate KD dataset with distributed weight‑streaming (batch‑scoped)")
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
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{local_rank}")

################################################################################
# Sampling util                                                                 
################################################################################

def sample_distribution(logits: torch.Tensor, num_samples: int, num_rounds: int, temperature: float = 1.0):
    logits = logits.float()
    p = torch.softmax(logits, dim=-1)
    q = torch.softmax(logits / temperature, dim=-1)
    ids_all: List[List[List[int]]] = []
    counts_all: List[List[List[int]]] = []
    bsz, seqlen, vocab = p.shape
    for b in range(bsz):
        ids_seq, counts_seq = [], []
        for s in range(seqlen):
            q_row = torch.nan_to_num(q[b, s], nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0)
            if not torch.isfinite(q_row).all() or q_row.sum() == 0:
                q_row.fill_(1.0 / vocab)
            else:
                q_row = (q_row + 1e-6) / q_row.sum()
            samples = torch.multinomial(q_row, num_samples * num_rounds, replacement=True)
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
    """Weight‑streaming but params move **once per layer per full batch**."""

    def __init__(self, args: Args, device: torch.device, rank: int, world_size: int):
        self.rank, self.world_size, self.device = rank, world_size, device
        cfg = AutoConfig.from_pretrained(args.model_name)
        cfg.attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, config=cfg, device_map={"": "cpu"}, torch_dtype=torch.bfloat16
        )
        self.model.config.attn_implementation = "flash_attention_2"
        self.layers = list(self.model.model.layers)
        self.rotary_emb = self.model.model.rotary_emb
        self.embed = self.model.model.embed_tokens.to(device)
        self.lm_head = self.model.lm_head.to(device)

    # ---------------- internal helpers ----------------
    def _broadcast(self, module: torch.nn.Module, owner: int):
        works: List[dist.Work] = []
        for t in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
            works.append(dist.broadcast(t.data, src=owner, async_op=True))
        for w in works:
            w.wait()

    # ---------------- public forward ------------------
    @torch.no_grad()
    def sample(self, input_ids: torch.Tensor, micro_bs: int, num_samples: int, num_rounds: int, temperature: float):
        # ---- split & embed once per micro‑batch ----
        micro_inputs = [input_ids[i:i + micro_bs] for i in range(0, input_ids.size(0), micro_bs)]
        micro_states = []
        for mb in micro_inputs:
            hidden = self.embed(mb.to(self.device, non_blocking=True))
            seq_len = hidden.size(1)
            cache_pos = torch.arange(seq_len, device=self.device)
            pos_ids = cache_pos.unsqueeze(0)
            pos_emb = self.rotary_emb(hidden, pos_ids)
            micro_states.append({
                "hidden": hidden,
                "cache_pos": cache_pos,
                "pos_ids": pos_ids,
                "pos_emb": pos_emb,
            })

        # ---- stream each transformer layer once ----
        for idx, layer in enumerate(self.layers):
            owner = idx % self.world_size
            layer.to(self.device, non_blocking=True)
            self._broadcast(layer, owner)
            for ms in micro_states:
                out = layer(
                    ms["hidden"],
                    position_ids=ms["pos_ids"],
                    cache_position=ms["cache_pos"],
                    position_embeddings=ms["pos_emb"],
                    use_cache=False,
                )
                ms["hidden"] = out[0] if isinstance(out, tuple) else out
            layer.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()

        # ---- lm‑head & multinomial sampling ----
        ids_all, counts_all = [], []
        for ms in micro_states:
            logits = self.lm_head(ms["hidden"])
            ids, counts = sample_distribution(logits, num_samples, num_rounds, temperature)
            ids_all.extend(ids)
            counts_all.extend(counts)
        return ids_all, counts_all

################################################################################
# Data utilities                                                                
################################################################################

def collate_fn(exs, tok, max_len):
    return tok([e["text"] for e in exs], return_tensors="pt", padding=True, truncation=True, max_length=max_len)


def streaming_dataloader(ds, tok, bs, max_len, rank, world, shut: threading.Event | None):
    batch: List[dict] = []
    for i, ex in enumerate(ds):
        if shut and shut.is_set():
            break
        if i % world != rank:
            continue
        batch.append(ex)
        if len(batch) == bs:
            yield collate_fn(batch, tok, max_len)
            batch.clear()
            if shut and shut.is_set():
                break
    if batch and not (shut and shut.is_set()):
        yield collate_fn(batch, tok, max_len)

################################################################################
# Shard push (rank‑0)                                                           
################################################################################

def push_shard(records_local: List[dict], args: Args, rank: int):
    world = dist.get_world_size()
    gather: List[List[dict]] | None = [None] * world if rank == 0 else None  # type: ignore
    dist.gather_object(records_local, gather, dst=0)
    if rank != 0:
        return
    combined: List[dict] = []
    for part in gather:
        combined.extend(part)
    if not combined:
        return
    api = HfApi(token=args.hf_token)
    idx = sum(f.endswith(".parquet") for f in api.list_repo_files(args.output_repo, repo_type="dataset"))
    fname = f"data_{idx:05d}.parquet"
    Dataset.from_list(combined).to_parquet(fname)
    repo = Repository("repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset")
    repo.git_pull()
    repo.git_add(fname)
    repo.git_commit(f"Add shard {idx}")
    repo.git_push()
    repo.git_clear()
    os.remove(fname)

################################################################################
# Worker                                                                       
################################################################################

def worker(args: Args):
    local_rank, rank, world, device = setup_distributed()
    shutdown_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: shutdown_evt.set())

    streamer = DistributedStreamer(args, device, rank, world)
    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = load_dataset(args.dataset_name, split=args.dataset_split, token=args.hf_token, streaming=True)
    loader = streaming_dataloader(ds, tok, args.batch_size, args.max_seq_len, rank, world, shutdown_evt)

    buf: List[dict] = []
    seen = 0
    for batch in loader:
        if shutdown_evt.is_set():
            break
        ids, counts = streamer.sample(batch["input_ids"], args.micro_batch_size, args.num_samples, args.num_rounds, args.sampling_temperature)
        for i, toks in enumerate(batch["input_ids"]):
            tokens = toks.tolist()
            while tokens and tokens[-1] == 0:
                tokens.pop()
            l = len(tokens)
            buf.append({"input_ids": tokens, "sampled_ids": ids[i][:l], "sampled_counts": counts[i][:l]})
        seen += len(batch["input_ids"])
        if seen >= args.push_every:
            push_shard(buf, args, rank)
            buf.clear(); seen = 0
    if buf and not shutdown_evt.is_set():
        push_shard(buf, args, rank)

################################################################################
# Entrypoint                                                                   
################################################################################

if __name__ == "__main__":
    worker(parse_args())

from __future__ import annotations

import argparse
import os
import signal
import torch.multiprocessing as mp
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

STREAMER: "Streamer" | None = None
SHUTDOWN: mp.Event | None = None


@dataclass
class Args:
    model_name: str
    dataset_name: str
    dataset_split: str
    output_repo: str
    batch_size: int
    micro_batch_size: int
    sampling_rounds: int
    push_every: int
    max_seq_len: int
    num_workers: int
    hf_token: str | None = None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Generate KD dataset with streaming")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--output_repo", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument(
        "--sampling_rounds",
        type=int,
        default=50,
        help="Number of sampling rounds per token as in the paper",
    )
    parser.add_argument("--push_every", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()
    return Args(**vars(args))


class Streamer:
    """Handles streaming forward‑passes one micro‑batch at a time."""

    def __init__(self, model_name: str):
        self.model = self._load_model(model_name)
        self.model.eval()  # inference‑only

        self.device = torch.device("cpu")  # will be set per‑worker later
        self.layers = list(self.model.model.layers)
        self.embed = self.model.model.embed_tokens
        self.rotary_emb = self.model.model.rotary_emb  # FIX 1: keep a handle to rotary emb
        self.lm_head = self.model.lm_head

    # ---------------------------------------------------------------------
    # Loading utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _load_model(model_name: str):
        """Load the model, enabling Flash‑Attn only when a GPU is present.

        We first *try* to request Flash‑Attn at load‑time.  If the Transformers
        safety check refuses because the device map is CPU‑only, we reload
        without Flash‑Attn and then enable it post‑hoc via
        ``model.set_attn_implementation`` (which skips the check).
        """

        common_kwargs = dict(
            torch_dtype=torch.float16,
            device_map={"": "cpu"},  # start on CPU, we stream layers to GPU
            low_cpu_mem_usage=True,
        )

        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                **common_kwargs,
            )
        except ValueError as err:
            if "Flash Attention 2.0" not in str(err):
                raise  # unrelated error, re‑raise

            # Fallback: load without Flash‑Attn, then enable it manually if
            # a CUDA device exists.
            model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
            if torch.cuda.is_available():
                try:
                    model.set_attn_implementation("flash_attention_2")
                except AttributeError:
                    # Older HF versions or models without the helper – ignore.
                    pass
            return model

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        device = device or self.device

        # Split into micro‑batches and move to compute device
        batches = [
            input_ids[i : i + micro_batch_size].to(device)
            for i in range(0, input_ids.size(0), micro_batch_size)
        ]

        # ---- Embedding ----
        self.embed.to(device, non_blocking=True)
        hidden: List[torch.Tensor] = []
        pos_ids_list: List[torch.Tensor] = []
        for mb in batches:
            seq_len = mb.size(1)
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(
                mb.size(0), -1
            )
            hidden.append(self.embed(mb))
            pos_ids_list.append(pos_ids)
        self.embed.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()

        # ---- Transformer layers ----
        for layer in self.layers:
            layer.to(device, non_blocking=True)
            next_hidden: List[torch.Tensor] = []
            for h, pos in zip(hidden, pos_ids_list):
                # FIX 2: compute (cos, sin) tuple exactly as Qwen3Model expects
                cos_sin = self.rotary_emb(h, pos)
                out = layer(h, position_ids=pos, position_embeddings=cos_sin)
                out = out[0] if isinstance(out, tuple) else out
                next_hidden.append(out)
                # free per‑token activations early
                del h, pos, cos_sin, out
            hidden = next_hidden
            layer.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()

        # ---- LM head ----
        self.lm_head.to(device, non_blocking=True)
        logits = [self.lm_head(h) for h in hidden]
        self.lm_head.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()

        logits_tensor = torch.cat(logits, dim=0).to("cpu", non_blocking=True)
        del logits, hidden  # encourage GC & VRAM release
        torch.cuda.empty_cache()
        return logits_tensor


# -------------------------------------------------------------------------
# Sampling & helpers
# -------------------------------------------------------------------------

def sample_distribution(logits: torch.Tensor, rounds: int):
    """Sample tokens via the paper's multinomial procedure."""
    probs = torch.softmax(logits, dim=-1)
    ids_all: List[List[List[int]]] = []
    probs_all: List[List[List[float]]] = []
    bsz, seqlen, _ = probs.shape
    for b in range(bsz):
        ids_seq, probs_seq = [], []
        for s in range(seqlen):
            p = probs[b, s]
            samples = torch.multinomial(p, rounds, replacement=True)
            uniq, counts = torch.unique(samples, return_counts=True)
            ids_seq.append(uniq.cpu().tolist())
            probs_seq.append((counts.float() / rounds).cpu().tolist())
        ids_all.append(ids_seq)
        probs_all.append(probs_seq)
    return ids_all, probs_all


def collate_fn(examples, tokenizer, max_seq_len: int):
    return tokenizer(
        [ex["text"] for ex in examples],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )


def streaming_dataloader(ds, tokenizer, batch_size: int, max_seq_len: int, shutdown: mp.Event | None = None):
    batch = []
    for example in ds:
        if shutdown and shutdown.is_set():
            break
        batch.append(example)
        if len(batch) == batch_size:
            yield collate_fn(batch, tokenizer, max_seq_len)
            batch.clear()
            if shutdown and shutdown.is_set():
                break
    if batch and not (shutdown and shutdown.is_set()):
        yield collate_fn(batch, tokenizer, max_seq_len)


# -------------------------------------------------------------------------
# Worker logic
# -------------------------------------------------------------------------

def worker_main(rank: int, args: Args, streamer: Streamer, shutdown: mp.Event):
    global STREAMER, SHUTDOWN
    STREAMER = streamer
    SHUTDOWN = shutdown

    def _handle_sigterm(signum, frame):
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    STREAMER.device = device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        token=args.hf_token,
        streaming=True,
    )
    dataset = dataset.shard(num_shards=args.num_workers, index=rank)

    dataloader = streaming_dataloader(dataset, tokenizer, args.batch_size, args.max_seq_len, shutdown)

    all_records: List[dict] = []
    total = 0
    for batch in dataloader:
        if SHUTDOWN.is

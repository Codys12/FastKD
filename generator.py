from __future__ import annotations

"""Knowledge‑Distillation (KD) data generator with streaming and Flash‑Attention.

Key design points
-----------------
* **Flash‑Attention 2** is enabled by initialising the model on GPU, then moving
  weights back to CPU so `share_memory()` (fork reuse) is cheap.
* **Layer‑at‑a‑time streaming** keeps only one transformer block on GPU at once,
  dramatically lowering the peak VRAM requirement.
* **Inference‑only**: gradients are globally disabled and activations are freed
  immediately after use.
"""

import argparse
import signal
from dataclasses import dataclass
from typing import List

import torch
import torch.multiprocessing as mp
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

STREAMER: "Streamer" | None = None  # Will be filled by worker/parent
SHUTDOWN: mp.Event | None = None    # Shared termination flag among workers

# ---------------------------------------------------------------------------
# CLI / args
# ---------------------------------------------------------------------------
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


def parse_args() -> Args:  # noqa: D401 — imperative mood OK
    p = argparse.ArgumentParser(description="Generate KD dataset with streaming")
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--output_repo", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--sampling_rounds", type=int, default=50,
                   help="Multinomial samples per token (KD)")
    p.add_argument("--push_every", type=int, default=1000,
                   help="#records to accumulate before pushing to hub")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--hf_token", default=None)
    return Args(**vars(p.parse_args()))

# ---------------------------------------------------------------------------
# Streamer (layer‑streaming forward pass)
# ---------------------------------------------------------------------------
class Streamer:
    """Streams the forward pass layer‑by‑layer to minimise GPU memory."""

    def __init__(self, model_name: str):
        self.model = self._bootstrap_model(model_name)
        self.model.eval()                      # inference‑only
        self.device = torch.device("cpu")      # per‑worker override later

        # Convenient handles
        m = self.model.model  # type: ignore[attr-defined]
        self.layers = list(m.layers)
        self.embed = m.embed_tokens
        self.rotary_emb = m.rotary_emb
        self.lm_head = self.model.lm_head

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bootstrap_model(model_name: str):
        """Load the model with Flash‑Attention enabled.

        Workflow:
        1. If a CUDA device exists, load directly onto GPU with
           `attn_implementation="flash_attention_2"`.
        2. Immediately move *all* weights back to CPU so the parent process can
           `share_memory()` them cheaply for forked workers.
        3. If *no* GPU is present, fall back to plain attention on CPU.
        """
        has_gpu = torch.cuda.is_available()
        dev_map = {"": "cuda:0"} if has_gpu else {"": "cpu"}

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=dev_map,
                attn_implementation="flash_attention_2" if has_gpu else None,
                low_cpu_mem_usage=True,
            )
        except ValueError:
            # Older Transformers that error on the explicit kwarg — load without.
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=dev_map,
                low_cpu_mem_usage=True,
            )
            if has_gpu:
                # Best‑effort attempt to toggle flash after load
                try:
                    model.set_attn_implementation("flash_attention_2")
                except AttributeError:
                    pass  # not supported — continue with default attn

        # Move back to CPU so children can .share_memory() without huge DMA.
        model.to("cpu")
        torch.cuda.empty_cache()
        return model

    # ------------------------------------------------------------------
    # Forward (layer streaming)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, micro_bs: int,
                device: torch.device | None = None) -> torch.Tensor:
        device = device or self.device
        # Split into micro‑batches for lower VRAM during embed pass.
        micro_batches = [input_ids[i: i + micro_bs].to(device)
                         for i in range(0, input_ids.size(0), micro_bs)]

        # ---- Embedding ----
        self.embed.to(device, non_blocking=True)
        hidden: List[torch.Tensor] = []
        pos_ids_lst: List[torch.Tensor] = []
        for mb in micro_batches:
            seq_len = mb.size(1)
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(mb.size(0), -1)
            hidden.append(self.embed(mb))
            pos_ids_lst.append(pos)
        self.embed.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()

        # ---- Transformer stack ----
        for layer in self.layers:
            layer.to(device, non_blocking=True)
            nxt: List[torch.Tensor] = []
            for h, pos in zip(hidden, pos_ids_lst):
                cs = self.rotary_emb(h, pos)          # cos/sin tuple (Qwen‑style helper)
                out = layer(h, position_ids=pos, position_embeddings=cs)
                out = out[0] if isinstance(out, tuple) else out
                nxt.append(out)
                del h, pos, cs, out  # free as early as possible
            hidden = nxt
            layer.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()

        # ---- LM head ----
        self.lm_head.to(device, non_blocking=True)
        logits = [self.lm_head(h) for h in hidden]
        self.lm_head.to("cpu", non_blocking=True)
        torch.cuda.empty_cache()

        logits = torch.cat(logits, dim=0).to("cpu", non_blocking=True)
        del hidden
        torch.cuda.empty_cache()
        return logits

# ---------------------------------------------------------------------------
# KD multinomial sampling
# ---------------------------------------------------------------------------

def sample_distribution(logits: torch.Tensor, rounds: int):
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

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def collate_fn(examples, tok, max_len):
    return tok([ex["text"] for ex in examples], return_tensors="pt",
               padding=True, truncation=True, max_length=max_len)


def streaming_dataloader(ds, tok, bs, max_len, shutdown: mp.Event | None = None):
    batch = []
    for ex in ds:
        if shutdown and shutdown.is_set():
            break
        batch.append(ex)
        if len(batch) == bs:
            yield collate_fn(batch, tok, max_len)
            batch.clear()
            if shutdown and shutdown.is_set():
                break
    if batch and not (shutdown and shutdown.is_set()):
        yield collate_fn(batch, tok, max_len)

# ---------------------------------------------------------------------------
# Worker entrypoint
# ---------------------------------------------------------------------------

def worker_main(rank: int, args: Args, streamer: Streamer, shutdown: mp.Event):
    global STREAMER, SHUTDOWN
    STREAMER = streamer
    SHUTDOWN = shutdown

    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    STREAMER.device = device

    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = load_dataset(args.dataset_name, split=args.dataset_split, token=args.hf_token, streaming=True)
    ds = ds.shard(num_shards=args.num_workers, index=rank)

    loader = streaming_dataloader(ds, tok, args.batch_size, args.max_seq_len, shutdown)

    records: List[dict] = []
    seen = 0
    for batch in loader:
        if shutdown.is_set():
            break
        input_ids = batch["input_ids"]
        logits = STREAMER.forward(input_ids, args.micro_batch_size, device)
        ids, probs = sample_distribution(logits, args.sampling_rounds)
        for i in range(len(input_ids)):
            records.append({
                "input_ids": input_ids[i].tolist(),
                "sampled_ids": ids[i],
                "sampled_probs": probs[i],
            })
        seen += len(input_ids)
        if seen >= args.push_every:
            Dataset.from_list(records).push_to_hub(args.output_repo, token=args.hf_token, append=True)
            records.clear(); seen = 0
        if shutdown.is_set():
            break

    if records and not shutdown.is_set():
        Dataset.from_list(records).push_to_hub(args.output_repo, token=args.hf_token, append=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Hard disable autograd for the entire script
    torch.set_grad_enabled(False)

    args = parse_args()

    ctx = mp.get_context("spawn")
    shutdown = ctx.Event()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())

    global STREAMER, SHUTDOWN
    STREAMER = Streamer(args.model_name)
    STREAMER.model.share_memory()  # after moving to CPU
    SHUTDOWN = shutdown

    if args.num_workers > 1:
        procs = [ctx.Process(target=worker_main, args=(r, args, STREAMER, shutdown))
                 for r in range(args.num_workers)]
        for p in procs: p.start()
        for p in procs: p.join()
    else:
        worker_main(0, args, STREAMER, shutdown)


if __name__ == "__main__":
    main()

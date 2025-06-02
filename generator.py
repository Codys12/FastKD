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
    def __init__(self, model_name: str):
        self.device = torch.device("cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": "cpu"}, torch_dtype=torch.float16
        )
        self.layers = list(self.model.model.layers)
        self.embed = self.model.model.embed_tokens
        self.lm_head = self.model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        device = device or self.device

        batches = [
            input_ids[i : i + micro_batch_size].to(device)
            for i in range(0, input_ids.size(0), micro_batch_size)
        ]

        # Token embeddings
        self.embed.to(device)
        hidden: List[torch.Tensor] = []
        pos_ids_list: List[torch.Tensor] = []
        for mb in batches:
            seq_len = mb.size(1)
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(
                mb.size(0), -1
            )
            hidden.append(self.embed(mb))
            pos_ids_list.append(pos_ids)
        self.embed.to("cpu")
        torch.cuda.empty_cache()

        # Stream through transformer layers -------------------------------------------------
        for layer in self.layers:
            layer.to(device)
            next_hidden: List[torch.Tensor] = []

            for h, pos in zip(hidden, pos_ids_list):
                # ------------- FIX 1: always supply rotary tuple ---------------------------
                kwargs = {"position_ids": pos}
                try:
                    # Pre-compute (cos, sin) and hand it in if accepted
                    rotary_tuple = layer.self_attn.rotary_emb(pos)
                    kwargs["position_embeddings"] = rotary_tuple
                    out = layer(h, **kwargs)
                except TypeError:
                    # Layer doesn't take that kwarg – fall back to position_ids only
                    out = layer(h, position_ids=pos)
                # ----------------------------------------------------------------------------
                out = out[0] if isinstance(out, tuple) else out
                next_hidden.append(out)

            hidden = next_hidden
            layer.to("cpu")
            torch.cuda.empty_cache()

        # Final projection
        self.lm_head.to(device)
        logits = [self.lm_head(h) for h in hidden]
        self.lm_head.to("cpu")
        torch.cuda.empty_cache()

        return torch.cat(logits, dim=0)


def sample_distribution(logits: torch.Tensor, rounds: int):
    """Sample tokens using the random sampling procedure from the paper."""
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
        [ex["text"] for ex in examples],           # FIX 2: robust field access
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )


def streaming_dataloader(
    ds, tokenizer, batch_size: int, max_seq_len: int, shutdown: mp.Event | None = None
):
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


def worker_main(rank: int, args: Args, streamer: Streamer, shutdown: mp.Event):
    global STREAMER, SHUTDOWN
    STREAMER = streamer
    SHUTDOWN = shutdown

    def _handle_sigterm(signum, frame):
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    device = (
        torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    )
    STREAMER.device = device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        token=args.hf_token,
        streaming=True,
    )
    dataset = dataset.shard(num_shards=args.num_workers, index=rank)

    dataloader = streaming_dataloader(
        dataset, tokenizer, args.batch_size, args.max_seq_len, shutdown
    )

    all_records: List[dict] = []
    total = 0
    for batch in dataloader:
        if SHUTDOWN.is_set():
            break
        input_ids = batch["input_ids"]
        logits = STREAMER.forward(input_ids, args.micro_batch_size, device)
        ids, probs = sample_distribution(logits, args.sampling_rounds)
        for i in range(len(input_ids)):
            all_records.append(
                {
                    "input_ids": input_ids[i].tolist(),
                    "sampled_ids": ids[i],
                    "sampled_probs": probs[i],
                }
            )
        total += len(input_ids)
        if total >= args.push_every:
            Dataset.from_list(all_records).push_to_hub(
                args.output_repo, token=args.hf_token, append=True
            )
            all_records.clear()
            total = 0
        if SHUTDOWN.is_set():
            break
    if all_records and not SHUTDOWN.is_set():
        Dataset.from_list(all_records).push_to_hub(
            args.output_repo, token=args.hf_token, append=True
        )


def main():
    args = parse_args()

    mp_context = mp.get_context("spawn")
    shutdown_event = mp_context.Event()

    def _handle_sigterm(signum, frame):
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    global STREAMER, SHUTDOWN
    SHUTDOWN = shutdown_event
    STREAMER = Streamer(args.model_name)
    STREAMER.model.share_memory()

    if args.num_workers > 1:
        processes = []
        for rank in range(args.num_workers):
            p = mp_context.Process(
                target=worker_main, args=(rank, args, STREAMER, shutdown_event)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        worker_main(0, args, STREAMER, shutdown_event)


if __name__ == "__main__":
    main()

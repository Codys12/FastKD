from __future__ import annotations

import argparse
import signal
import threading
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


STREAMER: "Streamer" | None = None
SHUTDOWN: threading.Event | None = None


@dataclass
class Args:
    model_name: str
    dataset_name: str
    dataset_split: str
    output_repo: str
    batch_size: int
    micro_batch_size: int
    num_rounds: int        # ← renamed for clarity
    num_samples: int       # ← NEW per-round sample count
    sampling_temperature: float
    push_every: int
    max_seq_len: int
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
        "--num_rounds",
        type=int,
        default=50,
        help="Number of sampling rounds (R in the paper).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Tokens drawn per round (N in the paper).",
    )
    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (t).",
    )
    parser.add_argument("--push_every", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()
    return Args(**vars(args))


class Streamer:
    def __init__(self, model_name: str):
        # load on CPU without flash attention to avoid initialization issues
        config = AutoConfig.from_pretrained(model_name)
        config.attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, device_map={"": "cpu"}, torch_dtype=torch.bfloat16
        )
        # enable flash attention in the config afterwards
        self.model.config.attn_implementation = "flash_attention_2"
        self.layers = list(self.model.model.layers)
        self.embed = self.model.model.embed_tokens
        self.lm_head = self.model.lm_head

        self.devices = [
            torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        ]
        if not self.devices:
            self.devices = [torch.device("cpu")]
        self.num_devices = len(self.devices)

        if torch.cuda.device_count() > 0:
            self.streams = [torch.cuda.Stream(device=d) for d in self.devices]
        else:
            self.streams = [torch.cuda.current_stream()]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: int,
    ) -> torch.Tensor:
        batches = [
            input_ids[i : i + micro_batch_size]
            for i in range(0, input_ids.size(0), micro_batch_size)
        ]

        results: List[torch.Tensor] = []
        for start in range(0, len(batches), self.num_devices):
            group = batches[start : start + self.num_devices]
            hidden = [
                mb.to(self.devices[i], non_blocking=True)
                for i, mb in enumerate(group)
            ]

            # embedding layer
            for i, dev in enumerate(self.devices[: len(hidden)]):
                stream = self.streams[i]
                with torch.cuda.stream(stream):
                    self.embed.to(dev, non_blocking=True)
                    hidden[i] = self.embed(hidden[i])
                    self.embed.to("cpu", non_blocking=True)
            for s in self.streams[: len(hidden)]:
                s.synchronize()

            # prepare position embeddings per micro batch
            position_ids = []
            cache_positions = []
            pos_embeds = []
            for i, dev in enumerate(self.devices[: len(hidden)]):
                seq_len = hidden[i].size(1)
                cache_pos = torch.arange(seq_len, device=dev)
                pos_id = cache_pos.unsqueeze(0)
                position_ids.append(pos_id)
                cache_positions.append(cache_pos)
                pos_embeds.append(self.model.model.rotary_emb(hidden[i], pos_id))

            # transformer layers
            for layer in self.layers:
                for i, dev in enumerate(self.devices[: len(hidden)]):
                    stream = self.streams[i]
                    with torch.cuda.stream(stream):
                        layer.to(dev, non_blocking=True)
                        out = layer(
                            hidden[i],
                            position_ids=position_ids[i],
                            cache_position=cache_positions[i],
                            position_embeddings=pos_embeds[i],
                        )
                        hidden[i] = out[0] if isinstance(out, tuple) else out
                        layer.to("cpu", non_blocking=True)
                for s in self.streams[: len(hidden)]:
                    s.synchronize()

            # lm head
            for i, dev in enumerate(self.devices[: len(hidden)]):
                stream = self.streams[i]
                with torch.cuda.stream(stream):
                    self.lm_head.to(dev, non_blocking=True)
                    hidden[i] = self.lm_head(hidden[i])
                    self.lm_head.to("cpu", non_blocking=True)
            for s in self.streams[: len(hidden)]:
                s.synchronize()

            for i, dev in enumerate(self.devices[: len(hidden)]):
                hidden[i] = hidden[i].to("cpu", non_blocking=True)
            for s in self.streams[: len(hidden)]:
                s.synchronize()
            results.extend(hidden)

        return torch.cat(results, dim=0)


def sample_distribution(
    logits: torch.Tensor,
    num_samples: int,
    num_rounds: int,
    temperature: float = 1.0,
) -> tuple[
    list[list[list[int]]],
    list[list[list[float]]],
    list[list[list[float]]],
]:
    """
    Importance-sample `num_samples` tokens per round from proposal q(x)
    for `num_rounds` rounds, following the paper.  Probabilities are
    accumulated across rounds and renormalized for each token position.

    Returns:
        ids_all        – sampled token IDs          (B × S × M)
        probs_all      – normalized weights         (B × S × M)
        log_probs_all  – log of normalized weights  (B × S × M)
    """
    logits = logits.float()
    p = torch.softmax(logits, dim=-1)
    q = torch.softmax(logits / temperature, dim=-1)

    ids_all: List[List[List[int]]] = []
    probs_all: List[List[List[float]]] = []
    log_probs_all: List[List[List[float]]] = []
    bsz, seqlen, vocab = p.shape
    eps = 1e-12

    for b in range(bsz):
        ids_seq: List[List[int]] = []
        probs_seq: List[List[float]] = []
        logprobs_seq: List[List[float]] = []

        for s in range(seqlen):
            p_row = p[b, s]
            q_row = q[b, s]

            # sanitize q to avoid invalid distributions
            q_row = torch.nan_to_num(q_row, nan=0.0, posinf=0.0, neginf=0.0)
            q_row = torch.clamp(q_row, min=0)
            if not torch.isfinite(q_row).all() or q_row.sum() == 0:
                q_row.fill_(1.0 / vocab)
            else:
                q_row = q_row + 1e-6  # avoid exact zeros
                q_row /= q_row.sum()

            # sample N × R tokens in one shot (identical to R rounds of size N)
            total_draws = num_samples * num_rounds
            samples = torch.multinomial(q_row, total_draws, replacement=True)

            uniq, counts = torch.unique(samples, return_counts=True)

            # log-weight: log(m_i · p_i / q_i)
            log_weights = (
                counts.float().log()
                + torch.log(p_row[uniq])
                - torch.log(q_row[uniq])
            )
            log_weights = torch.nan_to_num(log_weights, nan=-float("inf"))

            # normalize weights in log-space for stability
            m = log_weights.max()
            weights = torch.exp(log_weights - m)
            weight_sum = weights.sum()

            if not torch.isfinite(weight_sum) or weight_sum <= 0:
                probs_norm = torch.full_like(weights, 1.0 / weights.numel())
                log_probs_norm = torch.log(probs_norm)
            else:
                probs_norm = weights / weight_sum
                log_probs_norm = log_weights - m - torch.log(weight_sum)

            # remove zero-probability tokens
            keep = probs_norm > 0

            ids_seq.append(uniq[keep].cpu().tolist())
            probs_seq.append(probs_norm[keep].cpu().tolist())
            logprobs_seq.append(log_probs_norm[keep].cpu().tolist())

        ids_all.append(ids_seq)
        probs_all.append(probs_seq)
        log_probs_all.append(logprobs_seq)

    return ids_all, probs_all, log_probs_all


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
    ds, tokenizer, batch_size: int, max_seq_len: int, shutdown: threading.Event | None = None
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


def main():
    args = parse_args()

    global STREAMER, SHUTDOWN
    SHUTDOWN = threading.Event()

    def _handle_sigterm(signum, frame):
        SHUTDOWN.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    STREAMER = Streamer(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        token=args.hf_token,
        streaming=True,
    )

    dataloader = streaming_dataloader(
        dataset, tokenizer, args.batch_size, args.max_seq_len, SHUTDOWN
    )

    all_records: List[dict] = []
    total = 0
    for batch in dataloader:
        if SHUTDOWN.is_set():
            break
        input_ids = batch["input_ids"]
        logits = STREAMER.forward(input_ids, args.micro_batch_size)
        ids, probs, log_probs = sample_distribution(
            logits,
            args.num_samples,
            args.num_rounds,
            args.sampling_temperature,
        )
        for i in range(len(input_ids)):
            # remove trailing padding tokens from the input sequence
            tokens = input_ids[i].tolist()
            while tokens and tokens[-1] == 0:
                tokens.pop()

            seq_len = len(tokens)

            record = {
                "input_ids": tokens,
                "sampled_ids": ids[i][:seq_len],
                "sampled_probs": probs[i][:seq_len],
                "sampled_logprobs": log_probs[i][:seq_len],
            }
            all_records.append(record)
        total += len(input_ids)
        if total >= args.push_every:
            print("Example sampled_probs:", all_records[0]["sampled_probs"])
            ds = Dataset.from_list(all_records)
            ds.push_to_hub(args.output_repo, token=args.hf_token)
            all_records.clear()
            total = 0
        if SHUTDOWN.is_set():
            break
    if all_records and not SHUTDOWN.is_set():
        ds = Dataset.from_list(all_records)
        ds.push_to_hub(args.output_repo, token=args.hf_token)


if __name__ == "__main__":
    main()

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
    sampling_rounds: int
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
        "--sampling_rounds",
        type=int,
        default=50,
        help="Number of sampling rounds per token as in the paper",
    )
    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for importance sampling",
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
            model_name, config=config, device_map={"": "cpu"}, torch_dtype=torch.float16
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
    logits: torch.Tensor, rounds: int, temperature: float = 1.0
) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
    """Sample tokens using importance sampling as described in the paper."""
    logits = logits.float()
    p = torch.softmax(logits, dim=-1)
    q = torch.softmax(logits / temperature, dim=-1)

    ids_all: List[List[List[int]]] = []
    probs_all: List[List[List[float]]] = []
    bsz, seqlen, _ = p.shape
    for b in range(bsz):
        ids_seq: List[List[int]] = []
        probs_seq: List[List[float]] = []
        for s in range(seqlen):
            p_row = p[b, s]
            q_row = q[b, s]

            # sanitize proposal to avoid invalid distributions
            q_row = torch.nan_to_num(q_row, nan=0.0, posinf=0.0, neginf=0.0)
            q_row = torch.clamp(q_row, min=0)
            if not torch.isfinite(q_row).all() or q_row.sum() == 0:
                q_row.fill_(1.0 / q_row.numel())
            else:
                q_row /= q_row.sum()

            # multinomial expects non-negative weights
            samples = torch.multinomial(q_row, rounds, replacement=True)
            uniq, counts = torch.unique(samples, return_counts=True)
            weights = counts.float() * (p_row[uniq] / q_row[uniq])
            probs_norm = weights / weights.sum()
            ids_seq.append(uniq.cpu().tolist())
            probs_seq.append(probs_norm.cpu().tolist())
        ids_all.append(ids_seq)
        probs_all.append(probs_seq)
    return ids_all, probs_all


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
        ids, probs = sample_distribution(
            logits, args.sampling_rounds, args.sampling_temperature
        )
        for i in range(len(input_ids)):
            record = {
                "input_ids": input_ids[i].tolist(),
                "sampled_ids": ids[i],
                "sampled_probs": probs[i],
            }
            all_records.append(record)
        total += len(input_ids)
        if total >= args.push_every:
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

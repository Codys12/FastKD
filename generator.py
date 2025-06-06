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


# --------------------------------------------------------------------------- #
#                                CLI arguments                                #
# --------------------------------------------------------------------------- #
@dataclass
class Args:
    model_name: str
    dataset_name: str
    dataset_split: str
    output_repo: str
    batch_size: int
    micro_batch_size: int
    num_rounds: int        # total multinomial draws per position
    sampling_temperature: float
    push_every: int
    max_seq_len: int
    hf_token: str | None = None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Generate KD dataset (ids + counts)")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--output_repo", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=255,
        help="Number of draws per token position.",
    )
    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument("--push_every", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()
    return Args(**vars(args))


# --------------------------------------------------------------------------- #
#                                   Model                                     #
# --------------------------------------------------------------------------- #
class Streamer:
    """Runs the model micro-batch-wise and samples token IDs with counts."""

    def __init__(self, model_name: str):
        # load on CPU (no flash-attention yet)
        config = AutoConfig.from_pretrained(model_name)
        config.attn_implementation = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, device_map={"": "cpu"}, torch_dtype=torch.bfloat16
        )
        # enable flash-attention afterwards
        self.model.config.attn_implementation = "flash_attention_2"
        self.layers = list(self.model.model.layers)
        self.embed = self.model.model.embed_tokens
        self.lm_head = self.model.lm_head

        self.devices = [
            torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
        ] or [torch.device("cpu")]
        self.num_devices = len(self.devices)
        self.streams = (
            [torch.cuda.Stream(device=d) for d in self.devices]
            if torch.cuda.device_count() > 0
            else [torch.cuda.current_stream()]
        )

    def _broadcast_module(self, module: torch.nn.Module) -> None:
        """Broadcast module parameters from device 0 to all other devices asynchronously."""
        if self.num_devices <= 1:
            return
        tensors = [p.data for p in module.parameters()] + [b.data for b in module.buffers()]
        work = torch.cuda.comm.broadcast_coalesced(tensors, self.devices, async_op=True)
        # wait for the broadcast to complete before using the parameters
        if hasattr(work, "wait"):
            work.wait()

    # --------------------------------------------------------------------- #
    #                  Forward pass used by `sample()`                      #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: int,
    ) -> torch.Tensor:
        """Plain forward pass returning logits on CPU."""
        batches = [
            input_ids[i : i + micro_batch_size]
            for i in range(0, input_ids.size(0), micro_batch_size)
        ]

        results: List[torch.Tensor] = []
        copy_streams = [torch.cuda.Stream(device=d) for d in self.devices]
        for start in range(0, len(batches), self.num_devices):
            group = batches[start : start + self.num_devices]
            hidden = [
                mb.to(self.devices[i], non_blocking=True)
                for i, mb in enumerate(group)
            ]

            # embeddings -------------------------------------------------- #
            with torch.cuda.stream(copy_streams[0]):
                self.embed.to(self.devices[0], non_blocking=True)
            for i in range(1, len(hidden)):
                with torch.cuda.stream(copy_streams[i]):
                    copy_streams[i].wait_stream(copy_streams[i - 1])
                    self.embed.to(self.devices[i], non_blocking=True)
            events = [torch.cuda.Event() for _ in range(len(hidden))]
            for i, dev in enumerate(self.devices[: len(hidden)]):
                copy_streams[i].record_event(events[i])
                with torch.cuda.stream(self.streams[i]):
                    self.streams[i].wait_event(events[i])
                    hidden[i] = self.embed(hidden[i])
            for s in self.streams[: len(hidden)]:
                s.synchronize()
            self.embed.to("cpu", non_blocking=True)

            # rotary embeddings/position ids ------------------------------ #
            position_ids, cache_positions, rot_embeds = [], [], []
            for i, dev in enumerate(self.devices[: len(hidden)]):
                seq_len = hidden[i].size(1)
                cache_pos = torch.arange(seq_len, device=dev)
                position_ids.append(cache_pos.unsqueeze(0))
                cache_positions.append(cache_pos)
                rot_embeds.append(self.model.model.rotary_emb(hidden[i], position_ids[-1]))

            # transformer layers ------------------------------------------ #
            for layer in self.layers:
                with torch.cuda.stream(copy_streams[0]):
                    layer.to(self.devices[0], non_blocking=True)
                for i in range(1, len(hidden)):
                    with torch.cuda.stream(copy_streams[i]):
                        copy_streams[i].wait_stream(copy_streams[i - 1])
                        layer.to(self.devices[i], non_blocking=True)
                events = [torch.cuda.Event() for _ in range(len(hidden))]
                for i, dev in enumerate(self.devices[: len(hidden)]):
                    copy_streams[i].record_event(events[i])
                    with torch.cuda.stream(self.streams[i]):
                        self.streams[i].wait_event(events[i])
                        out = layer(
                            hidden[i],
                            position_ids=position_ids[i],
                            cache_position=cache_positions[i],
                            position_embeddings=rot_embeds[i],
                        )
                        hidden[i] = out[0] if isinstance(out, tuple) else out
                for s in self.streams[: len(hidden)]:
                    s.synchronize()
                layer.to("cpu", non_blocking=True)
            
            # LM head ------------------------------------------------------ #
            with torch.cuda.stream(copy_streams[0]):
                self.lm_head.to(self.devices[0], non_blocking=True)
            for i in range(1, len(hidden)):
                with torch.cuda.stream(copy_streams[i]):
                    copy_streams[i].wait_stream(copy_streams[i - 1])
                    self.lm_head.to(self.devices[i], non_blocking=True)
            events = [torch.cuda.Event() for _ in range(len(hidden))]
            for i, dev in enumerate(self.devices[: len(hidden)]):
                copy_streams[i].record_event(events[i])
                with torch.cuda.stream(self.streams[i]):
                    self.streams[i].wait_event(events[i])
                    outs = [self.lm_head(h.unsqueeze(0)) for h in hidden[i]]
                    hidden[i] = torch.cat(outs, dim=0)
            for s in self.streams[: len(hidden)]:
                s.synchronize()
            self.lm_head.to("cpu", non_blocking=True)
            for s in self.streams[: len(hidden)]:
                s.synchronize()

            # back to CPU -------------------------------------------------- #
            for i in range(len(hidden)):
                hidden[i] = hidden[i].to("cpu", non_blocking=True)
            for s in self.streams[: len(hidden)]:
                s.synchronize()
            results.extend(hidden)

        return torch.cat(results, dim=0)

    # --------------------------------------------------------------------- #
    #                           Sampling wrapper                             #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: int,
        num_rounds: int,
        temperature: float = 1.0,
    ) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
        """Return token IDs and their draw counts for each sequence position."""
        batches = [
            input_ids[i : i + micro_batch_size]
            for i in range(0, input_ids.size(0), micro_batch_size)
        ]

        ids_all, counts_all = [], []
        for start in range(0, len(batches), self.num_devices):
            group = batches[start : start + self.num_devices]
            # full logits on CPU
            logits = self.forward(torch.cat(group, dim=0), micro_batch_size)
            # split back into the group size
            split_sizes = [g.size(0) for g in group]
            logits_split = torch.split(logits, split_sizes, dim=0)
            for l in logits_split:
                ids, counts = sample_distribution(
                    l, num_draws=num_rounds, temperature=temperature
                )
                ids_all.extend(ids)
                counts_all.extend(counts)

        return ids_all, counts_all


# --------------------------------------------------------------------------- #
#                             Sampling utilities                              #
# --------------------------------------------------------------------------- #
def sample_distribution(
    logits: torch.Tensor,
    num_draws: int,
    temperature: float = 1.0,
) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    """
    Draw `num_draws` samples for every batch-step-token position.
    Returns:
        ids_all    – token IDs (B × S × K)
        counts_all – draw counts (B × S × K)
    """
    logits = logits.float()
    q = torch.softmax(logits / temperature, dim=-1)

    ids_all, counts_all = [], []
    bsz, seqlen, _ = logits.shape

    for b in range(bsz):
        ids_seq, counts_seq = [], []
        for s in range(seqlen):
            q_row = q[b, s]

            # fix any numerical issues
            q_row = torch.nan_to_num(q_row, nan=0.0, posinf=0.0, neginf=0.0)
            if q_row.sum() == 0 or not torch.isfinite(q_row).all():
                q_row.fill_(1.0 / q_row.numel())
            else:
                q_row = q_row + 1e-6
                q_row /= q_row.sum()

            samples = torch.multinomial(q_row, num_draws, replacement=True)
            uniq, cnts = torch.unique(samples, return_counts=True)
            ids_seq.append(uniq.cpu().tolist())
            counts_seq.append(cnts.cpu().tolist())
        ids_all.append(ids_seq)
        counts_all.append(counts_seq)

    return ids_all, counts_all


# --------------------------------------------------------------------------- #
#                             Data helpers                                    #
# --------------------------------------------------------------------------- #
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
    ds, tokenizer, batch_size: int, max_seq_len: int, shutdown: threading.Event | None
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


# --------------------------------------------------------------------------- #
#                                    Main                                     #
# --------------------------------------------------------------------------- #
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
        ids, counts = STREAMER.sample(
            input_ids,
            args.micro_batch_size,
            args.num_rounds,
            args.sampling_temperature,
        )

        for i in range(len(input_ids)):
            # strip padding
            tokens = input_ids[i].tolist()
            while tokens and tokens[-1] == 0:
                tokens.pop()
            seq_len = len(tokens)

            record = {
                "input_ids": tokens,
                "sampled_ids": ids[i][:seq_len],
                "sampled_counts": counts[i][:seq_len],
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

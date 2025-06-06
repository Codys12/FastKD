from __future__ import annotations

import argparse
import signal
import threading
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import HfApi, Repository
import multiprocessing as mp
import os


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

            # lm head with micro-batches of size 1
            for i, dev in enumerate(self.devices[: len(hidden)]):
                stream = self.streams[i]
                with torch.cuda.stream(stream):
                    self.lm_head.to(dev, non_blocking=True)
                    outputs = []
                    for mb in hidden[i].split(1, dim=0):
                        out = self.lm_head(mb.to(dev, non_blocking=True))
                        outputs.append(out)
                    hidden[i] = torch.cat(outputs, dim=0)
                    self.lm_head.to("cpu", non_blocking=True)
            for s in self.streams[: len(hidden)]:
                s.synchronize()

            for i, dev in enumerate(self.devices[: len(hidden)]):
                hidden[i] = hidden[i].to("cpu", non_blocking=True)
            for s in self.streams[: len(hidden)]:
                s.synchronize()
            results.extend(hidden)

        return torch.cat(results, dim=0)

    @torch.no_grad()
    def sample(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: int,
        num_samples: int,
        num_rounds: int,
        temperature: float = 1.0,
    ) -> tuple[
        list[list[list[int]]],
        list[list[list[int]]],
    ]:
        """Run the model in microbatches and sample logits on-device."""
        batches = [
            input_ids[i : i + micro_batch_size]
            for i in range(0, input_ids.size(0), micro_batch_size)
        ]

        ids_all: list[list[list[int]]] = []
        counts_all: list[list[list[int]]] = []

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

            # lm head and sampling with micro batches of size 1
            for i, dev in enumerate(self.devices[: len(hidden)]):
                stream = self.streams[i]
                with torch.cuda.stream(stream):
                    self.lm_head.to(dev, non_blocking=True)
                    for mb in hidden[i].split(1, dim=0):
                        logits = self.lm_head(mb.to(dev, non_blocking=True))
                        ids, counts = sample_distribution(
                            logits,
                            num_samples=num_samples,
                            num_rounds=num_rounds,
                            temperature=temperature,
                        )
                        ids_all.extend(ids)
                        counts_all.extend(counts)
                    self.lm_head.to("cpu", non_blocking=True)
            for s in self.streams[: len(hidden)]:
                s.synchronize()

        return ids_all, counts_all


def sample_distribution(
    logits: torch.Tensor,
    num_samples: int,
    num_rounds: int,
    temperature: float = 1.0,
) -> tuple[
    list[list[list[int]]],
    list[list[list[int]]],
]:
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

            # sanitize q to avoid invalid distributions
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


def push_shard(records: List[dict], args: Args) -> None:
    """Append a dataset shard to the Hub repository."""
    api = HfApi(token=args.hf_token)
    files = api.list_repo_files(args.output_repo, repo_type="dataset")
    idx = sum(f.endswith(".parquet") for f in files)

    filename = f"data_{idx:05d}.parquet"
    ds = Dataset.from_list(records)
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


def worker(rank: int, args: Args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

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
            all_records.append(record)
        total += len(input_ids)
        if total >= args.push_every:
            print("Example sampled_counts:", all_records[0]["sampled_counts"])
            push_shard(all_records, args)
            all_records.clear()
            total = 0
        if SHUTDOWN.is_set():
            break
    if all_records and not SHUTDOWN.is_set():
        push_shard(all_records, args)

def main():
    args = parse_args()

    num_devices = torch.cuda.device_count() or 1

    if num_devices > 1:
        procs = []
        for rank in range(num_devices):
            p = mp.Process(target=worker, args=(rank, args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        worker(0, args)


if __name__ == "__main__":
    main()

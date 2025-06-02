import argparse
import os
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def init_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.distributed.init_process_group("nccl")


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
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()
    return Args(**vars(args))


class Streamer:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": "cpu"}, torch_dtype=torch.float16
        )
        self.layers = list(self.model.model.layers)
        self.embed = self.model.model.embed_tokens
        self.lm_head = self.model.lm_head

    def forward(self, input_ids: torch.Tensor, micro_batch_size: int) -> torch.Tensor:
        batches = [
            input_ids[i : i + micro_batch_size].to(self.device)
            for i in range(0, input_ids.size(0), micro_batch_size)
        ]

        self.embed.to(self.device)
        hidden = [self.embed(mb) for mb in batches]
        self.embed.to("cpu")
        torch.cuda.empty_cache()

        for layer in self.layers:
            layer.to(self.device)
            next_hidden = []
            for h in hidden:
                out = layer(h)
                out = out[0] if isinstance(out, tuple) else out
                next_hidden.append(out)
            hidden = next_hidden
            layer.to("cpu")
            torch.cuda.empty_cache()

        self.lm_head.to(self.device)
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
        ids_seq = []
        probs_seq = []
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
        examples["text"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )


def main():
    args = parse_args()
    init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        token=args.hf_token,
        streaming=True,
    )

    if torch.distributed.is_initialized():
        dataset = dataset.shard(
            num_shards=torch.distributed.get_world_size(),
            index=local_rank,
        )

    def streaming_dataloader(ds):
        batch = []
        for example in ds:
            batch.append(example)
            if len(batch) == args.batch_size:
                yield collate_fn(batch, tokenizer, args.max_seq_len)
                batch.clear()
        if batch:
            yield collate_fn(batch, tokenizer, args.max_seq_len)

    dataloader = streaming_dataloader(dataset)
    streamer = Streamer(args.model_name, device)

    all_records: List[dict] = []
    total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"]
        logits = streamer.forward(input_ids, args.micro_batch_size)
        ids, probs = sample_distribution(logits, args.sampling_rounds)
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
            ds.push_to_hub(args.output_repo, token=args.hf_token, append=True)
            all_records.clear()
            total = 0
    if all_records:
        ds = Dataset.from_list(all_records)
        ds.push_to_hub(args.output_repo, token=args.hf_token, append=True)


if __name__ == "__main__":
    main()

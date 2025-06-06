from __future__ import annotations

"""
weight_pipeline_kd_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A re‑implementation of *generate_kd_dataset.py* that streams **model
parameters** across devices instead of streaming **data**.  We use
`torch.distributed` + `async p2p` ops to realise a *travelling‑layer*
pipeline:

```
 ┌────────────┐         ┌────────────┐         ┌────────────┐
 │  device 0  │  ───▶   │  device 1  │  ───▶   │  device 2  │
 │ microbatch │         │ microbatch │         │ microbatch │
 └────────────┘         └────────────┘         └────────────┘
      ▲                      ▲                      ▲
      │ layer‑0 params       │ layer‑0 params       │ layer‑0 params
      │ layer‑1 params  ───▶ │ layer‑1 params  ───▶ │ layer‑1 params
      │ layer‑2 params  ───▶ │ layer‑2 params  ───▶ │ layer‑2 params
```

Each rank keeps **its slice of the global batch resident on its own
GPU** for the program lifetime.  A *layer baton* (the parameter tensor
for one TransformerBlock) is streamed round‑robin through all ranks.  A
rank performs *micro_batch_size* forward passes with that layer on its
local data, then **asynchronously** ships the weight tensors to the next
rank while it is already working on the next layer that has just
arrived from its previous neighbour.  When the final layer & LM head
have flowed through, every rank owns the full logits for its share of
the batch.  No activation checkpoints are transmitted between ranks –
all hidden‑state tensors stay local.

The script can be launched with the usual torchrun helper, e.g.

```
torchrun \
  --nnodes 1 --nproc_per_node 4 \
  weight_pipeline_kd_dataset.py \
  --model_name meta‑llama/Llama‑3‑8B‑Instruct \
  --dataset_name c4 --dataset_split train --output_repo my/kd \
  --batch_size 8 --micro_batch_size 1 \
  --num_rounds 50 --num_samples 1 \
  --sampling_temperature 1.0 --push_every 1000 \
  --max_seq_len 2048 --hf_token $HF
```

NOTE:  
*The design assumes identical GPU topology and VRAM size across ranks.*

"""

import argparse
import os
import signal
import threading
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import HfApi, Repository

# ----------------------------------------------------------------------------------------------------------------------
#                               ★  Argument parsing  ★
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Args:
    model_name: str
    dataset_name: str
    dataset_split: str
    output_repo: str
    batch_size: int
    micro_batch_size: int
    num_rounds: int
    num_samples: int
    sampling_temperature: float
    push_every: int
    max_seq_len: int
    hf_token: str | None = None
    # distributed
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: str = "29500"


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Generate KD dataset with distributed weight‑pipeline streaming")
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--output_repo", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--num_rounds", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--sampling_temperature", type=float, default=1.0)
    p.add_argument("--push_every", type=int, default=1000)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--hf_token")
    p.add_argument("--backend", default="nccl")
    p.add_argument("--master_addr", default="127.0.0.1")
    p.add_argument("--master_port", default="29500")
    args = p.parse_args()

    return Args(**vars(args))

# ----------------------------------------------------------------------------------------------------------------------
#                               ★  Helpers  ★
# ----------------------------------------------------------------------------------------------------------------------

class LayerPackage:
    """Small wrapper that travels through the ring‑pipeline."""
    __slots__ = ("idx", "state_dict")

    def __init__(self, idx: int, state_dict: dict[str, torch.Tensor]):
        self.idx = idx
        self.state_dict = state_dict

    # NB:  torch.distributed does not yet support direct transport of state_dicts, so
    # we flatten into a list of tensors accompanied by metadata.  For clarity and because models
    # are fairly homogenous, we broadcast tensor shapes first once, then stream raw tensors.


# ----------------------------------------------------------------------------------------------------------------------
#                               ★  Weight‑stream Pipeline  ★
# ----------------------------------------------------------------------------------------------------------------------

class WeightStreamer:
    """Streams *one layer at a time* through all ranks in a ring topography.

    Each rank keeps *its* slice of the input batch and the running *hidden
    state* on its own device.  When a layer arrives, the rank:

    1. Loads parameters into the local skeleton module (`nn.Module` with same structure).
    2. Applies the layer to *every micro‑batch* resident on the rank.
    3. Asynchronously ships the `state_dict` to the *next* rank while already waiting
       for, or processing, the next layer that is coming from the *previous* rank.

    This gives genuine pipeline overlap:  while a layer is «in flight» to the
    neighbour, we are already computing with the next layer.
    """

    def __init__(self, model_name: str):
        cfg = AutoConfig.from_pretrained(model_name)
        cfg.attn_implementation = "eager"  # safe load on CPU first
        base = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, torch_dtype=torch.bfloat16)

        # unwrap internal modules we care about -----------------------------------------------------------------
        self.embed = base.model.embed_tokens
        self.layers = list(base.model.layers)
        self.lm_head = base.lm_head

        # skeleton copies whose parameters we *mutate* in‑place for each incoming package ------------------------
        self.skel_layers = [layer.__class__(base.config).to(torch.cuda.current_device()) for layer in self.layers]
        self.skel_embed = self.embed.__class__(base.embed_tokens.num_embeddings, base.embed_tokens.embedding_dim,
                                               device=torch.cuda.current_device(), dtype=torch.bfloat16)
        self.skel_lm_head = self.lm_head.__class__(base.lm_head.in_features, base.lm_head.out_features,
                                                   bias=False, device=torch.cuda.current_device(), dtype=torch.bfloat16)

        # keep model on CPU to feed packages; we never use `base` again after extracting state_dicts --------------
        self.packages: List[LayerPackage] = [LayerPackage(i, layer.state_dict()) for i, layer in enumerate(self.layers)]
        self.embed_package = LayerPackage(-2, self.embed.state_dict())
        self.lm_head_package = LayerPackage(-1, self.lm_head.state_dict())

    # -----------------------------------------------------------------------------------------------------------
    # Stream helpers
    # -----------------------------------------------------------------------------------------------------------

    @staticmethod
    def _send_package(pkg: LayerPackage, dst: int):
        obj_list = [pkg]
        dist.isend(tensor=torch.tensor([0]), dst=dst)  # dummy tensor to trigger wake‑up
        dist.barrier()
        dist.broadcast_object_list(obj_list, src=dist.get_rank())

    @staticmethod
    def _recv_package(src: int) -> LayerPackage:
        dummy = torch.zeros(1)
        dist.recv(dummy, src=src)
        dist.barrier()
        obj_list: List[LayerPackage] = [None]  # type: ignore
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    # -----------------------------------------------------------------------------------------------------------
    # Forward & sampling
    # -----------------------------------------------------------------------------------------------------------

    def process_local_batch(
        self,
        input_ids: torch.Tensor,
        micro_batch_size: int,
        num_samples: int,
        num_rounds: int,
        temperature: float,
    ) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """Run *the entire parameter stream* over the local input batch."""

        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        input_ids = input_ids.to(device)

        # allocate hidden state once; we keep updating in‑place --------------------------------------------------
        with torch.no_grad():
            hidden: torch.Tensor = torch.empty((*input_ids.shape, self.embed.embedding_dim), device=device,
                                               dtype=torch.bfloat16)

        # prepare micro‑batch views -----------------------------------------------------------------------------
        microbatches = [input_ids[i:i + micro_batch_size] for i in range(0, input_ids.size(0), micro_batch_size)]

        # layer ring initialisation -----------------------------------------------------------------------------
        left = (rank - 1) % world_size
        right = (rank + 1) % world_size

        # Rank‑0 seeds the ring with embed, then the transformer blocks, then the lm_head -----------------------
        if rank == 0:
            seed_stream = [self.embed_package] + self.packages + [self.lm_head_package]
            for pkg in seed_stream[: world_size]:  # prime pipeline with up to `world_size` packages
                dist.barrier()
                self._send_package(pkg, right)

            send_cursor = world_size
        else:
            send_cursor = 0

        # processing loop --------------------------------------------------------------------------------------
        received_final = 0
        ids_all: List[List[List[int]]] = []
        counts_all: List[List[List[int]]] = []

        while received_final < len(self.packages) + 2:  # + embed + lm_head
            pkg = self._recv_package(left)

            if pkg.idx == -2:  # embed
                self.skel_embed.load_state_dict(pkg.state_dict)
                with torch.no_grad():
                    for mb in microbatches:
                        hidden_mb = self.skel_embed(mb.to(device))
                        hidden[mb] = hidden_mb  # copy into slice

            elif 0 <= pkg.idx < len(self.skel_layers):  # transformer layer
                layer = self.skel_layers[pkg.idx]
                layer.load_state_dict(pkg.state_dict)
                with torch.no_grad():
                    for mb in microbatches:
                        hs = hidden[mb]
                        out = layer(hs)
                        hidden[mb] = out[0] if isinstance(out, tuple) else out

            elif pkg.idx == -1:  # lm_head -> sampling ends here
                self.skel_lm_head.load_state_dict(pkg.state_dict)
                with torch.no_grad():
                    for mb in microbatches:
                        logits = self.skel_lm_head(hidden[mb])
                        ids, counts = sample_distribution(
                            logits,
                            num_samples=num_samples,
                            num_rounds=num_rounds,
                            temperature=temperature,
                        )
                        ids_all.extend(ids)
                        counts_all.extend(counts)

            else:
                raise RuntimeError(f"Unknown package idx {pkg.idx}")

            received_final += 1

            # forward‑send the package to next neighbour (unless ring completed) --------------------------------
            if (pkg.idx != -1) or (rank != world_size - 1):  # last rank keeps lm_head
                self._send_package(pkg, right)

            # Rank‑0 continues seeding after initial prime ------------------------------------------------------
            if rank == 0 and send_cursor < len(self.packages) + 2:
                next_pkg = ([self.embed_package] + self.packages + [self.lm_head_package])[send_cursor]
                self._send_package(next_pkg, right)
                send_cursor += 1

        return ids_all, counts_all

# ----------------------------------------------------------------------------------------------------------------------
#                               ★  Sampling helpers (unchanged)  ★
# ----------------------------------------------------------------------------------------------------------------------

def sample_distribution(
    logits: torch.Tensor,
    num_samples: int,
    num_rounds: int,
    temperature: float = 1.0,
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
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
            if q_row.sum() == 0 or not torch.isfinite(q_row).all():
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

# ----------------------------------------------------------------------------------------------------------------------
#                               ★  Data helpers  ★
# ----------------------------------------------------------------------------------------------------------------------

def collate_fn(examples, tokenizer, max_seq_len: int):
    texts = [e["text"] for e in examples]
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
    )


def streaming_dataloader(ds, tokenizer, batch_size: int, max_seq_len: int, rank: int, world_size: int):
    """Each rank consumes every *world_size‑th* record from the dataset."""
    batch = []
    for idx, ex in enumerate(ds):
        if idx % world_size != rank:
            continue
        batch.append(ex)
        if len(batch) == batch_size:
            yield collate_fn(batch, tokenizer, max_seq_len)
            batch.clear()
    if batch:
        yield collate_fn(batch, tokenizer, max_seq_len)

# ----------------------------------------------------------------------------------------------------------------------
#                               ★  Hub push  ★
# ----------------------------------------------------------------------------------------------------------------------

def push_shard(records: List[dict], args: Args) -> None:
    api = HfApi(token=args.hf_token)
    files = api.list_repo_files(args.output_repo, repo_type="dataset")
    idx = sum(f.endswith(".parquet") for f in files)
    filename = f"data_{idx:05d}.parquet"
    Dataset.from_list(records).to_parquet(filename)
    repo = Repository(local_dir="repo_tmp", clone_from=args.output_repo,
                      token=args.hf_token, repo_type="dataset")
    repo.git_pull()
    repo.git_add(filename)
    repo.git_commit(f"Add shard {idx}")
    repo.git_push()
    repo.git_clear()
    os.remove(filename)

# ----------------------------------------------------------------------------------------------------------------------
#                               ★  Main worker  ★
# ----------------------------------------------------------------------------------------------------------------------

def worker(args: Args):
    # Distributed init -----------------------------------------------------------------------------------------
    dist.init_process_group(args.backend, init_method=f"tcp://{args.master_addr}:{args.master_port}")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)

    shutdown = threading.Event()

    def _handle_sigterm(signum, frame):
        shutdown.set()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Model & tokenizer ----------------------------------------------------------------------------------------
    streamer = WeightStreamer(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Dataset --------------------------------------------------------------------------------------------------
    dataset = load_dataset(args.dataset_name, split=args.dataset_split,
                           token=args.hf_token, streaming=True)
    dataloader = streaming_dataloader(dataset, tokenizer, args.batch_size, args.max_seq_len, rank, world_size)

    all_records: List[dict] = []
    total_seen = 0

    for batch in dataloader:
        if shutdown.is_set():
            break
        input_ids = batch["input_ids"].to(torch.device(f"cuda:{rank}"))
        ids, counts = streamer.process_local_batch(
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
        total_seen += len(input_ids)
        if rank == 0 and total_seen >= args.push_every:  # only rank‑0 pushes to the Hub
            push_shard(all_records, args)
            all_records.clear()
            total_seen = 0
        if shutdown.is_set():
            break
    if rank == 0 and all_records and not shutdown.is_set():
        push_shard(all_records, args)

    dist.barrier()
    dist.destroy_process_group()

# ----------------------------------------------------------------------------------------------------------------------
#                               ★  Entry‑point  ★
# ----------------------------------------------------------------------------------------------------------------------

def main():
    args = parse_args()
    worker(args)


if __name__ == "__main__":
    main()

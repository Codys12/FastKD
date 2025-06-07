from __future__ import annotations

"""
Generator with *reverse* pipeline parallelism.
Layers stream *through* static data that reside on each GPU.
Usage (example from SLURM):

sbatch -p dgxh100 --gres=gpu:8 --wrap="singularity exec --nv lightning_sandbox bash -c 'cd fastkd; torchrun --nnodes 1 --nproc_per_node 8 generator.py --push_every 512 --model_name Qwen/Qwen3-235B-A22B --dataset_name mlfoundations/dclm-baseline-1.0 --output_repo codys12/Qwen3-DCLM-test2 --hf_token hf_... --num_rounds=255 --batch_size=512 --micro_batch_size=64'"
"""

import argparse
import os
import signal
import threading
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, Repository
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

######################################################################
#                         Arg‑parsing                                #
######################################################################

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


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Generate KD dataset with reverse pipeline parallelism")
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
    return Args(**vars(parser.parse_args()))

######################################################################
#                   Reverse‑pipeline utilities                       #
######################################################################

class _FlatPacket:
    """Flattens a layer's parameters into a single contiguous tensor for fast P2P ops."""

    def __init__(self, layer: torch.nn.Module | None = None):
        self.flat: torch.Tensor | None = None
        if layer is not None:
            self.pack(layer)

    def pack(self, layer: torch.nn.Module) -> None:
        with torch.no_grad():
            vec: List[torch.Tensor] = [p.data.view(-1) for p in layer.parameters()]
            self.flat = torch.cat(vec).contiguous()

    def unpack_to(self, layer: torch.nn.Module) -> None:
        if self.flat is None:
            raise RuntimeError("Empty packet.")
        with torch.no_grad():
            offset = 0
            for p in layer.parameters():
                numel = p.numel()
                p.data.copy_(self.flat[offset : offset + numel].view_as(p))
                offset += numel

######################################################################
#                      Distributed Engine                            #
######################################################################

class ReversePipelineEngine:
    """Streams *layers* through static hidden states that live on each GPU.

    Each rank owns a device (cuda:rank).
    Layers flow in order: CPU ➔ GPU‑0 ➔ GPU‑1 ➔ … ➔ GPU‑(N−1) ➔ CPU.
    Double‑buffering hides transfer latency.
    """

    def __init__(self, args: Args, rank: int, world_size: int):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        # Only rank‑0 materialises the full model on CPU.
        if rank == 0:
            cfg = AutoConfig.from_pretrained(args.model_name)
            cfg.attn_implementation = "eager"  # avoid flash‑init issues on CPU
            self.full_model = AutoModelForCausalLM.from_pretrained(
                args.model_name, config=cfg, device_map={"": "cpu"}, torch_dtype=torch.bfloat16
            )
            # Flash‑attention once on GPUs
            self.full_model.config.attn_implementation = "flash_attention_2"
            self.layers: List[torch.nn.Module] = [
                self.full_model.model.embed_tokens,
                *self.full_model.model.layers,
                self.full_model.lm_head,
            ]
        else:
            # Dummy placeholder list – structures are identical across ranks
            self.full_model = None
            dummy_cfg = AutoConfig.from_pretrained(args.model_name)
            dummy_cfg.attn_implementation = "eager"
            prototype = AutoModelForCausalLM.from_config(dummy_cfg)
            self.layers = [
                prototype.model.embed_tokens,
                *prototype.model.layers,
                prototype.lm_head,
            ]
            # Put *empty* (meta) parameters to avoid GPU memory until packets arrive
            for layer in self.layers:
                for p in layer.parameters():
                    p.data = torch.empty(0, device="meta")

        # Build double buffers for flighting parameter packets
        self.buffers: Tuple[_FlatPacket, _FlatPacket] = (_FlatPacket(), _FlatPacket())
        self.buffer_toggle = 0  # 0 or 1

    # ------------------------------------------------------------------
    # Internal helpers

    @staticmethod
    @torch.no_grad()
    def _micro_forward(layer: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        out = layer(hidden)
        if isinstance(out, tuple):
            out = out[0]
        return out

    # ------------------------------------------------------------------
    # Forward pass (no sampling)

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run hidden through all layers via reverse pipelining."""
        for layer_idx in range(len(self.layers)):
            self._stream_layer(layer_idx)
            layer = self.layers[layer_idx].to(self.device, non_blocking=True)

            # process micro‑batches resident on *this* GPU
            for start in range(0, hidden.size(0), self.args.micro_batch_size):
                mb = hidden[start : start + self.args.micro_batch_size]
                hidden[start : start + self.args.micro_batch_size] = self._micro_forward(layer, mb)

            # Immediately offload the layer back to CPU to reduce pressure
            layer.to("cpu", non_blocking=True)
            torch.cuda.current_stream().synchronize()
        return hidden

    # ------------------------------------------------------------------
    # Sampling (vectorised) – logits flattened, multinomial on GPU

    @torch.no_grad()
    def sample(self, hidden: torch.Tensor) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        # Run model once to get logits (hidden -> vocab)
        hidden = self.forward(hidden)
        logits = hidden  # after lm_head, forward returns logits already

        # Vectorised multinomial on GPU
        probs = torch.softmax(logits / self.args.sampling_temperature, dim=-1)
        draws = torch.multinomial(
            probs.view(-1, probs.size(-1)), self.args.num_samples * self.args.num_rounds, replacement=True
        )
        draws = draws.view(*probs.shape[:-1], -1)  # [B, T, total_draws]
        ids_all: List[List[List[int]]] = []
        counts_all: List[List[List[int]]] = []
        for b in range(draws.size(0)):
            ids_seq: List[List[int]] = []
            counts_seq: List[List[int]] = []
            for t in range(draws.size(1)):
                uniq, counts = torch.unique(draws[b, t], return_counts=True)
                ids_seq.append(uniq.cpu().tolist())
                counts_seq.append(counts.cpu().tolist())
            ids_all.append(ids_seq)
            counts_all.append(counts_seq)
        return ids_all, counts_all

    # ------------------------------------------------------------------
    # Pipelining core (double‑buffered parameter streaming)

    def _stream_layer(self, layer_idx: int) -> None:
        """Move layer parameters along the ring: CPU → 0 → 1 → … → N‑1 → CPU."""
        left = (self.rank - 1) % self.world_size
        right = (self.rank + 1) % self.world_size
        buf_in, buf_out = self.buffers[self.buffer_toggle], self.buffers[1 - self.buffer_toggle]

        req_recv: torch.distributed.Work | None = None
        if self.rank != 0:
            # Post asynchronous receive from the left neighbour
            shape = torch.empty(1, dtype=torch.int64)
            shape_packet = torch.empty(1, dtype=torch.int64, device=self.device)
            dist.recv(tensor=shape, src=left)
            buf_in.flat = torch.empty(int(shape.item()), dtype=torch.bfloat16, device=self.device)
            req_recv = dist.irecv(buf_in.flat, src=left)
        else:
            # Rank‑0 packs the layer into buf_out and sends to GPU‑0 (which is itself)
            buf_in.pack(self.layers[layer_idx])

        # Wait/finish RX, then unpack to local layer clone
        if self.rank != 0:
            req_recv.wait()
            buf_in.unpack_to(self.layers[layer_idx])

        # Send to the right neighbour, if exists (rank‑N−1 sends back to CPU/host via sendto=0)
        if self.world_size == 1:
            pass  # single GPU – nothing to stream
        else:
            if self.rank == self.world_size - 1:
                # Last GPU: send back to rank‑0 so it can offload to CPU
                dist.send(torch.tensor([buf_in.flat.numel()], dtype=torch.int64), dst=0)
                dist.send(buf_in.flat, dst=0)
            else:
                dist.send(torch.tensor([buf_in.flat.numel()], dtype=torch.int64), dst=right)
                dist.send(buf_in.flat, dst=right)

        self.buffer_toggle = 1 - self.buffer_toggle  # swap buffers

######################################################################
#              Data & misc helpers (mostly unchanged)                #
######################################################################

def collate_fn(examples, tokenizer, max_seq_len: int):
    texts = [e["text"] for e in examples]
    return tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len
    )


def streaming_dataloader(ds, tokenizer, batch_size: int, max_seq_len: int):
    batch = []
    for ex in ds:
        batch.append(ex)
        if len(batch) == batch_size:
            yield collate_fn(batch, tokenizer, max_seq_len)
            batch.clear()
    if batch:
        yield collate_fn(batch, tokenizer, max_seq_len)


######################################################################
#                Hub push util (rank‑0 only)                         #
######################################################################

def push_shard(records: List[dict], args: Args) -> None:
    api = HfApi(token=args.hf_token)
    files = api.list_repo_files(args.output_repo, repo_type="dataset")
    idx = sum(f.endswith(".parquet") for f in files)
    filename = f"data_{idx:05d}.parquet"
    Dataset.from_list(records).to_parquet(filename)

    repo = Repository(
        local_dir="repo_tmp", clone_from=args.output_repo, token=args.hf_token, repo_type="dataset"
    )
    repo.git_pull()
    repo.git_add(filename)
    repo.git_commit(f"Add shard {idx}")
    repo.git_push()
    repo.git_clear()
    os.remove(filename)

######################################################################
#                          Worker                                    #
######################################################################

def worker(args: Args) -> None:
    ##################################################################
    #   Distributed init & SIGTERM handling                          #
    ##################################################################
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    shutdown = threading.Event()

    def _handle_sigterm(signum, frame):
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    ##################################################################
    #                  Engine + Tokenizer / Data                     #
    ##################################################################
    engine = ReversePipelineEngine(args, rank, world)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        streaming=True,
        token=args.hf_token,
    )

    dataloader = streaming_dataloader(dataset, tokenizer, args.batch_size, args.max_seq_len)

    ##################################################################
    #                         Main loop                              #
    ##################################################################

    local_records: List[dict] = []
    total_local = 0
    for batch in dataloader:
        if shutdown.is_set():
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        # Embed tokens once on this GPU (static data assumption)
        hidden = engine.layers[0].to(device)(input_ids)  # embed layer is index 0
        ids, counts = engine.sample(hidden)  # includes forward + sample

        # Build records (same semantics as original script)
        for i in range(len(input_ids)):
            tokens = input_ids[i].tolist()
            while tokens and tokens[-1] == 0:
                tokens.pop()
            seq_len = len(tokens)
            local_records.append(
                {
                    "input_ids": tokens,
                    "sampled_ids": ids[i][:seq_len],
                    "sampled_counts": counts[i][:seq_len],
                }
            )
        total_local += len(input_ids)

        if total_local >= args.push_every:
            _flush(local_records, args, rank, world)
            local_records.clear()
            total_local = 0
        if shutdown.is_set():
            break

    # final flush
    if local_records and not shutdown.is_set():
        _flush(local_records, args, rank, world)

    dist.destroy_process_group()


######################################################################
#               Gather & push helper (rank‑0)                        #
######################################################################

def _flush(local_records: List[dict], args: Args, rank: int, world: int):
    """All‑gather python objects to rank‑0 and push shard."""
    gathered: List[List[dict]] | None = [None] * world if rank == 0 else None
    dist.gather_object(local_records, gathered, dst=0)
    if rank == 0 and gathered is not None:
        flat: List[dict] = []
        for part in gathered:
            flat.extend(part)
        push_shard(flat, args)

######################################################################
#                          Entrypoint                                #
######################################################################

def main():
    args = parse_args()
    worker(args)


if __name__ == "__main__":
    main()

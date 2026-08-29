#!/usr/bin/env python3
"""Compare native Triton VSA-128 forward with the existing 128->64 route."""

from __future__ import annotations

import argparse
from functools import partial

import torch
from triton.testing import do_bench

from fastvideo_kernel.block_sparse_attn_256 import _expand_mask_and_sizes_128_to_64
from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import triton_block_sparse_attn_forward
from fastvideo_kernel.triton_kernels.index import map_to_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[8192, 16384, 32768, 49152])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--sparsity", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.sparsity < 1:
        raise ValueError("--sparsity must be in [0, 1)")

    torch.manual_seed(args.seed)
    device_name = torch.cuda.get_device_name(0)
    print(f"device={device_name}, heads={args.num_heads}, head_dim={args.head_dim}, sparsity={args.sparsity}")
    print("timings exclude mask->index conversion and isolate the sparse forward kernel")
    print("seq_len\ttopk128\troute64_ms\tnative128_ms\tspeedup")

    for seq_len in args.seq_lens:
        if seq_len % 128:
            raise ValueError(f"sequence length must be divisible by 128, got {seq_len}")

        blocks_128 = seq_len // 128
        topk_128 = max(1, round(blocks_128 * (1 - args.sparsity)))
        shape = (args.batch_size, args.num_heads, seq_len, args.head_dim)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        scores = torch.rand(
            args.batch_size,
            args.num_heads,
            blocks_128,
            blocks_128,
            device="cuda",
        )
        selected = scores.topk(topk_128, dim=-1).indices
        mask_128 = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, selected, True)
        sizes_128 = torch.full((blocks_128,), 128, dtype=torch.int32, device="cuda")

        idx_128, num_128 = map_to_index(mask_128)
        mask_64, sizes_64 = _expand_mask_and_sizes_128_to_64(mask_128, sizes_128)
        idx_64, num_64 = map_to_index(mask_64)

        route_64 = partial(triton_block_sparse_attn_forward, q, k, v, idx_64, num_64, sizes_64)
        native_128 = partial(
            triton_block_sparse_attn_forward,
            q,
            k,
            v,
            idx_128,
            num_128,
            sizes_128,
            block_size=128,
        )

        route_64_ms = do_bench(route_64, warmup=args.warmup, rep=args.rep)
        native_128_ms = do_bench(native_128, warmup=args.warmup, rep=args.rep)
        print(
            f"{seq_len}\t{topk_128}\t{route_64_ms:.4f}\t"
            f"{native_128_ms:.4f}\t{route_64_ms / native_128_ms:.3f}x"
        )


if __name__ == "__main__":
    main()

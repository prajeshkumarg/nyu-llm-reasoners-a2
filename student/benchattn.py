import argparse
import csv
import math
import os
import timeit
from itertools import product

import torch
import torch.nn.functional as F
from a1_basics.model import scaled_dot_product_attention

D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
BATCH_SIZE = 8
NUM_STEPS = 100
WARMUP_STEPS = 10


def attention(Q, K, V):
    """Naive scaled dot-product attention (no mask, no heads)."""
    return scaled_dot_product_attention(Q, K, V)

def benchmark_attention(d_model, seq_len, device="cuda"):
    use_cuda = device == "cuda" and torch.cuda.is_available()

    # ── inputs ────────────────────────────────────────────────────────────────
    # shape: [batch, seq_len, d_model] — no head dimension
    def make_inputs(requires_grad=True):
        return [
            torch.randn(BATCH_SIZE, seq_len, d_model,
                       device=device, requires_grad=requires_grad)
            for _ in range(3)
        ]

    # ── warmup ────────────────────────────────────────────────────────────────
    try:
        for _ in range(WARMUP_STEPS):
            Q, K, V = make_inputs()
            out = attention(Q, K, V)
            loss = out.sum()
            loss.backward()
            if use_cuda:
                torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return None

    # ── forward timing ────────────────────────────────────────────────────────
    fwd_times = []
    try:
        for _ in range(NUM_STEPS):
            Q, K, V = make_inputs()
            if use_cuda:
                torch.cuda.synchronize()
            start = timeit.default_timer()
            out = attention(Q, K, V)
            if use_cuda:
                torch.cuda.synchronize()
            fwd_times.append(timeit.default_timer() - start)
    except torch.cuda.OutOfMemoryError:
        return {"d_model": d_model, "seq_len": seq_len,
                "fwd_ms": "OOM", "pre_bwd_mem_mb": "OOM",
                "bwd_ms": "OOM"}

    fwd_avg = sum(fwd_times) / len(fwd_times) * 1000

    # ── measure memory before backward ────────────────────────────────────────
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    Q, K, V = make_inputs()
    out = attention(Q, K, V)
    loss = out.sum()
    if use_cuda:
        torch.cuda.synchronize()
    pre_bwd_mem = torch.cuda.memory_allocated(device) / (1024 ** 2) if use_cuda else 0.0

    # ── backward timing ───────────────────────────────────────────────────────
    bwd_times = []
    try:
        for _ in range(NUM_STEPS):
            Q, K, V = make_inputs()
            out = attention(Q, K, V)
            loss = out.sum()
            if use_cuda:
                torch.cuda.synchronize()
            start = timeit.default_timer()
            loss.backward()
            if use_cuda:
                torch.cuda.synchronize()
            bwd_times.append(timeit.default_timer() - start)
    except torch.cuda.OutOfMemoryError:
        return {"d_model": d_model, "seq_len": seq_len,
                "fwd_ms": round(fwd_avg, 3),
                "pre_bwd_mem_mb": round(pre_bwd_mem, 1),
                "bwd_ms": "OOM"}

    bwd_avg = sum(bwd_times) / len(bwd_times) * 1000

    return {
        "d_model":        d_model,
        "seq_len":        seq_len,
        "fwd_ms":         round(fwd_avg, 3),
        "pre_bwd_mem_mb": round(pre_bwd_mem, 1),
        "bwd_ms":         round(bwd_avg, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/attention_bench.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = []
    for d_model, seq_len in product(D_MODELS, SEQ_LENS):
        print(f"d_model={d_model:4d}  seq_len={seq_len:6d} ... ", end="", flush=True)
        row = benchmark_attention(d_model, seq_len, args.device)
        if row is None:
            row = {"d_model": d_model, "seq_len": seq_len,
                   "fwd_ms": "OOM", "pre_bwd_mem_mb": "OOM", "bwd_ms": "OOM"}
        results.append(row)
        print(f"fwd={row['fwd_ms']} ms  mem={row['pre_bwd_mem_mb']} MB  bwd={row['bwd_ms']} ms")

    # ── print summary table ───────────────────────────────────────────────────
    print(f"\n{'d_model':>8} {'seq_len':>8} {'fwd_ms':>10} {'mem_MB':>10} {'bwd_ms':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['d_model']:>8} {r['seq_len']:>8} "
              f"{str(r['fwd_ms']):>10} {str(r['pre_bwd_mem_mb']):>10} "
              f"{str(r['bwd_ms']):>10}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
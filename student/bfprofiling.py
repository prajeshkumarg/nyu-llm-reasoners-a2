"""Mixed-precision benchmarking script for BasicsTransformerLM.

Times forward and backward passes with and without BF16 autocast.
Can be run for a single (model, precision) combination (used by SLURM
sweep) or sweep all sizes internally.

Usage
-----
# Single run — called by SLURM sweep:
    uv run python student/mixed_precision_bench.py --model-size small
    uv run python student/mixed_precision_bench.py --model-size small --mixed-precision
    uv run python student/mixed_precision_bench.py --model-size small --mixed-precision --forward-only

# Internal sweep of all sizes (both precisions):
    uv run python student/mixed_precision_bench.py --sweep
    uv run python student/mixed_precision_bench.py --sweep --output results/bench_mixed_precision.csv
"""

import argparse
import csv
import math
import os
import timeit
from contextlib import nullcontext

import torch
from a1_basics.model import BasicsTransformerLM

MODEL_SIZES = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def benchmark(
    model_size: str,
    vocab_size: int = 10_000,
    context_length: int = 128,
    batch_size: int = 4,
    warmup_steps: int = 5,
    num_steps: int = 10,
    backward: bool = True,
    mixed_precision: bool = False,
    device: str = "cuda",
) -> dict | None:
    """Run benchmark for one (model_size, precision) combination.
    Returns a results dict, or None if OOM.
    """
    use_cuda = device == "cuda" and torch.cuda.is_available()
    preset = MODEL_SIZES[model_size]
    precision_label = "bf16" if mixed_precision else "fp32"
    mode_label = "forward-backward" if backward else "forward-only"

    print(f"\n{'='*60}")
    print(f"  model={model_size}  precision={precision_label}  mode={mode_label}")
    print(f"{'='*60}")

    # ── Build model ────────────────────────────────────────────────────────────
    try:
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=preset["d_model"],
            num_layers=preset["num_layers"],
            num_heads=preset["num_heads"],
            d_ff=preset["d_ff"],
            rope_theta=10_000.0,
        ).to(device)
    except torch.cuda.OutOfMemoryError:
        print("  OOM building model — skipping.")
        return None

    model.train() if backward else model.eval()
    print(f"  Parameters: {model.get_num_params():,}")

    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    # ── Autocast context ───────────────────────────────────────────────────────
    # nullcontext() is a no-op so step() is identical for both precisions
    # without branching inside the timed region.
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if mixed_precision
        else nullcontext()
    )

    def step():
        with autocast_ctx:
            logits = model(x)
            if backward:
                loss = logits.sum()
                loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

    # ── Warm-up ────────────────────────────────────────────────────────────────
    try:
        for _ in range(warmup_steps):
            step()
            model.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError:
        print("  OOM during warmup — skipping.")
        return None

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # ── Timed runs ─────────────────────────────────────────────────────────────
    times = []
    try:
        for _ in range(num_steps):
            if use_cuda:
                torch.cuda.synchronize()
            start = timeit.default_timer()
            step()
            end = timeit.default_timer()
            times.append(end - start)
            model.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError:
        print("  OOM during timed runs — skipping.")
        return None

    # ── Stats ──────────────────────────────────────────────────────────────────
    avg_ms  = sum(times) / len(times) * 1000
    std_ms  = math.sqrt(sum((t * 1000 - avg_ms) ** 2 for t in times) / len(times))
    min_ms  = min(times) * 1000
    max_ms  = max(times) * 1000
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if use_cuda else 0.0

    # Same print format as basicprofiling.py so SLURM regex parsing works
    print(f"Average step time: {avg_ms:.2f} ms  (std: {std_ms:.2f} ms)")
    print(f"Min step time:     {min_ms:.2f} ms")
    print(f"Max step time:     {max_ms:.2f} ms")
    print(f"Peak GPU memory:   {peak_mb:.2f} MB")

    return {
        "model":     model_size,
        "precision": precision_label,
        "mode":      mode_label,
        "avg_ms":    round(avg_ms,  2),
        "std_ms":    round(std_ms,  2),
        "min_ms":    round(min_ms,  2),
        "max_ms":    round(max_ms,  2),
        "peak_mb":   round(peak_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Mixed-precision benchmarking for BasicsTransformerLM"
    )
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_SIZES.keys()),
        default=None,
        help="Single model size (required unless --sweep is set)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use BF16 autocast (default: fp32)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep all model sizes x both precisions internally",
    )
    parser.add_argument("--vocab-size",     type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size",     type=int, default=4)
    parser.add_argument("--warmup-steps",   type=int, default=5)
    parser.add_argument("--num-steps",      type=int, default=10)
    parser.add_argument("--forward-only",   action="store_true")
    parser.add_argument("--device",         type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV output path (only used with --sweep)",
    )
    args = parser.parse_args()

    common = dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        backward=not args.forward_only,
        device=args.device,
    )

    # ── Single run mode (called by SLURM sweep script) ─────────────────────────
    if not args.sweep:
        if args.model_size is None:
            parser.error("--model-size is required unless --sweep is set")
        benchmark(
            model_size=args.model_size,
            mixed_precision=args.mixed_precision,
            **common,
        )
        return

    # ── Internal sweep mode ────────────────────────────────────────────────────
    results = []
    for model_size in MODEL_SIZES:
        for mixed_precision in [False, True]:
            row = benchmark(
                model_size=model_size,
                mixed_precision=mixed_precision,
                **common,
            )
            if row is not None:
                results.append(row)

    # Print summary with speedup column
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    fp32_avg = {
        (r["model"], r["mode"]): r["avg_ms"]
        for r in results if r["precision"] == "fp32"
    }
    header = f"{'Model':<8} {'Prec':<6} {'Mode':<18} {'Avg(ms)':>8} {'Speedup':>8} {'Peak MB':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        key = (r["model"], r["mode"])
        speedup = (
            f"{fp32_avg[key] / r['avg_ms']:.2f}x"
            if key in fp32_avg and r["precision"] == "bf16"
            else "baseline"
        )
        print(
            f"{r['model']:<8} {r['precision']:<6} {r['mode']:<18} "
            f"{r['avg_ms']:>8.2f} {speedup:>8} {r['peak_mb']:>9.1f}"
        )

    if args.output and results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
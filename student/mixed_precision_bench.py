"""Mixed-precision benchmarking script for BasicsTransformerLM."""

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
    model_size, vocab_size=10_000, context_length=128, batch_size=4,
    warmup_steps=5, num_steps=10, backward=True, mixed_precision=False,
    compiled=False, device="cuda",
):
    use_cuda = device == "cuda" and torch.cuda.is_available()
    preset = MODEL_SIZES[model_size]
    precision_label = "bf16" if mixed_precision else "fp32"
    compile_label   = "compiled" if compiled else "eager"
    mode_label      = "forward-backward" if backward else "forward-only"

    print(f"\n{'='*60}")
    print(f"  model={model_size}  precision={precision_label}  "
          f"compile={compile_label}  mode={mode_label}")
    print(f"{'='*60}")

    try:
        model = BasicsTransformerLM(
            vocab_size=vocab_size, context_length=context_length,
            d_model=preset["d_model"], num_layers=preset["num_layers"],
            num_heads=preset["num_heads"], d_ff=preset["d_ff"],
            rope_theta=10_000.0,
        ).to(device)
    except torch.cuda.OutOfMemoryError:
        print("  OOM building model — skipping.")
        return None

    model.train() if backward else model.eval()
    print(f"  Parameters: {model.get_num_params():,}")

    # ── torch.compile ──────────────────────────────────────────────────────────
    # Applied after .to(device). Warmup absorbs Triton JIT compilation cost.
    if compiled:
        model = torch.compile(model)

    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    autocast_ctx = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if mixed_precision else nullcontext()
    )

    def step():
        with autocast_ctx:
            logits = model(x)
            if backward:
                loss = logits.sum()
                loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

    # more warmup for compiled to absorb Triton compilation
    effective_warmup = warmup_steps * 3 if compiled else warmup_steps
    try:
        for _ in range(effective_warmup):
            step()
            model.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError:
        print("  OOM during warmup — skipping.")
        return None

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    try:
        for _ in range(num_steps):
            if use_cuda:
                torch.cuda.synchronize()
            start = timeit.default_timer()
            step()
            times.append(timeit.default_timer() - start)
            model.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError:
        print("  OOM during timed runs — skipping.")
        return None

    avg_ms  = sum(times) / len(times) * 1000
    std_ms  = math.sqrt(sum((t * 1000 - avg_ms) ** 2 for t in times) / len(times))
    min_ms  = min(times) * 1000
    max_ms  = max(times) * 1000
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if use_cuda else 0.0

    print(f"Average step time: {avg_ms:.2f} ms  (std: {std_ms:.2f} ms)")
    print(f"Min step time:     {min_ms:.2f} ms")
    print(f"Max step time:     {max_ms:.2f} ms")
    print(f"Peak GPU memory:   {peak_mb:.2f} MB")

    return {
        "model": model_size, "precision": precision_label,
        "compile": compile_label, "mode": mode_label,
        "avg_ms": round(avg_ms, 2), "std_ms": round(std_ms, 2),
        "min_ms": round(min_ms, 2), "max_ms": round(max_ms, 2),
        "peak_mb": round(peak_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=list(MODEL_SIZES.keys()), default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compiled",
                        help="Wrap model with torch.compile before benchmarking")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--vocab-size",     type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size",     type=int, default=4)
    parser.add_argument("--warmup-steps",   type=int, default=5)
    parser.add_argument("--num-steps",      type=int, default=10)
    parser.add_argument("--forward-only",   action="store_true")
    parser.add_argument("--device",         type=str, default="cuda")
    parser.add_argument("--output",         type=str, default=None)
    args = parser.parse_args()

    common = dict(
        vocab_size=args.vocab_size, context_length=args.context_length,
        batch_size=args.batch_size, warmup_steps=args.warmup_steps,
        num_steps=args.num_steps, backward=not args.forward_only,
        device=args.device,
    )

    if not args.sweep:
        if args.model_size is None:
            parser.error("--model-size is required unless --sweep is set")
        benchmark(model_size=args.model_size, mixed_precision=args.mixed_precision,
                  compiled=args.compiled, **common)
        return

    # ── Sweep: eager fp32, eager bf16, compiled fp32, compiled bf16 ───────────
    results = []
    for model_size in MODEL_SIZES:
        for compiled in [False, True]:
            for mixed_precision in [False, True]:
                row = benchmark(model_size=model_size, mixed_precision=mixed_precision,
                                compiled=compiled, **common)
                if row is not None:
                    results.append(row)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    fp32_eager_avg = {
        (r["model"], r["mode"]): r["avg_ms"]
        for r in results if r["precision"] == "fp32" and r["compile"] == "eager"
    }
    header = (f"{'Model':<8} {'Prec':<6} {'Compile':<10} {'Mode':<18} "
              f"{'Avg(ms)':>8} {'Speedup':>8} {'Peak MB':>9}")
    print(header)
    print("-" * len(header))
    for r in results:
        key = (r["model"], r["mode"])
        baseline = fp32_eager_avg.get(key)
        is_baseline = r["precision"] == "fp32" and r["compile"] == "eager"
        speedup = "baseline" if is_baseline else (
            f"{baseline / r['avg_ms']:.2f}x" if baseline else "N/A"
        )
        print(f"{r['model']:<8} {r['precision']:<6} {r['compile']:<10} "
              f"{r['mode']:<18} {r['avg_ms']:>8.2f} {speedup:>8} {r['peak_mb']:>9.1f}")

    if args.output and results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
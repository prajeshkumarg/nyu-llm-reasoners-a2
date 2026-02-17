"""
Script 3: Fused vs Non-Fused Kernel Comparison
From Percy Liang's Lecture 6

Demonstrates why kernel fusion matters for performance using GeLU as an example.
Key analogy: warehouse (DRAM) : factory (SRAM) - minimize round trips!
"""

import time
from typing import Callable
import torch
from torch.profiler import ProfilerActivity


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def pytorch_gelu(x: torch.Tensor) -> torch.Tensor:
    """PyTorch's fused GeLU implementation (tanh approximation)."""
    return torch.nn.functional.gelu(x, approximate="tanh")


def manual_gelu(x: torch.Tensor) -> torch.Tensor:
    """Manual GeLU implementation (NOT fused - each operation is a separate kernel)."""
    # Each of these operations triggers a separate CUDA kernel:
    # 1. x * x * x (2 multiplications)
    # 2. 0.044715 * x^3
    # 3. x + (0.044715 * x^3)
    # 4. 0.79788456 * (...)
    # 5. torch.tanh(...)
    # 6. 1 + tanh(...)
    # 7. x * (1 + tanh(...))
    # 8. 0.5 * (...)
    # Each operation reads from DRAM and writes back to DRAM!
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


def run_operation1(dim: int, operation: Callable) -> Callable:
    """Setup one random dim x dim matrix and return a function to run the operation."""
    x = torch.randn(dim, dim, device=get_device())
    return lambda: operation(x)


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """Benchmark a function and return mean time in ms."""
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for trial in range(num_trials):
        start_time = time.time()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)

    mean_time = mean(times)
    print(f"{description}: {mean_time:.2f} ms")
    return mean_time


def profile(description: str, run: Callable, num_warmups: int = 1) -> str:
    """Profile a function and return the profiler table."""
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        max_name_column_width=80,
        row_limit=10
    )
    return table


def check_equal(func1: Callable, func2: Callable, rtol: float = 1e-4, atol: float = 1e-5):
    """Check that two functions produce the same output."""
    x = torch.randn(1000, device=get_device())
    y1 = func1(x)
    y2 = func2(x)
    if torch.allclose(y1, y2, rtol=rtol, atol=atol):
        print("  Implementations match!")
    else:
        max_diff = (y1 - y2).abs().max().item()
        print(f"  WARNING: Max difference = {max_diff}")


def main():
    print("Kernel Fusion: Fused vs Non-Fused Comparison")
    print("=" * 80)
    print()
    print("KEY CONCEPT: Kernel Fusion")
    print("-" * 80)
    print("Analogy from Horace He's blog (https://horace.io/brrr_intro.html):")
    print("  - DRAM = warehouse (big, far away)")
    print("  - SRAM = factory floor (small, close)")
    print()
    print("Without fusion: each operation is a separate trip to the warehouse")
    print("  [Read from DRAM] -> [Compute] -> [Write to DRAM]")
    print("  [Read from DRAM] -> [Compute] -> [Write to DRAM]")
    print("  [Read from DRAM] -> [Compute] -> [Write to DRAM]")
    print()
    print("With fusion: combine operations, only one trip")
    print("  [Read from DRAM] -> [Compute A, B, C] -> [Write to DRAM]")
    print()

    if not torch.cuda.is_available():
        print("WARNING: Running on CPU. Fusion benefits are less visible.")

    # Verify correctness
    print("\n" + "=" * 80)
    print("VERIFYING CORRECTNESS")
    print("=" * 80)
    x = torch.tensor([1.0, -1.0, 0.0, 2.0], device=get_device())
    y_pytorch = pytorch_gelu(x)
    y_manual = manual_gelu(x)
    print(f"Input:        {x.tolist()}")
    print(f"PyTorch GeLU: {y_pytorch.tolist()}")
    print(f"Manual GeLU:  {y_manual.tolist()}")
    check_equal(pytorch_gelu, manual_gelu)

    # Benchmark comparison
    print("\n" + "=" * 80)
    print("BENCHMARKING (dim=16384)")
    print("=" * 80)
    dim = 16384
    manual_time = benchmark("manual_gelu (NOT fused)", run_operation1(dim=dim, operation=manual_gelu))
    pytorch_time = benchmark("pytorch_gelu (fused)", run_operation1(dim=dim, operation=pytorch_gelu))

    if manual_time > 0 and pytorch_time > 0:
        speedup = manual_time / pytorch_time
        print(f"\nSpeedup from fusion: {speedup:.2f}x")

    # Profile to see the difference
    print("\n" + "=" * 80)
    print("PROFILING - manual_gelu (NOT fused)")
    print("=" * 80)
    print("Notice: MULTIPLE kernels are launched!")
    manual_profile = profile("manual_gelu", run_operation1(dim=dim, operation=manual_gelu))
    print(manual_profile)

    print("\n" + "=" * 80)
    print("PROFILING - pytorch_gelu (fused)")
    print("=" * 80)
    print("Notice: Only ONE kernel is launched!")
    pytorch_profile = profile("pytorch_gelu", run_operation1(dim=dim, operation=pytorch_gelu))
    print(pytorch_profile)

    # Summary
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("1. Manual GeLU launches MULTIPLE kernels (one per operation)")
    print("   - Each kernel reads from and writes to global memory (DRAM)")
    print("   - Memory bandwidth becomes the bottleneck")
    print()
    print("2. PyTorch's fused GeLU launches ONE kernel")
    print("   - All computation happens in registers/shared memory")
    print("   - Only ONE read and ONE write to DRAM")
    print()
    print("3. The speedup from fusion can be SIGNIFICANT (2-10x)")
    print("   - GeLU is memory-bound, not compute-bound")
    print("   - Reducing memory traffic is the key optimization")
    print()
    print("4. This principle applies broadly:")
    print("   - Activation functions")
    print("   - Layer normalization")
    print("   - Attention computations (FlashAttention!)")
    print("=" * 80)


if __name__ == "__main__":
    main()

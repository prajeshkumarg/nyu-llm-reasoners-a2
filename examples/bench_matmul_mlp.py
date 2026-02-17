"""
Script 1: Benchmarking matrix multiplication and MLP
From Percy Liang's Lecture 6

Demonstrates how to benchmark PyTorch operations and understand scaling behavior.
"""

import time
from typing import Callable
import torch
import torch.nn as nn


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    """Setup and return a function that runs the MLP forward/backward."""
    model = MLP(dim, num_layers).to(get_device())
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        for step in range(num_steps):
            y = model(x).mean()
            y.backward()
    return run


def run_operation2(dim: int, operation: Callable) -> Callable:
    """Setup two random dim x dim matrices and return a function to run the operation."""
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    return lambda: operation(x, y)


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """Benchmark `run` by running it `num_trials` times, return mean time in ms."""
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time it
    times: list[float] = []
    for trial in range(num_trials):
        start_time = time.time()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)

    mean_time = mean(times)
    print(f"{description}: {mean_time:.2f} ms (over {num_trials} trials)")
    return mean_time


def print_gpu_specs():
    """Print GPU specifications if available."""
    num_devices = torch.cuda.device_count()
    print(f"\n{num_devices} CUDA device(s) available")
    for i in range(num_devices):
        properties = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {properties.name}")
        print(f"    Total memory: {properties.total_memory / 1e9:.2f} GB")
        print(f"    Multiprocessors: {properties.multi_processor_count}")


def benchmark_matmul():
    """Benchmark matrix multiplication at different sizes."""
    print("\n" + "=" * 60)
    print("BENCHMARKING MATRIX MULTIPLICATION")
    print("=" * 60)

    if torch.cuda.is_available():
        dims = (1024, 2048, 4096, 8192, 16384)
    else:
        dims = (1024, 2048)

    results = []
    for dim in dims:
        result = benchmark(
            f"matmul(dim={dim})",
            run_operation2(dim=dim, operation=lambda a, b: a @ b)
        )
        results.append((dim, result))

    print("\nSummary:")
    for dim, time_ms in results:
        # Compute TFLOPS: 2*N^3 FLOPs for NxN @ NxN matmul
        flops = 2 * dim ** 3
        tflops = flops / (time_ms / 1000) / 1e12
        print(f"  dim={dim}: {time_ms:.2f} ms, {tflops:.2f} TFLOPS")


def benchmark_mlp_scaling():
    """Benchmark MLP with different scaling factors."""
    print("\n" + "=" * 60)
    print("BENCHMARKING MLP SCALING")
    print("=" * 60)

    # Base configuration
    dim = 256
    num_layers = 4
    batch_size = 256
    num_steps = 2

    print(f"\nBase config: dim={dim}, layers={num_layers}, batch={batch_size}, steps={num_steps}")
    base_time = benchmark(
        "run_mlp (base)",
        run_mlp(dim=dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps)
    )

    # Scale number of steps
    print("\n--- Scaling number of steps ---")
    for scale in (2, 3, 4, 5):
        result = benchmark(
            f"run_mlp({scale}x num_steps)",
            run_mlp(dim=dim, num_layers=num_layers, batch_size=batch_size, num_steps=scale * num_steps)
        )
        print(f"  Expected ~{scale}x, actual ~{result/base_time:.2f}x")

    # Scale number of layers
    print("\n--- Scaling number of layers ---")
    for scale in (2, 3, 4, 5):
        result = benchmark(
            f"run_mlp({scale}x num_layers)",
            run_mlp(dim=dim, num_layers=scale * num_layers, batch_size=batch_size, num_steps=num_steps)
        )
        print(f"  Expected ~{scale}x, actual ~{result/base_time:.2f}x")

    # Scale batch size
    print("\n--- Scaling batch size ---")
    for scale in (2, 3, 4, 5):
        result = benchmark(
            f"run_mlp({scale}x batch_size)",
            run_mlp(dim=dim, num_layers=num_layers, batch_size=scale * batch_size, num_steps=num_steps)
        )
        print(f"  Expected ~{scale}x, actual ~{result/base_time:.2f}x")

    # Scale dimension
    print("\n--- Scaling dimension ---")
    for scale in (2, 3, 4, 5):
        result = benchmark(
            f"run_mlp({scale}x dim)",
            run_mlp(dim=scale * dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps)
        )
        print(f"  Expected ~{scale**2}x (quadratic), actual ~{result/base_time:.2f}x")


def main():
    print("Benchmarking Matrix Multiplication and MLP")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: Running on CPU. For full experience, use a GPU.")

    print_gpu_specs()

    # Quick demo of benchmarking
    print("\n--- Sanity check: benchmarking sleep ---")
    benchmark("sleep(50ms)", lambda: time.sleep(50 / 1000))

    benchmark_matmul()
    benchmark_mlp_scaling()

    print("\n" + "=" * 60)
    print("OBSERVATIONS:")
    print("- Scaling steps/layers: should scale ~linearly")
    print("- Scaling batch size: may not scale linearly (better parallelism)")
    print("- Scaling dimension: should scale ~quadratically for linear layers")
    print("- Actual timings vary due to CUDA kernel selection, caching, etc.")
    print("=" * 60)


if __name__ == "__main__":
    main()

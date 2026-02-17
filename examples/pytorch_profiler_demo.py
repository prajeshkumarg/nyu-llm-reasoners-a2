"""
Script 2: PyTorch Profiler Examples
From Percy Liang's Lecture 6

Demonstrates how to use PyTorch's built-in profiler to understand
where time is spent and what CUDA kernels are being called.
"""

import time
import os
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def run_operation1(dim: int, operation: Callable) -> Callable:
    """Setup one random dim x dim matrix and return a function to run the operation."""
    x = torch.randn(dim, dim, device=get_device())
    return lambda: operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    """Setup two random dim x dim matrices and return a function to run the operation."""
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    return lambda: operation(x, y)


def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False) -> str:
    """Profile a function and return the profiler table."""
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Run with profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=with_stack,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Get table
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        max_name_column_width=80,
        row_limit=10
    )

    # Optionally export stack trace for flame graph visualization
    if with_stack:
        os.makedirs("var", exist_ok=True)
        text_path = f"var/stacks_{description}.txt"
        try:
            prof.export_stacks(text_path, "self_cuda_time_total")
            print(f"Stack trace exported to: {text_path}")
        except Exception as e:
            print(f"Could not export stack trace: {e}")

    return table


def profile_basic_operations():
    """Profile basic PyTorch operations."""
    print("\n" + "=" * 80)
    print("PROFILING BASIC OPERATIONS")
    print("=" * 80)

    # Profile sleep (baseline)
    print("\n--- sleep (50ms) ---")
    sleep_profile = profile("sleep", lambda: time.sleep(50 / 1000))
    print(sleep_profile)

    # Profile addition
    print("\n--- element-wise addition (dim=2048) ---")
    add_profile = profile("add", run_operation2(dim=2048, operation=lambda a, b: a + b))
    print(add_profile)

    # Profile matrix multiplication at different sizes
    print("\n--- matrix multiplication (dim=2048) ---")
    matmul_profile = profile("matmul_2048", run_operation2(dim=2048, operation=lambda a, b: a @ b))
    print(matmul_profile)

    print("\n--- matrix multiplication (dim=128) ---")
    matmul_profile_small = profile("matmul_128", run_operation2(dim=128, operation=lambda a, b: a @ b))
    print(matmul_profile_small)

    print("\nOBSERVATIONS:")
    print("- Different CUDA kernels are invoked depending on tensor dimensions")
    print("- Kernel names reveal implementation details (e.g., cutlass, tile sizes)")
    print("- Example: cutlass_80_simt_sgemm_256x128_8x4_nn_align1")
    print("  - cutlass: NVIDIA's CUDA library for linear algebra")
    print("  - 256x128: tile size used in the kernel")


def profile_composite_operations():
    """Profile composite PyTorch operations."""
    print("\n" + "=" * 80)
    print("PROFILING COMPOSITE OPERATIONS")
    print("=" * 80)

    # cdist (pairwise distances)
    print("\n--- cdist (pairwise distances, dim=2048) ---")
    cdist_profile = profile("cdist", run_operation2(dim=2048, operation=lambda a, b: torch.cdist(a, b)))
    print(cdist_profile)

    # GeLU
    print("\n--- gelu (dim=2048) ---")
    gelu_profile = profile("gelu", run_operation2(dim=2048, operation=lambda a, b: torch.nn.functional.gelu(a + b)))
    print(gelu_profile)

    # Softmax
    print("\n--- softmax (dim=2048) ---")
    softmax_profile = profile("softmax", run_operation2(dim=2048, operation=lambda a, b: torch.nn.functional.softmax(a + b, dim=-1)))
    print(softmax_profile)


def profile_mlp():
    """Profile the MLP model."""
    print("\n" + "=" * 80)
    print("PROFILING MLP")
    print("=" * 80)

    if torch.cuda.is_available():
        mlp_run = run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2)
    else:
        mlp_run = run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=2)

    print("\n--- MLP forward/backward ---")
    mlp_profile = profile("mlp", mlp_run, with_stack=True)
    print(mlp_profile)

    print("\nOBSERVATIONS:")
    print("- Forward and backward passes invoke different kernels")
    print("- Matrix multiplications dominate compute time")
    print("- GeLU activations are relatively fast")
    print("- Check var/stacks_mlp.txt for flame graph data")


def main():
    print("PyTorch Profiler Demo")
    print("=" * 80)
    print("While benchmarking measures end-to-end time, profiling shows WHERE time is spent.")
    print("This helps you understand what CUDA kernels are actually being called.")
    print()
    print("PyTorch profiler docs: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html")

    if not torch.cuda.is_available():
        print("\nWARNING: Running on CPU. CUDA profiling will show 0 times.")

    profile_basic_operations()
    profile_composite_operations()
    profile_mlp()

    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("- Profiling reveals which CUDA kernels are called")
    print("- Different input sizes may trigger different kernel implementations")
    print("- Composite operations may call multiple kernels under the hood")
    print("- Use with_stack=True to generate flame graph data")
    print("=" * 80)


if __name__ == "__main__":
    main()

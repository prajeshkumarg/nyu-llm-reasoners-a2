"""
Script 4: CUDA GeLU Kernel
From Percy Liang's Lecture 6

Demonstrates how to write a custom CUDA kernel for the GeLU activation function.
This shows what happens "under the hood" of a GPU computation.

REQUIRES: CUDA-capable GPU and PyTorch with CUDA support
"""

import time
import os
from typing import Callable, Optional
import torch
from torch.profiler import ProfilerActivity


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


# The CUDA kernel source code
CUDA_GELU_SRC = '''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel: each thread processes one element
__global__ void gelu_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            int num_elements) {
    // Compute global thread index
    // blockIdx.x = which block this thread is in
    // blockDim.x = number of threads per block
    // threadIdx.x = index of this thread within the block
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check (some threads may be out of range)
    if (i < num_elements) {
        // Read input value
        float xi = x[i];

        // Compute GeLU using tanh approximation:
        // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // sqrt(2/pi) ≈ 0.79788456
        float a = 0.79788456f * (xi + 0.044715f * xi * xi * xi);
        float tanh_a = tanhf(a);
        float yi = 0.5f * xi * (1.0f + tanh_a);

        // Write output value
        y[i] = yi;
    }
}

// C++ wrapper function called from Python
torch::Tensor gelu(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");

    // Allocate output tensor (same shape as input)
    auto y = torch::empty_like(x);

    // Compute grid dimensions
    int num_elements = x.numel();
    int threads_per_block = 256;  // Typical choice (multiple of 32 = warp size)
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel!
    // <<<num_blocks, threads_per_block>>> is CUDA's kernel launch syntax
    gelu_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        num_elements
    );

    return y;
}
'''

# C++ declaration for binding
CPP_GELU_SRC = "torch::Tensor gelu(torch::Tensor x);"


def pytorch_gelu(x: torch.Tensor) -> torch.Tensor:
    """PyTorch's fused GeLU implementation."""
    return torch.nn.functional.gelu(x, approximate="tanh")


def manual_gelu(x: torch.Tensor) -> torch.Tensor:
    """Manual GeLU implementation (NOT fused)."""
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


def create_cuda_gelu() -> Optional[Callable]:
    """Compile the CUDA kernel and return the gelu function."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA kernel compilation")
        return None

    # Set this so CUDA reports errors properly
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Create build directory
    build_dir = "var/cuda_gelu"
    os.makedirs(build_dir, exist_ok=True)

    print("Compiling CUDA kernel (this may take a moment)...")

    try:
        from torch.utils.cpp_extension import load_inline

        module = load_inline(
            cuda_sources=[CUDA_GELU_SRC],
            cpp_sources=[CPP_GELU_SRC],
            functions=["gelu"],
            extra_cflags=["-O2"],
            verbose=True,
            name="inline_gelu",
            build_directory=build_dir,
        )
        print("CUDA kernel compiled successfully!")
        return getattr(module, "gelu")

    except Exception as e:
        print(f"Failed to compile CUDA kernel: {e}")
        return None


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
    print("CUDA GeLU Kernel")
    print("=" * 80)
    print()
    print("CUDA KERNEL CONCEPTS:")
    print("-" * 80)
    print("CUDA is an extension of C/C++ with APIs for managing GPUs.")
    print()
    print("Execution Model:")
    print("  Grid -> Thread Blocks -> Threads")
    print()
    print("  - Grid: collection of thread blocks")
    print("  - Thread Block: collection of threads (scheduled on a single SM)")
    print("    - Threads in a block can use shared memory")
    print("  - Thread: single unit of execution")
    print()
    print("You write code that each thread executes, using:")
    print("  - blockIdx.x: which block this thread belongs to")
    print("  - blockDim.x: number of threads per block")
    print("  - threadIdx.x: index within the block")
    print("  - Global index = blockIdx.x * blockDim.x + threadIdx.x")
    print()

    if not torch.cuda.is_available():
        print("=" * 80)
        print("CUDA NOT AVAILABLE - Cannot run CUDA kernel examples")
        print("Please run this script on a machine with a CUDA-capable GPU")
        print("=" * 80)
        return

    # Show the CUDA source code
    print("=" * 80)
    print("CUDA KERNEL SOURCE CODE:")
    print("=" * 80)
    print(CUDA_GELU_SRC)

    # Compile the kernel
    print("\n" + "=" * 80)
    print("COMPILING CUDA KERNEL")
    print("=" * 80)
    cuda_gelu = create_cuda_gelu()

    if cuda_gelu is None:
        print("Failed to compile CUDA kernel")
        return

    # Verify correctness
    print("\n" + "=" * 80)
    print("VERIFYING CORRECTNESS")
    print("=" * 80)
    print("Checking cuda_gelu vs manual_gelu:")
    check_equal(cuda_gelu, manual_gelu)
    print("Checking cuda_gelu vs pytorch_gelu:")
    check_equal(cuda_gelu, pytorch_gelu)

    # Benchmark all implementations
    print("\n" + "=" * 80)
    print("BENCHMARKING (dim=16384)")
    print("=" * 80)
    dim = 16384

    manual_time = benchmark("manual_gelu (Python, NOT fused)", run_operation1(dim=dim, operation=manual_gelu))
    pytorch_time = benchmark("pytorch_gelu (PyTorch fused)", run_operation1(dim=dim, operation=pytorch_gelu))
    cuda_time = benchmark("cuda_gelu (our CUDA kernel)", run_operation1(dim=dim, operation=cuda_gelu))

    print("\nSpeedups:")
    if manual_time > 0:
        print(f"  cuda_gelu vs manual_gelu: {manual_time / cuda_time:.2f}x")
    if pytorch_time > 0:
        print(f"  pytorch_gelu vs cuda_gelu: {cuda_time / pytorch_time:.2f}x slower (PyTorch is more optimized)")

    # Profile CUDA kernel
    print("\n" + "=" * 80)
    print("PROFILING cuda_gelu")
    print("=" * 80)
    cuda_profile = profile("cuda_gelu", run_operation1(dim=dim, operation=cuda_gelu))
    print(cuda_profile)

    # Summary
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("1. Our CUDA kernel is FASTER than manual Python (it's fused!)")
    print()
    print("2. But PyTorch's implementation is STILL FASTER because:")
    print("   - Better memory access patterns (coalescing)")
    print("   - Vectorized loads/stores (loading 4 floats at once)")
    print("   - Better thread occupancy tuning")
    print()
    print("3. Elementwise operations are EASY in CUDA:")
    print("   - Each thread processes one element independently")
    print("   - No shared memory needed")
    print()
    print("4. More complex operations (matmul, softmax, attention) need:")
    print("   - Shared memory management")
    print("   - Thread synchronization")
    print("   - Tiling strategies")
    print("   - This is where Triton helps (coming in the next lecture)")
    print("=" * 80)


if __name__ == "__main__":
    main()

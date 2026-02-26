"""Basic end-to-end benchmarking of forward and backward passes for BasicsTransformerLM."""

import argparse
import math
import timeit

import torch
from a1_basics.model import BasicsTransformerLM

# Model size presets from Table 1 (§1.1.2)
MODEL_SIZES = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def benchmark(
    vocab_size: int = 10_000,
    context_length: int = 128,
    d_model: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    d_ff: int = 3072,
    rope_theta: float = 10_000.0,
    batch_size: int = 4,
    warmup_steps: int = 5,
    num_steps: int = 10,
    backward: bool = True,
    device: str = "cuda",
):
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if device == "cuda" and not use_cuda:
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # Initialize model with random weights
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    if backward:
        model.train()
    else:
        model.eval()

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Device: {device}")
    print(f"Mode: {'forward + backward' if backward else 'forward only'}")
    print(f"Batch size: {batch_size}, Context length: {context_length}")
    print(f"Warm-up steps: {warmup_steps}, Timed steps: {num_steps}")
    print()

    # Generate a random batch of token ids
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    def step():
        logits = model(x)
        if backward:
            loss = logits.sum()
            loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

    # Warm-up
    for _ in range(warmup_steps):
        step()
        model.zero_grad(set_to_none=True)

    # Timed runs
    times = []
    for _ in range(num_steps):
        start = timeit.default_timer()
        step()
        end = timeit.default_timer()
        times.append(end - start)
        model.zero_grad(set_to_none=True)

    avg_time = sum(times) / len(times)
    std_time = math.sqrt(sum((t - avg_time) ** 2 for t in times) / len(times))

    print(f"Average step time: {avg_time * 1000:.2f} ms  (std: {std_time * 1000:.2f} ms)")
    print(f"Min step time:     {min(times) * 1000:.2f} ms")
    print(f"Max step time:     {max(times) * 1000:.2f} ms")

    if use_cuda:
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"Peak GPU memory:   {peak_mem:.2f} MB")

    return times


def main():
    parser = argparse.ArgumentParser(description="Basic profiling of BasicsTransformerLM")

    parser.add_argument(
        "--model-size", type=str, choices=list(MODEL_SIZES.keys()), default=None,
        help="Named model size from Table 1 (overrides individual hyperparams)",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--forward-only", action="store_true", help="Only run forward pass")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    if args.model_size is not None:
        preset = MODEL_SIZES[args.model_size]
        d_model, d_ff = preset["d_model"], preset["d_ff"]
        num_layers, num_heads = preset["num_layers"], preset["num_heads"]
    else:
        d_model, d_ff = args.d_model, args.d_ff
        num_layers, num_heads = args.num_layers, args.num_heads

    benchmark(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        num_steps=args.num_steps,
        backward=not args.forward_only,
        device=args.device,
    )


if __name__ == "__main__":
    main()

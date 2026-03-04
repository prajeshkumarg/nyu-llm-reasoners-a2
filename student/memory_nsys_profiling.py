from __future__ import annotations

import argparse
import math
import os
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx

# ── project imports ───────────────────────────────────────────────────────────
import a1_basics.model as _model_mod
from a1_basics.model import BasicsTransformerLM
from a1_basics.optimizer import AdamW

# ── model size presets from Table 1 ──────────────────────────────────────────
MODEL_SIZES: dict[str, dict] = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


# ── NVTX helper ───────────────────────────────────────────────────────────────

class nvtx_range:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        nvtx.range_push(self.name)
        return self

    def __exit__(self, *_):
        nvtx.range_pop()


# ── SDPA monkey-patch ─────────────────────────────────────────────────────────

def _patch_sdpa() -> object:
    from einops import einsum as _einsum
    _orig_sdpa = _model_mod.scaled_dot_product_attention
    _softmax   = _model_mod.softmax

    def _annotated_sdpa(Q, K, V, mask=None):
        d_k = K.shape[-1]

        with nvtx_range("attn/qk_matmul"):
            attention_scores = (
                _einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
                / math.sqrt(d_k)
            )

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx_range("attn/softmax"):
            attention_weights = _softmax(attention_scores, dim=-1)

        with nvtx_range("attn/av_matmul"):
            return _einsum(
                attention_weights, V,
                "... query key, ... key d_v -> ... query d_v",
            )

    _model_mod.scaled_dot_product_attention = _annotated_sdpa
    return _orig_sdpa


def _unpatch_sdpa(original) -> None:
    _model_mod.scaled_dot_product_attention = original


# ── module NVTX hooks ─────────────────────────────────────────────────────────

def _register_nvtx_hooks(model: BasicsTransformerLM, per_layer: bool = True) -> list:
    handles: list = []

    def _hook_module(mod: torch.nn.Module, name: str) -> None:
        def _pre(m, inp):
            nvtx.range_push(name)
        def _post(m, inp, out):
            nvtx.range_pop()
        handles.append(mod.register_forward_pre_hook(_pre))
        handles.append(mod.register_forward_hook(_post))

    _hook_module(model.token_embeddings, "embedding")
    if per_layer:
        for i, layer in enumerate(model.layers):
            _hook_module(layer,      f"layer_{i}")
            _hook_module(layer.ln1,  f"layer_{i}/ln1")
            _hook_module(layer.attn, f"layer_{i}/attn")
            _hook_module(layer.ln2,  f"layer_{i}/ln2")
            _hook_module(layer.ffn,  f"layer_{i}/ffn")
    _hook_module(model.ln_final, "ln_final")
    _hook_module(model.lm_head,  "lm_head")
    return handles


# ── shared model + warmup builder ─────────────────────────────────────────────

def _build_and_warmup(
    model_size: str,
    vocab_size: int,
    context_length: int,
    batch_size: int,
    warmup_steps: int,
    forward_only: bool,
    mixed_precision: bool,   # ← ADDED
    device: str,
):
    """Build model + optimizer, run warmup. Returns (model, optimizer, x, autocast_ctx) or None on OOM."""
    use_cuda = device == "cuda" and torch.cuda.is_available()
    preset   = MODEL_SIZES[model_size]

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
        print(f"OOM building model={model_size} ctx={context_length} — skipping.")
        return None

    model.train()
    print(f"params (non-embedding): {model.get_num_params():,}")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    # ── autocast context: bf16 or no-op ──────────────────────────────────────
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if mixed_precision else nullcontext()
    )

    print(f"warming up ({warmup_steps} step(s))…")
    try:
        for _ in range(warmup_steps):
            with autocast_ctx:
                if forward_only:
                    with torch.no_grad():
                        _ = model(x)
                else:
                    logits = model(x)
                    logits.sum().backward()
            if not forward_only:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"OOM during warmup — skipping.")
        return None

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    return model, optimizer, x, autocast_ctx   # ← returns autocast_ctx too


# ── nsys profiling routine ────────────────────────────────────────────────────

def run_profile(
    model_size: str = "small",
    vocab_size: int = 10_000,
    context_length: int = 128,
    batch_size: int = 4,
    warmup_steps: int = 3,
    forward_only: bool = False,
    mixed_precision: bool = False,   # ← ADDED
    per_layer_hooks: bool = True,
    device: str = "cuda",
) -> None:
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if device == "cuda" and not use_cuda:
        print("WARNING: CUDA not available — falling back to CPU.")
        device = "cpu"

    precision = "bf16" if mixed_precision else "fp32"
    mode      = "forward-only" if forward_only else "forward+backward+optimizer"
    print(f"[nsys] model={model_size}  ctx={context_length}  precision={precision}  mode={mode}")

    result = _build_and_warmup(
        model_size, vocab_size, context_length, batch_size,
        warmup_steps, forward_only, mixed_precision, device
    )
    if result is None:
        return
    model, optimizer, x, autocast_ctx = result

    orig_sdpa = _patch_sdpa()
    handles   = _register_nvtx_hooks(model, per_layer=per_layer_hooks)

    if use_cuda:
        torch.cuda.cudart().cudaProfilerStart()

    try:
        with nvtx_range("forward"):
            with autocast_ctx:
                if forward_only:
                    with torch.no_grad():
                        logits = model(x)
                else:
                    logits = model(x)
            if use_cuda:
                torch.cuda.synchronize()

        if not forward_only:
            with nvtx_range("loss"):
                loss = logits.sum()

            with nvtx_range("backward"):
                loss.backward()
                if use_cuda:
                    torch.cuda.synchronize()

            with nvtx_range("optimizer_step"):
                optimizer.step()
                if use_cuda:
                    torch.cuda.synchronize()

    except torch.cuda.OutOfMemoryError:
        print(f"OOM during profiled step — trace may be incomplete.")
    finally:
        if use_cuda:
            torch.cuda.cudart().cudaProfilerStop()
        for h in handles:
            h.remove()
        _unpatch_sdpa(orig_sdpa)

    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"peak GPU memory: {peak_mb:.1f} MB")

    print("done — nsys capture region ended.")


# ── memory profiling routine ──────────────────────────────────────────────────

def run_memory_profile(
    model_size: str = "2.7B",
    vocab_size: int = 10_000,
    context_length: int = 128,
    batch_size: int = 4,
    warmup_steps: int = 3,
    forward_only: bool = False,
    mixed_precision: bool = False,   # ← ADDED
    device: str = "cuda",
    output_dir: str = "results/memory",
) -> None:
    """Run ONE step under torch.cuda.memory._record_memory_history and dump a pickle."""
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if device == "cuda" and not use_cuda:
        print("WARNING: CUDA not available — falling back to CPU.")
        device = "cpu"

    precision  = "bf16" if mixed_precision else "fp32"
    mode_label = "fwd" if forward_only else "train"
    print(f"[mem] model={model_size}  ctx={context_length}  precision={precision}  mode={mode_label}")

    result = _build_and_warmup(
        model_size, vocab_size, context_length, batch_size,
        warmup_steps, forward_only, mixed_precision, device
    )
    if result is None:
        return
    model, optimizer, x, autocast_ctx = result

    print("[mem] starting memory history recording…")
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    try:
        with autocast_ctx:
            if forward_only:
                with torch.no_grad():
                    logits = model(x)
            else:
                logits = model(x)

        if not forward_only:
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if use_cuda:
            torch.cuda.synchronize()

    except torch.cuda.OutOfMemoryError:
        print("[mem] OOM during profiled step — snapshot may be incomplete.")
    finally:
        os.makedirs(output_dir, exist_ok=True)
        # precision label in filename so fp32 and bf16 don't overwrite each other
        out_path = os.path.join(
            output_dir,
            f"memory_{model_size}_ctx{context_length}_{precision}_{mode_label}.pickle"
        )
        print(f"[mem] saving snapshot to {out_path} …")
        torch.cuda.memory._dump_snapshot(out_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"[mem] done. Load at https://pytorch.org/memory_viz")

    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"[mem] peak GPU memory: {peak_mb:.1f} MB")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVTX nsys profiling + memory profiling for BasicsTransformerLM",
    )
    parser.add_argument("--model-size",    choices=list(MODEL_SIZES.keys()), default="small")
    parser.add_argument("--vocab-size",    type=int,  default=10_000)
    parser.add_argument(
        "--context-length", type=int, default=128, choices=[128, 256, 512, 1024],
    )
    parser.add_argument("--batch-size",    type=int,  default=4)
    parser.add_argument("--warmup-steps",  type=int,  default=3)
    parser.add_argument(
        "--forward-only", action="store_true",
        help="Forward pass only (no backward, no optimizer step)",
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",   # ← ADDED
        help="Use BF16 autocast (default: fp32)",
    )
    parser.add_argument(
        "--no-layer-hooks", action="store_true",
        help="Skip per-layer NVTX hooks (useful for large models)",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--memory-profile", action="store_true",
        help="Run memory profiler instead of nsys profiler",
    )
    parser.add_argument(
        "--memory-output-dir", type=str, default="results/memory",
        help="Directory to save memory snapshot pickle files",
    )

    args = parser.parse_args()

    if args.memory_profile:
        run_memory_profile(
            model_size=args.model_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            forward_only=args.forward_only,
            mixed_precision=args.mixed_precision,   # ← ADDED
            device=args.device,
            output_dir=args.memory_output_dir,
        )
    else:
        run_profile(
            model_size=args.model_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            forward_only=args.forward_only,
            mixed_precision=args.mixed_precision,   # ← ADDED
            per_layer_hooks=not args.no_layer_hooks,
            device=args.device,
        )


if __name__ == "__main__":
    main()
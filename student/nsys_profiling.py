"""
nsys profiling script for BasicsTransformerLM.

Instruments the model with NVTX ranges so that Nsight Systems can attribute
CUDA kernels to specific model operations (forward, backward, optimizer step,
per-layer attention, FFN, and intra-attention ops: QK matmul, softmax, AV matmul).

Warmup steps run before the profiled region so JIT / cuDNN kernel selection
does not appear in the trace.  The script calls cudaProfilerStart/Stop around
the single profiled step, so you can use --capture-range=cudaProfilerApi to
keep the .nsys-rep file small.

────────────────────────────────────────────────────────────────────────────────
EXAMPLE nsys COMMANDS  (run from repo root on the cluster)
────────────────────────────────────────────────────────────────────────────────

# Forward pass only, small model, context 128:
nsys profile \\
    --capture-range=cudaProfilerApi --capture-range-end=stop \\
    --trace=cuda,nvtx \\
    -o results/nsys/small_ctx128_fwd \\
    uv run python student/nsys_profiling.py \\
        --model-size small --context-length 128 --forward-only

# Full training step (fwd + bwd + optimizer), medium model, context 256:
nsys profile \\
    --capture-range=cudaProfilerApi --capture-range-end=stop \\
    --trace=cuda,nvtx \\
    -o results/nsys/medium_ctx256_train \\
    uv run python student/nsys_profiling.py \\
        --model-size medium --context-length 256

# To open a report in the Nsight Systems GUI:
#   nsys-ui results/nsys/small_ctx128_fwd.nsys-rep
# Then use "Stats Systems View" → "CUDA GPU Kernel Summary" and filter
# by NVTX range (e.g. "forward", "layer_0/attn", "backward", etc.)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.cuda.nvtx as nvtx

# ── project imports ───────────────────────────────────────────────────────────
import a1_basics.model as _model_mod          # we monkey-patch SDPA here
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
    """Lightweight context manager that pushes/pops an NVTX range."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        nvtx.range_push(self.name)
        return self

    def __exit__(self, *_):
        nvtx.range_pop()


# ── SDPA monkey-patch ─────────────────────────────────────────────────────────

def _patch_sdpa() -> object:
    """
    Replace a1_basics.model.scaled_dot_product_attention with an annotated
    version that wraps the QK matmul, softmax, and AV matmul in NVTX ranges.

    Because CausalMultiHeadSelfAttention.forward looks up the name
    'scaled_dot_product_attention' in the module's __dict__ at call time,
    patching the module attribute is sufficient.

    Returns the original function so the caller can restore it.
    """
    from einops import einsum as _einsum

    # Capture references at patch time
    _orig_sdpa = _model_mod.scaled_dot_product_attention
    _softmax   = _model_mod.softmax          # nn_utils.softmax, imported into model

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

def _register_nvtx_hooks(
    model: BasicsTransformerLM,
    per_layer: bool = True,
) -> list:
    """
    Register forward pre/post hooks on key submodules to bracket their
    execution with NVTX push/pop.  Returns a list of hook handles.

    NVTX range hierarchy produced during a forward pass:
        forward                         ← wrapped by run_profile, not a hook
          embedding
          layer_0
            layer_0/ln1
            layer_0/attn
              attn/qk_matmul            ← from SDPA patch
              attn/softmax
              attn/av_matmul
            layer_0/ln2
            layer_0/ffn
          layer_1 … layer_N
          ln_final
          lm_head
    """
    handles: list = []

    def _hook_module(mod: torch.nn.Module, name: str) -> None:
        """Attach a push/pop hook pair to a module."""
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


# ── main profiling routine ────────────────────────────────────────────────────

def run_profile(
    model_size: str = "small",
    vocab_size: int = 10_000,
    context_length: int = 128,
    batch_size: int = 4,
    warmup_steps: int = 3,
    forward_only: bool = False,
    per_layer_hooks: bool = True,
    device: str = "cuda",
) -> None:
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if device == "cuda" and not use_cuda:
        print("[nsys_profiling] WARNING: CUDA not available — falling back to CPU.")
        device = "cpu"

    preset = MODEL_SIZES[model_size]
    mode   = "forward-only" if forward_only else "forward+backward+optimizer"

    print(
        f"[nsys_profiling] model={model_size}  ctx={context_length}  "
        f"batch={batch_size}  mode={mode}  device={device}"
    )

    # ── build model ───────────────────────────────────────────────────────────
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
        print(f"[nsys_profiling] OOM building model={model_size} ctx={context_length} — skipping.")
        return

    model.train()   # keep in train mode; use torch.no_grad() for forward-only
    print(f"[nsys_profiling] params (non-embedding): {model.get_num_params():,}")

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    # ── warmup ────────────────────────────────────────────────────────────────
    print(f"[nsys_profiling] warming up ({warmup_steps} step(s))…")
    try:
        for _ in range(warmup_steps):
            if forward_only:
                with torch.no_grad():
                    _ = model(x)
            else:
                logits = model(x)
                logits.sum().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(
            f"[nsys_profiling] OOM during warmup (model={model_size} ctx={context_length}) "
            "— skipping."
        )
        return

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # ── instrument the model ──────────────────────────────────────────────────
    orig_sdpa = _patch_sdpa()
    handles   = _register_nvtx_hooks(model, per_layer=per_layer_hooks)

    # ── start the cudaProfiler capture region ─────────────────────────────────
    # nsys must be invoked with --capture-range=cudaProfilerApi for this to
    # limit collection to exactly the region below.
    if use_cuda:
        torch.cuda.cudart().cudaProfilerStart()

    try:
        # ── FORWARD ──────────────────────────────────────────────────────────
        with nvtx_range("forward"):
            if forward_only:
                with torch.no_grad():
                    logits = model(x)
            else:
                logits = model(x)

        if use_cuda:
            torch.cuda.synchronize()

        if not forward_only:
            # ── LOSS ─────────────────────────────────────────────────────────
            # Wrap in its own range so loss ops don't pollute the backward range.
            with nvtx_range("loss"):
                loss = logits.sum()

            # ── BACKWARD ─────────────────────────────────────────────────────
            with nvtx_range("backward"):
                loss.backward()

            if use_cuda:
                torch.cuda.synchronize()

            # ── OPTIMIZER STEP ────────────────────────────────────────────────
            with nvtx_range("optimizer_step"):
                optimizer.step()

            if use_cuda:
                torch.cuda.synchronize()

    except torch.cuda.OutOfMemoryError:
        print(
            f"[nsys_profiling] OOM during profiled step "
            f"(model={model_size} ctx={context_length}) — trace may be incomplete."
        )
    finally:
        if use_cuda:
            torch.cuda.cudart().cudaProfilerStop()
        for h in handles:
            h.remove()
        _unpatch_sdpa(orig_sdpa)

    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"[nsys_profiling] peak GPU memory: {peak_mb:.1f} MB")

    print("[nsys_profiling] done — nsys capture region ended.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NVTX-annotated nsys profiling for BasicsTransformerLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_SIZES.keys()),
        default="small",
        help="Named model size from Table 1 (default: small)",
    )
    parser.add_argument("--vocab-size",      type=int,   default=10_000)
    parser.add_argument(
        "--context-length",
        type=int,
        default=128,
        choices=[128, 256, 512, 1024],
        help="Sequence length to profile (default: 128)",
    )
    parser.add_argument("--batch-size",      type=int,   default=4)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help=(
            "Steps before the profiled region — enough for CUDA JIT and cuDNN "
            "autotuning to finish (default: 3)"
        ),
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Profile forward pass only (no backward, no optimizer step)",
    )
    parser.add_argument(
        "--no-layer-hooks",
        action="store_true",
        help=(
            "Skip per-layer NVTX hooks (embedding / layer_i / ln_final / lm_head "
            "still annotated). Useful for very large models where per-layer hooks "
            "add clutter."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )

    args = parser.parse_args()

    run_profile(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        forward_only=args.forward_only,
        per_layer_hooks=not args.no_layer_hooks,
        device=args.device,
    )


if __name__ == "__main__":
    main()

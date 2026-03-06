import csv
import os
import torch
import triton
import triton.testing

# adjust import to match your actual filename/location
from student.FlashAttention import FlashAttention2
from a1_basics.model import scaled_dot_product_attention

SEQ_LENS  = [2**i for i in range(7, 17)]   # 128 → 65536
D_MODELS  = [2**i for i in range(4, 8)]    # 16  → 128
PRECISIONS = [torch.float32, torch.bfloat16]
BATCH_SIZE = 1


def pytorch_attn(Q, K, V):
    return scaled_dot_product_attention(Q, K, V)          # no mask — causal not supported in naive


def triton_attn(Q, K, V):
    return FlashAttention2.apply(Q, K, V, True)


def run_benchmark(seq_len, d_model, dtype, mode):
    device = "cuda"

    def make_inputs():
        return [
            torch.randn(BATCH_SIZE, seq_len, d_model,
                        device=device, dtype=dtype,
                        requires_grad=True)
            for _ in range(3)
        ]

    results = {}
    for name, fn in [("pytorch", pytorch_attn), ("triton_fa2", triton_attn)]:
        try:
            # smoke test
            Q, K, V = make_inputs()
            out = fn(Q, K, V)
            out.sum().backward()
            torch.cuda.empty_cache()

            if mode == "fwd":
                def bench_fn():
                    Q, K, V = make_inputs()
                    with torch.no_grad():
                        return fn(Q, K, V)
            elif mode == "bwd":
                # pre-run forward, time only backward
                Q, K, V = make_inputs()
                out = fn(Q, K, V)
                loss = out.sum()
                def bench_fn():
                    # re-use same graph — time backward only
                    nonlocal loss
                    Q2, K2, V2 = make_inputs()
                    out2 = fn(Q2, K2, V2)
                    loss2 = out2.sum()
                    loss2.backward()
            else:  # fwd_bwd
                def bench_fn():
                    Q, K, V = make_inputs()
                    out = fn(Q, K, V)
                    out.sum().backward()

            ms = triton.testing.do_bench(bench_fn, warmup=25, rep=100)
            results[name] = round(ms, 3)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[name] = "OOM"
        except Exception as e:
            results[name] = f"ERR"
            print(f"    ERROR {name} seq={seq_len} d={d_model}: {e}")

    return results


def main():
    os.makedirs("results", exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []
    for dtype in PRECISIONS:
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        for d_model in D_MODELS:
            for seq_len in SEQ_LENS:
                for mode in ["fwd", "bwd", "fwd_bwd"]:
                    print(f"  {dtype_str}  d={d_model:4d}  seq={seq_len:6d}  {mode} ...",
                          end="", flush=True)
                    res = run_benchmark(seq_len, d_model, dtype, mode)
                    print(f"  pytorch={res['pytorch']}  triton={res['triton_fa2']}")
                    rows.append({
                        "dtype":      dtype_str,
                        "d_model":    d_model,
                        "seq_len":    seq_len,
                        "mode":       mode,
                        "pytorch_ms": res["pytorch"],
                        "triton_ms":  res["triton_fa2"],
                    })

    out_path = "results/flash_bench.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
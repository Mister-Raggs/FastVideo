"""Modal A/B microbench: naive per-projection input quant vs shared input quant.

Isolates exactly what `perf/wan-shared-input-quant` removes. Both variants run
the SAME three split fp8 GEMMs (q/k/v, dim->dim); they differ only in how many
times the shared activation is quantized:

  * naive  : quantize x inside each projection  -> 3 activation-quant passes
             (what the generic FP8 linear path does today for Wan's to_q/k/v)
  * shared : quantize x once, reuse across q/k/v -> 1 activation-quant pass
             (this branch / LTX-2's pattern)

The delta is the 2 redundant memory-bound quant passes over [tokens, dim] that
the branch eliminates. Torch-only (no FastVideo/checkpoint), ~30s.

Run (L40S is the canonical FastVideo perf surface; fp8 needs sm89+):
    modal run fastvideo/tests/modal/shared_quant_ab.py
    modal run fastvideo/tests/modal/shared_quant_ab.py --model 1.3B
    GPU_TYPE=H100 modal run fastvideo/tests/modal/shared_quant_ab.py
"""
import os

import modal

GPU_TYPE = os.environ.get("GPU_TYPE", "L40S")
TORCH_SPEC = os.environ.get("QKV_AB_TORCH", "torch")

app = modal.App("shared-quant-ab")
image = modal.Image.debian_slim(python_version="3.12").pip_install(TORCH_SPEC)

MODELS = {
    "1.3B": dict(dim=1536, heads=12),
    "14B": dict(dim=5120, heads=40),
}
DEFAULT_SEQS = "2048,8192,32760"


@app.function(gpu=GPU_TYPE, image=image, timeout=1200)
def run(model: str, dtype_str: str, seqs: list[int], bs: int, iters: int,
        warmup: int) -> str:
    import statistics

    import torch

    dtype = getattr(torch, dtype_str)
    device = "cuda"
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log(f"torch {torch.__version__}  device={torch.cuda.get_device_name()}  "
        f"cap={torch.cuda.get_device_capability()}  gpu_type={GPU_TYPE}")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9 and not (cap[0] == 8 and cap[1] >= 9):
        return log(f"fp8 unsupported on cap {cap} (need sm89+)") or "\n".join(lines)

    fp8 = torch.float8_e4m3fn

    def q(t):  # per-tensor symmetric quant to e4m3 (amax/448)
        s = (t.abs().amax() / 448.0).clamp(min=1e-12).to(torch.float32)
        return (t / s).clamp(-448, 448).to(fp8), s

    def time_fn(fn) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)

    names = list(MODELS) if model == "all" else [model]
    for name in names:
        dim = MODELS[name]["dim"]
        log(f"\n=== Wan-{name}  dim={dim}  dtype={dtype}  bs={bs} ===")
        torch.manual_seed(0)
        # three fp8 q/k/v weights ([dim,dim]) + bf16 bias, quantized once up front
        split_wq = []
        for _ in range(3):
            w = torch.randn(dim, dim, device=device, dtype=dtype)
            wq, ws = q(w)
            b = torch.randn(dim, device=device, dtype=dtype)
            split_wq.append((wq, ws, b))

        hdr = (f"  {'seq':>7} {'naive us':>10} {'shared us':>11} "
               f"{'speedup':>9} {'saved us':>9}")
        log(hdr)
        log("  " + "-" * (len(hdr) - 2))
        for seq in seqs:
            M = ((seq + 15) // 16) * 16  # _scaled_mm wants M % 16 == 0
            x2 = torch.randn(bs * M, dim, device=device, dtype=dtype)

            def naive():  # quantize x separately inside each projection
                outs = []
                for wq, ws, b in split_wq:
                    xq, xs = q(x2)
                    outs.append(torch._scaled_mm(xq, wq.t(), xs, ws, bias=b,
                                                 out_dtype=dtype,
                                                 use_fast_accum=True))
                return outs

            def shared():  # quantize x once, reuse across q/k/v
                xq, xs = q(x2)
                return [torch._scaled_mm(xq, wq.t(), xs, ws, bias=b,
                                         out_dtype=dtype, use_fast_accum=True)
                        for wq, ws, b in split_wq]

            try:
                n_us = time_fn(naive) * 1000.0
                s_us = time_fn(shared) * 1000.0
            except Exception as ex:  # noqa: BLE001
                log(f"  {seq:>7}  fp8 skipped: {type(ex).__name__}: {ex}")
                continue
            tag = f"{seq}" + ("" if M == seq else f"->{M}")
            log(f"  {tag:>7} {n_us:>10.2f} {s_us:>11.2f} "
                f"{n_us / s_us:>8.2f}x {n_us - s_us:>8.2f}")
            del x2
            torch.cuda.empty_cache()

    log("\nNotes:")
    log("  * speedup>1 => shared (this branch) faster. The gap is the 2 "
        "redundant activation-quant passes naive does over [tokens, dim].")
    log("  * This is projection-only. For the e2e share, multiply 'saved us' "
        "by num_layers x denoise_steps, then confirm with a real fp8 Wan A/B.")
    return "\n".join(lines)


@app.local_entrypoint()
def main(model: str = "all", dtype: str = "bfloat16", seqs: str = DEFAULT_SEQS,
         bs: int = 1, iters: int = 50, warmup: int = 20) -> None:
    seq_list = sorted(int(s) for s in str(seqs).split(","))
    report = run.remote(model, dtype, seq_list, bs, iters, warmup)
    print("\n===== report =====")
    print(report)

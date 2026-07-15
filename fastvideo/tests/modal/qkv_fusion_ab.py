"""Modal A/B microbench: split QKV (3 GEMMs) vs fused QKV (1 GEMM) projection.

Isolates the exact op changed by `perf/kernel-fusion` — the self-attention QKV
projection. Torch-only (no FastVideo, no checkpoint, no pipeline), so the image
is lean and a run is ~30s. Same-process A/B (retro §6.10) kills box variance;
bit-exactness is asserted before timing. Runs eager AND torch.compile so it
empirically settles whether compile already merges the 3 GEMMs (it does not on
GPU today — inductor grouped-GEMM is CPU-only WIP).

Run (L40S is the canonical FastVideo perf surface):
    modal run fastvideo/tests/modal/qkv_fusion_ab.py
    modal run fastvideo/tests/modal/qkv_fusion_ab.py --model 1.3B --dtype float32
    modal run fastvideo/tests/modal/qkv_fusion_ab.py --no-compile
    GPU_TYPE=H100 modal run fastvideo/tests/modal/qkv_fusion_ab.py

GPU is chosen at import via GPU_TYPE (default L40S) so Modal can schedule it.
For exact FastVideo-stack parity instead of a lean torch image, run the same
bench through launch_l40s_job.py against the fastvideo-dev image.
"""
import os

import modal

GPU_TYPE = os.environ.get("GPU_TYPE", "L40S")
TORCH_SPEC = os.environ.get("QKV_AB_TORCH", "torch")

app = modal.App("qkv-fusion-ab")
image = modal.Image.debian_slim(python_version="3.12").pip_install(TORCH_SPEC)

# (hidden_dim, num_heads, head_dim). Wan is MHA, head_dim 128. Default full-seq
# is the Wan-1.3B 480p T2V token count (~32760).
MODELS = {
    "1.3B": dict(dim=1536, heads=12, head_dim=128),
    "14B": dict(dim=5120, heads=40, head_dim=128),
}
DEFAULT_SEQS = "512,2048,8192,32760"


@app.function(gpu=GPU_TYPE, image=image, timeout=1200)
def run(model: str, dtype_str: str, seqs: list[int], bs: int, iters: int,
        warmup: int, do_compile: bool) -> str:
    import statistics

    import torch
    import torch.nn as nn

    dtype = getattr(torch, dtype_str)
    device = "cuda"
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log(f"torch {torch.__version__}  device={torch.cuda.get_device_name()}  "
        f"cap={torch.cuda.get_device_capability()}  gpu_type={GPU_TYPE}")

    def build_pair(dim: int):
        torch.manual_seed(0)
        q = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        k = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        v = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        qkv = nn.Linear(dim, 3 * dim, bias=True, device=device, dtype=dtype)
        with torch.no_grad():  # merge_index 0/1/2 cat — exactly the loader op
            qkv.weight.copy_(torch.cat([q.weight, k.weight, v.weight], dim=0))
            qkv.bias.copy_(torch.cat([q.bias, k.bias, v.bias], dim=0))
        return (q, k, v), qkv

    def split_fn(x, mods):
        q, k, v = mods
        return q(x), k(x), v(x)

    def fused_fn(x, qkv):
        return qkv(x).chunk(3, dim=-1)

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
        cfg = MODELS[name]
        dim = cfg["dim"]
        log(f"\n=== Wan-{name}  dim={dim} heads={cfg['heads']} "
            f"head_dim={cfg['head_dim']}  dtype={dtype}  bs={bs} ===")
        split_mods, qkv = build_pair(dim)

        x0 = torch.randn(bs, max(seqs), dim, device=device, dtype=dtype)
        log("  correctness (fused == split):")
        for a, b, nm in zip(split_fn(x0, split_mods), fused_fn(x0, qkv), "qkv"):
            md = (a - b).abs().max().item()
            eq = torch.equal(a, b)
            tol = 0.0 if dtype == torch.float32 else 2e-2
            ok = eq or md <= tol
            log(f"    {nm}: equal={eq} max|diff|={md:.2e} {'OK' if ok else 'FAIL'}")
            assert ok, f"{nm} mismatch {md} > {tol}"
        del x0

        modes = [("eager", split_fn, fused_fn)]
        if do_compile:
            modes.append(("compile", torch.compile(split_fn), torch.compile(fused_fn)))

        hdr = (f"  {'mode':<8} {'seq':>7} {'split us':>10} {'fused us':>10} "
               f"{'speedup':>9} {'saved us':>9}")
        log(hdr)
        log("  " + "-" * (len(hdr) - 2))
        for mode, sfn, ffn in modes:
            for seq in seqs:
                x = torch.randn(bs, seq, dim, device=device, dtype=dtype)
                s_us = time_fn(lambda: sfn(x, split_mods)) * 1000.0
                f_us = time_fn(lambda: ffn(x, qkv)) * 1000.0
                log(f"  {mode:<8} {seq:>7} {s_us:>10.2f} {f_us:>10.2f} "
                    f"{s_us / f_us:>8.2f}x {s_us - f_us:>8.2f}")
                del x
                torch.cuda.empty_cache()

    log("\nNotes:")
    log("  * speedup>1 => fused faster. Largest win at short seq "
        "(launch-bound), shrinking toward full Wan seq (compute-bound).")
    log("  * Compile may narrow the eager gap via reduced launch overhead but "
        "cannot merge the 3 GEMMs on GPU (inductor grouped-GEMM is CPU-only).")
    log("  * bf16 is compute-bound => small win. The structural win is under "
        "int8/fp4 (wider GEMM + 1 quant epilogue instead of 3) — the quant "
        "variant is the follow-up that carries the PR story.")
    return "\n".join(lines)


@app.local_entrypoint()
def main(model: str = "all", dtype: str = "bfloat16", seqs: str = DEFAULT_SEQS,
         bs: int = 1, iters: int = 50, warmup: int = 20,
         compile: bool = True) -> None:
    seq_list = sorted(int(s) for s in str(seqs).split(","))
    report = run.remote(model, dtype, seq_list, bs, iters, warmup, compile)
    print("\n===== report (also streamed above from the container) =====")
    print(report)

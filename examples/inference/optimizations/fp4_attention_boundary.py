"""Map the accuracy and latency boundary of the sm12x FP4 attention kernel.

This is an operator probe, not a model-quality benchmark. It compares the full
``sageattn_blackwell`` path (Q/K/V quantization included) with BF16 Torch SDPA
for equal-length self-attention and unequal-length cross-attention shapes.

Examples:

    # Cheap first pass: two self-attention cases and one cross-attention case.
    python examples/inference/optimizations/fp4_attention_boundary.py --preset smoke

    # The useful boundary sweep for deciding whether cross-attention is viable.
    python examples/inference/optimizations/fp4_attention_boundary.py \
        --preset cross --heads 16 --warmup 3 --iterations 10

    # Run all 30 geometry, padding-tail, logit-scale, and head-dimension cases.
    python examples/inference/optimizations/fp4_attention_boundary.py \
        --preset full --heads 16 --warmup 3 --iterations 10

    # Add an exact shape captured from a model. Format:
    # label,batch,heads,q_len,kv_len,head_dim,causal,input_scale
    python examples/inference/optimizations/fp4_attention_boundary.py \
        --case cosmos-cross,1,16,32768,512,128,false,0.3

The JSON receipt records inputs, accuracy, latency, memory, device, CUDA, Torch,
and git revision. Keep FP4 linear quantization out of this experiment: it uses
different kernels and changes a different fraction of the model's runtime.

A ``pass`` means the operator met the accuracy threshold. It does not mean the
FP4 path was faster; use ``timing.speedup_x > 1`` to find the performance
crossover, then confirm promising shapes with an end-to-end model-quality A/B.
"""
from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
KERNEL_ROOT = REPO_ROOT / "fastvideo-kernel"
SUPPORTED_CAPABILITIES = {(12, 0), (12, 1)}

faulthandler.enable()


@dataclass(frozen=True)
class AttentionCase:
    label: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    head_dim: int = 128
    causal: bool = False
    input_scale: float = 0.3

    def validate(self) -> None:
        dimensions = {
            "batch": self.batch,
            "heads": self.heads,
            "q_len": self.q_len,
            "kv_len": self.kv_len,
            "head_dim": self.head_dim,
        }
        non_positive = {name: value for name, value in dimensions.items() if value <= 0}
        if non_positive:
            raise ValueError(f"{self.label}: dimensions must be positive; got {non_positive}")
        if self.head_dim % 16:
            raise ValueError(f"{self.label}: head_dim must be divisible by 16; got {self.head_dim}")
        if self.causal and self.q_len != self.kv_len:
            raise ValueError(f"{self.label}: causal cases require q_len == kv_len")
        if not math.isfinite(self.input_scale) or self.input_scale <= 0:
            raise ValueError(f"{self.label}: input_scale must be finite and positive")


def _self_cases(heads: int, head_dim: int = 128) -> list[AttentionCase]:
    return [
        AttentionCase(f"self-q{length}", 1, heads, length, length, head_dim)
        for length in (128, 512, 2048, 4096, 8192, 16384, 32768)
    ]


def _cross_cases(heads: int, head_dim: int = 128) -> list[AttentionCase]:
    return [
        AttentionCase("cross-q2048-k512", 1, heads, 2048, 512, head_dim),
        AttentionCase("cross-q8192-k512", 1, heads, 8192, 512, head_dim),
        AttentionCase("cross-q16384-k512", 1, heads, 16384, 512, head_dim),
        AttentionCase("cross-q32768-k128", 1, heads, 32768, 128, head_dim),
        AttentionCase("cross-q32768-k512", 1, heads, 32768, 512, head_dim),
        AttentionCase("cross-q32768-k1024", 1, heads, 32768, 1024, head_dim),
    ]


def _padding_cases(heads: int, head_dim: int = 128) -> list[AttentionCase]:
    """Exercise both sides of the kernel's 128-token padding boundary."""
    return [
        AttentionCase("self-q127", 1, heads, 127, 127, head_dim),
        AttentionCase("self-q129", 1, heads, 129, 129, head_dim),
        AttentionCase("cross-q4095-k127", 1, heads, 4095, 127, head_dim),
        AttentionCase("cross-q4096-k129", 1, heads, 4096, 129, head_dim),
        AttentionCase("cross-q4097-k512", 1, heads, 4097, 512, head_dim),
        AttentionCase("cross-q8192-k513", 1, heads, 8192, 513, head_dim),
    ]


def _scale_cases(heads: int, head_dim: int = 128) -> list[AttentionCase]:
    """Probe sensitivity to Q/K logit range, not just sequence geometry."""
    cases = []
    for input_scale in (0.1, 0.3, 1.0, 3.0):
        scale_label = str(input_scale).replace(".", "p")
        cases.extend([
            AttentionCase(f"self-scale-{scale_label}", 1, heads, 4096, 4096, head_dim, input_scale=input_scale),
            AttentionCase(f"cross-scale-{scale_label}", 1, heads, 8192, 512, head_dim, input_scale=input_scale),
        ])
    return cases


def preset_cases(preset: str, heads: int) -> list[AttentionCase]:
    if preset == "smoke":
        return [
            AttentionCase("self-q128", 1, heads, 128, 128),
            AttentionCase("self-q4096", 1, heads, 4096, 4096),
            AttentionCase("cross-q4096-k512", 1, heads, 4096, 512),
        ]
    if preset == "sequence":
        return _self_cases(heads)
    if preset == "cross":
        return _cross_cases(heads)
    if preset == "padding":
        return _padding_cases(heads)
    if preset == "scale":
        return _scale_cases(heads)
    if preset == "full":
        cases = _self_cases(heads) + _cross_cases(heads) + _padding_cases(heads) + _scale_cases(heads)
        cases.extend([
            AttentionCase("self-q4096-d64", 1, heads, 4096, 4096, 64),
            AttentionCase("cross-q8192-k512-d64", 1, heads, 8192, 512, 64),
            AttentionCase("causal-q4096", 1, heads, 4096, 4096, 128, True),
        ])
        return cases
    raise ValueError(f"unknown preset: {preset}")


def parse_case(spec: str) -> AttentionCase:
    fields = [field.strip() for field in spec.split(",")]
    if len(fields) != 8:
        raise ValueError("--case requires label,batch,heads,q_len,kv_len,head_dim,causal,input_scale; "
                         f"got {spec!r}")
    label, batch, heads, q_len, kv_len, head_dim, causal, input_scale = fields
    causal_value = causal.lower()
    if causal_value not in {"true", "false"}:
        raise ValueError(f"{label}: causal must be true or false; got {causal!r}")
    case = AttentionCase(
        label=label,
        batch=int(batch),
        heads=int(heads),
        q_len=int(q_len),
        kv_len=int(kv_len),
        head_dim=int(head_dim),
        causal=causal_value == "true",
        input_scale=float(input_scale),
    )
    case.validate()
    return case


def load_cases_json(path: Path) -> list[AttentionCase]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("cases JSON must contain a list of objects")
    cases = [AttentionCase(**item) for item in raw]
    for case in cases:
        case.validate()
    return cases


def resolve_cases(args: argparse.Namespace) -> list[AttentionCase]:
    cases: list[AttentionCase] = []
    if args.preset:
        cases.extend(preset_cases(args.preset, args.heads))
    cases.extend(parse_case(spec) for spec in args.case)
    if args.cases_json:
        cases.extend(load_cases_json(args.cases_json))
    if not cases:
        cases.extend(preset_cases("smoke", args.heads))

    labels: set[str] = set()
    for case in cases:
        case.validate()
        if case.label in labels:
            raise ValueError(f"duplicate case label: {case.label}")
        labels.add(case.label)
    return cases


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p25_ms": _percentile(values, 0.25),
        "p75_ms": _percentile(values, 0.75),
        "max_ms": max(values),
    }


def _time_pair(
    sdpa: Callable[[], torch.Tensor],
    fp4: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        sdpa()
        fp4()
    torch.cuda.synchronize()

    timings = {"sdpa": [], "fp4": []}
    functions = {"sdpa": sdpa, "fp4": fp4}
    for iteration in range(iterations):
        order = ("sdpa", "fp4") if iteration % 2 == 0 else ("fp4", "sdpa")
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = functions[name]()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end))
            del output
    return timings["sdpa"], timings["fp4"]


def _peak_increment(fn: Callable[[], torch.Tensor]) -> int:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return max(0, peak - baseline)


def _accuracy(fp4_output: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    fp4_float = fp4_output.float()
    reference_float = reference.float()
    difference = fp4_float - reference_float
    fp4_flat = fp4_float.flatten()
    reference_flat = reference_float.flatten()
    denominator = torch.linalg.vector_norm(reference_flat).clamp_min(torch.finfo(torch.float32).tiny)
    cosine = F.cosine_similarity(fp4_flat, reference_flat, dim=0)
    return {
        "finite": bool(torch.isfinite(fp4_float).all().item()),
        "cosine_similarity": float(cosine.item()),
        "relative_l2": float((torch.linalg.vector_norm(difference.flatten()) / denominator).item()),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "max_absolute_error": float(difference.abs().max().item()),
    }


def run_case(
    case: AttentionCase,
    *,
    warmup: int,
    iterations: int,
    min_cosine: float,
    sageattn_blackwell: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    shape_q = (case.batch, case.heads, case.q_len, case.head_dim)
    shape_kv = (case.batch, case.heads, case.kv_len, case.head_dim)
    q = torch.randn(shape_q, device=device, dtype=torch.bfloat16) * case.input_scale
    k = torch.randn(shape_kv, device=device, dtype=torch.bfloat16) * case.input_scale
    v = torch.randn(shape_kv, device=device, dtype=torch.bfloat16) * case.input_scale

    def sdpa() -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, is_causal=case.causal)

    def fp4() -> torch.Tensor:
        return sageattn_blackwell(q, k, v, is_causal=case.causal)

    reference = sdpa()
    fp4_output = fp4()
    torch.cuda.synchronize()
    accuracy = _accuracy(fp4_output, reference)
    del reference, fp4_output

    sdpa_times, fp4_times = _time_pair(sdpa, fp4, warmup, iterations)
    sdpa_timing = _timing_summary(sdpa_times)
    fp4_timing = _timing_summary(fp4_times)
    speedup = sdpa_timing["median_ms"] / fp4_timing["median_ms"]
    result = {
        "case": asdict(case),
        "status": "pass" if accuracy["finite"] and accuracy["cosine_similarity"] >= min_cosine else "accuracy_fail",
        "accuracy": accuracy,
        "timing": {
            "sdpa": sdpa_timing,
            "fp4": fp4_timing,
            "speedup_x": speedup,
            "latency_reduction_percent": (1.0 - fp4_timing["median_ms"] / sdpa_timing["median_ms"]) * 100.0,
        },
        "peak_memory_increment_bytes": {
            "sdpa": _peak_increment(sdpa),
            "fp4": _peak_increment(fp4),
        },
    }
    torch.cuda.empty_cache()
    return result


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("fastvideo-kernel")
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    capability = tuple(torch.cuda.get_device_capability())
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "fastvideo_kernel_version": _package_version(),
        "device_name": torch.cuda.get_device_name(),
        "device_capability": list(capability),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "minimum_cosine": args.min_cosine,
        "reference": "torch_sdpa_bfloat16",
        "fp4_timing_includes_qkv_quantization": True,
    }


def _print_result(result: dict[str, Any]) -> None:
    case = result["case"]
    if result["status"] == "error":
        print(f"{case['label']:<26} ERROR {result['error_type']}: {result['error']}")
        return
    accuracy = result["accuracy"]
    timing = result["timing"]
    print(f"{case['label']:<26} {result['status']:<13} "
          f"Q/K={case['q_len']}/{case['kv_len']} H={case['heads']} D={case['head_dim']} "
          f"cos={accuracy['cosine_similarity']:.6f} rel_l2={accuracy['relative_l2']:.4f} "
          f"sdpa={timing['sdpa']['median_ms']:.3f}ms fp4={timing['fp4']['median_ms']:.3f}ms "
          f"speedup={timing['speedup_x']:.3f}x")


def _write_receipt(output: Path, receipt: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "sequence", "cross", "padding", "scale", "full"))
    parser.add_argument("--heads", type=int, default=4, help="Head count used by preset cases (default: 4).")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="SPEC",
        help="Repeatable label,batch,heads,q_len,kv_len,head_dim,causal,input_scale case.",
    )
    parser.add_argument("--cases-json", type=Path, help="JSON list of AttentionCase objects.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--min-cosine", type=float, default=0.97)
    parser.add_argument("--output", type=Path, help="Receipt path; defaults to a timestamped JSON file.")
    parser.add_argument("--list-cases",
                        action="store_true",
                        help="Print resolved cases as JSON without requiring CUDA.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        cases = resolve_cases(args)
        if args.warmup < 0 or args.iterations <= 0:
            raise ValueError("--warmup must be non-negative and --iterations must be positive")
        if not 0.0 <= args.min_cosine <= 1.0:
            raise ValueError("--min-cosine must be between 0 and 1")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.list_cases:
        print(json.dumps([asdict(case) for case in cases], indent=2))
        return 0

    if not torch.cuda.is_available():
        parser.error("CUDA is required (use --list-cases to inspect the matrix without a GPU)")
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in SUPPORTED_CAPABILITIES:
        parser.error(f"the CUTLASS FP4 probe requires sm_120 or sm_121; active capability is {capability}")

    if str(KERNEL_ROOT) not in sys.path:
        sys.path.insert(0, str(KERNEL_ROOT))
    try:
        from attn_qat_infer.api import sageattn_blackwell
    except ImportError as exc:
        parser.error(f"FP4 extension is unavailable: {exc}")

    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path(f"fp4-attention-boundary-{stamp}.json")

    print(f"GPU: {torch.cuda.get_device_name()} (sm_{capability[0]}{capability[1]})")
    print(f"Cases: {len(cases)} | warmup={args.warmup} iterations={args.iterations}")
    results: list[dict[str, Any]] = []
    receipt = {"metadata": _metadata(args), "active_case": None, "results": results}
    _write_receipt(output, receipt)
    for index, case in enumerate(cases, start=1):
        receipt["active_case"] = asdict(case)
        _write_receipt(output, receipt)
        print(f"[{index}/{len(cases)}] running {case.label}", flush=True)
        try:
            result = run_case(
                case,
                warmup=args.warmup,
                iterations=args.iterations,
                min_cosine=args.min_cosine,
                sageattn_blackwell=sageattn_blackwell,
            )
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            result = {
                "case": asdict(case),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            torch.cuda.empty_cache()
        results.append(result)
        receipt["active_case"] = None
        _write_receipt(output, receipt)
        _print_result(result)

    print(f"Receipt: {output}")

    failures = sum(result["status"] != "pass" for result in results)
    print(f"Summary: {len(results) - failures} passed, {failures} failed/error")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

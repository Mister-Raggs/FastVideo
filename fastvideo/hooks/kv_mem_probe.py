"""Worker-side peak-memory attribution for the causal / AR KV-cache path.

D6 Step-0 gate.  D1 measured the KV caches on the SFWan (unwindowed) path at
~5.6 GiB, ~69% of the reported peak.  A large share of the *bytes* is not proof
that those bytes set the *high-water mark*: D1 shrank Matrix-Game's KV
allocation (it scales with ``local_attn_size``) and peak did not move at all
(6029.46 MB @w5 vs 6029.3 MB @w6).

There is a structural reason to expect the same trap here.  ``kv_cache1`` /
``kv_cache2`` are locals in ``CausalDMDDenosingStage.forward`` -- they are never
attached to the batch or to ``self`` -- so they are released when the denoising
stage returns, i.e. *before* VAE decode runs.  If the global peak lands in
decode, then no amount of KV quantization can move it.

So this probe answers one question: **which pipeline stage owns the global
peak, and how many bytes do the KV caches actually hold at that moment.**  If
denoising owns the peak, KV quantization has a real memory payoff and the port
is worth doing.  If decode owns it, D6's memory thesis closes here for the
price of one run.

Reads ``os.environ`` directly rather than ``fastvideo.envs``; if any of this
ever ships, move the vars into ``envs.py`` alongside ``FASTVIDEO_TRACE_*``.

    FASTVIDEO_KV_MEM_PROBE=1          enable (no-op otherwise)
    FASTVIDEO_KV_MEM_PROBE_OUTPUT     JSON report path ("<pid>" is substituted)
    FASTVIDEO_KV_MEM_SNAPSHOT=1       also dump a torch memory-history snapshot

MUST be attached worker-side.  Generation runs in a spawned worker, so a
launcher-side patch never sees the pipeline -- this cost D1 two wasted runs.
"""

import atexit
import contextlib
import json
import os
from typing import Any

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_MIB = 1024.0**2


def _enabled() -> bool:
    return os.getenv("FASTVIDEO_KV_MEM_PROBE", "0") != "0"


def _snapshot_enabled() -> bool:
    return os.getenv("FASTVIDEO_KV_MEM_SNAPSHOT", "0") != "0"


def _output_path() -> str:
    raw = os.getenv("FASTVIDEO_KV_MEM_PROBE_OUTPUT", "/tmp/fv_kvmem_<pid>.json")
    return raw.replace("<pid>", str(os.getpid()))


def _snapshot_path() -> str:
    root, _ = os.path.splitext(_output_path())
    return root + "_snapshot.pickle"


def _tensor_bytes(t: Any) -> int:
    if isinstance(t, torch.Tensor):
        return t.numel() * t.element_size()
    return 0


class KVMemProbe:
    """Records per-stage allocator high-water marks and exact KV-cache bytes."""

    def __init__(self) -> None:
        self.stage_records: list[dict[str, Any]] = []
        self.kv_allocations: list[dict[str, Any]] = []
        self._restore: list[tuple[Any, str, Any]] = []
        self._call_index = 0
        self._reported = False
        self.baseline_alloc_mib = 0.0
        self.baseline_reserved_mib = 0.0

    # ---------------------------------------------------------------- attach

    def attach(self, stages: list[Any]) -> None:
        for stage in stages:
            self._wrap_stage(stage)
            self._wrap_kv_init(stage)

        if _snapshot_enabled():
            try:
                torch.cuda.memory._record_memory_history(max_entries=200_000)
                logger.info("[kv-mem] recording memory history")
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.warning("[kv-mem] could not start memory history: %s", exc)

        # Baseline: weights + anything allocated before the first stage runs.
        torch.cuda.reset_peak_memory_stats()
        self.baseline_alloc_mib = torch.cuda.memory_allocated() / _MIB
        self.baseline_reserved_mib = torch.cuda.memory_reserved() / _MIB
        atexit.register(self.report)

    def _wrap_stage(self, stage: Any) -> None:
        original = stage.forward
        name = f"{getattr(stage, '_pipeline_stage_name', '')}|{stage.__class__.__name__}"

        def wrapped(batch, fastvideo_args, _orig=original, _name=name):
            idx = self._call_index
            self._call_index += 1
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            alloc_in = torch.cuda.memory_allocated()
            try:
                return _orig(batch, fastvideo_args)
            finally:
                torch.cuda.synchronize()
                self.stage_records.append({
                    "order": idx,
                    "stage": _name,
                    "alloc_in_mib": round(alloc_in / _MIB, 2),
                    "alloc_out_mib": round(torch.cuda.memory_allocated() / _MIB, 2),
                    "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / _MIB, 2),
                    "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / _MIB, 2),
                })
                # Flush after every stage: this runs on a remote box, and an
                # atexit-only write loses the whole run if the worker is killed.
                self._write_report()

        stage.forward = wrapped
        self._restore.append((stage, "forward", original))

    def _wrap_kv_init(self, stage: Any) -> None:
        """Tally the bytes handed back by ``_initialize_kv_cache``.

        Wrapping the factory rather than reading the cache off the stage is
        deliberate: the caches are function locals, so there is nothing to read
        afterwards.
        """
        original = getattr(stage, "_initialize_kv_cache", None)
        if original is None:
            return

        def wrapped(*args, _orig=original, _stage=stage, **kwargs):
            cache = _orig(*args, **kwargs)
            k_bytes = sum(_tensor_bytes(blk.get("k")) for blk in cache)
            v_bytes = sum(_tensor_bytes(blk.get("v")) for blk in cache)
            first_k = cache[0]["k"] if cache else None
            self.kv_allocations.append({
                "stage": _stage.__class__.__name__,
                "num_blocks": len(cache),
                "kv_bytes": k_bytes + v_bytes,
                "kv_mib": round((k_bytes + v_bytes) / _MIB, 2),
                "per_block_mib": round((k_bytes + v_bytes) / max(len(cache), 1) / _MIB, 2),
                "shape": list(first_k.shape) if first_k is not None else None,
                "dtype": str(first_k.dtype) if first_k is not None else None,
                "local_attn_size": getattr(_stage, "local_attn_size", None),
                "sliding_window_num_frames": getattr(_stage, "sliding_window_num_frames", None),
                "frame_seq_length": getattr(_stage, "frame_seq_length", None),
            })
            return cache

        stage._initialize_kv_cache = wrapped
        self._restore.append((stage, "_initialize_kv_cache", original))

    # ---------------------------------------------------------------- report

    def _build_report(self) -> dict[str, Any] | None:
        if not self.stage_records:
            return None

        peak_stage = max(self.stage_records, key=lambda r: r["peak_alloc_mib"])
        # Two caches are allocated when a boundary timestep splits the experts
        # (the Wan2.2 MoE path), so this is a sum, not a single buffer.
        kv_mib = sum(a["kv_mib"] for a in self.kv_allocations)

        report = {
            "baseline_alloc_mib": round(self.baseline_alloc_mib, 2),
            "baseline_reserved_mib": round(self.baseline_reserved_mib, 2),
            "global_peak_alloc_mib": peak_stage["peak_alloc_mib"],
            "peak_owner_stage": peak_stage["stage"],
            "kv_total_mib": round(kv_mib, 2),
            "kv_alloc_count": len(self.kv_allocations),
            "kv_share_of_peak":
            (round(kv_mib / peak_stage["peak_alloc_mib"], 4) if peak_stage["peak_alloc_mib"] else None),
            # The gate. KV is freed when the denoising stage returns, so it can
            # only be live at the peak if the peak is owned by a stage that
            # allocated one. Matched on the recorded class name rather than a
            # substring -- the shipped class is spelled "CausalDMDDenosingStage".
            "kv_live_at_peak": any(a["stage"] == peak_stage["stage"].split("|")[-1] for a in self.kv_allocations),
            "kv_allocations": self.kv_allocations,
            "stages": self.stage_records,
        }
        return report

    def _write_report(self) -> dict[str, Any] | None:
        report = self._build_report()
        if report is None:
            return None
        path = _output_path()
        try:
            with open(path, "w") as fh:
                json.dump(report, fh, indent=2)
        except OSError as exc:  # pragma: no cover - diagnostic only
            logger.warning("[kv-mem] could not write %s: %s", path, exc)
        return report

    def report(self) -> dict[str, Any] | None:
        """Final write + human-readable summary. Safe to call more than once."""
        if self._reported:
            return None
        report = self._write_report()
        if report is None:
            return None
        self._reported = True
        path = _output_path()

        if _snapshot_enabled():
            try:
                torch.cuda.memory._dump_snapshot(_snapshot_path())
                torch.cuda.memory._record_memory_history(enabled=None)
                logger.info("[kv-mem] snapshot -> %s", _snapshot_path())
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.warning("[kv-mem] could not dump snapshot: %s", exc)

        # print, not logger: this runs from atexit, by which point logging
        # handlers may already be torn down and the summary would vanish.
        print(f"\n[kv-mem] GLOBAL PEAK {report['global_peak_alloc_mib']:.1f} MiB "
              f"owned by {report['peak_owner_stage']}")
        print(f"[kv-mem] KV caches {report['kv_total_mib']:.1f} MiB in "
              f"{report['kv_alloc_count']} allocation(s), "
              f"{report['kv_share_of_peak']} of peak")
        print(f"[kv-mem] *** KV LIVE AT PEAK: {report['kv_live_at_peak']} *** "
              "(False => quantizing KV cannot reduce peak)")
        print(f"[kv-mem] baseline (weights, pre-stage) {report['baseline_alloc_mib']:.1f} MiB")
        for rec in self.stage_records:
            print(f"[kv-mem]   {rec['stage']:<52} peak {rec['peak_alloc_mib']:9.1f} MiB  "
                  f"(in {rec['alloc_in_mib']:8.1f} -> out {rec['alloc_out_mib']:8.1f})")
        print(f"[kv-mem] report -> {path}\n")
        return report

    def detach(self) -> None:
        for obj, attr, original in self._restore:
            with contextlib.suppress(Exception):  # pragma: no cover - diagnostic only
                setattr(obj, attr, original)
        self._restore.clear()


def attach_kv_mem_probe(pipeline: Any) -> KVMemProbe | None:
    """Attach the Step-0 memory probe. No-op unless FASTVIDEO_KV_MEM_PROBE=1."""
    if not _enabled():
        return None
    if not torch.cuda.is_available():
        logger.warning("[kv-mem] CUDA unavailable; probe not attached")
        return None
    stages = getattr(pipeline, "stages", None)
    if not stages:
        logger.warning("[kv-mem] pipeline has no stages; probe not attached")
        return None

    probe = KVMemProbe()
    probe.attach(stages)
    logger.info("[kv-mem] attached to %d stages (pid %d)", len(stages), os.getpid())
    return probe

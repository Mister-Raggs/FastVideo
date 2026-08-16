# SPDX-License-Identifier: Apache-2.0
"""
D1 Phase-A: causal KV-cache attention profiler.

Measures where attention mass goes as a function of token age, per layer AND per
head, on the causal/AR DiTs. That is the measurement that decides whether a
layer/head-adaptive KV budget has anything to exploit, or whether the published
results (Forcing-KV arXiv 2605.09681, Future Forcing 2605.30083) fail to
reproduce on FastVideo's checkpoints.

Enable with FASTVIDEO_KV_PROBE=1. Output path via FASTVIDEO_KV_PROBE_OUTPUT
(default /tmp/fv_kvprobe_<pid>.json).

MUST run inside the worker process -- generation happens in a spawned worker, so
patching from the launcher never takes effect. This attaches from
ComposedPipelineBase.post_init(), same place as attach_activation_trace().

Instrumentation only: pre_forward returns its args unmodified, and the sampled
attention used for statistics is computed separately from the real attention.

NOTE: reads os.environ directly rather than fastvideo.envs. If any of this ever
ships, move the two vars into envs.py alongside FASTVIDEO_TRACE_*.
"""

from __future__ import annotations

import atexit
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Causal self-attention wrappers differ per model family -- CausalWanSelfAttention
# (Wan/SFWan), CausalMatrixGame2SelfAttention (Matrix-Game 2),
# DreamXPropeSelfAttention (DreamX-World) -- so match structurally instead of by
# name: a self-attention module that delegates to a LocalAttention child. Cross
# attention is excluded by name. Override with FASTVIDEO_KV_PROBE_CLS (comma
# separated) if the heuristic misses.
_LOCAL_ATTN_CLS = "LocalAttention"


def _is_causal_self_attn(module: nn.Module) -> bool:
    override = os.getenv("FASTVIDEO_KV_PROBE_CLS", "").strip()
    if override:
        return type(module).__name__ in {s.strip() for s in override.split(",") if s.strip()}
    cls = type(module).__name__
    if "SelfAttention" not in cls or "Cross" in cls:
        return False
    inner = getattr(module, "attn", None)
    return isinstance(inner, nn.Module) and type(inner).__name__ == _LOCAL_ATTN_CLS


def _enabled() -> bool:
    return os.getenv("FASTVIDEO_KV_PROBE", "0") != "0"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _output_path() -> Path:
    raw = os.getenv("FASTVIDEO_KV_PROBE_OUTPUT", "/tmp/fv_kvprobe_<pid>.json")
    return Path(raw.replace("<pid>", str(os.getpid())))


class KVProbeHook(ForwardHook):
    """Attached to the LocalAttention inside each causal self-attention block.

    Its pre_forward sees exactly (q, key_window, value_window) -- q is the new
    tokens, key_window is the slice of the KV cache actually attended, ordered
    oldest -> newest. That is precisely the input an eviction policy would have
    to make a decision about.
    """

    @classmethod
    def name(cls) -> str:
        return "KVProbeHook"

    def __init__(self, layer_name: str, layer_idx: int, sink_frac: float, cfg: dict[str, int],
                 store: dict[int, dict]) -> None:
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.sink_frac = sink_frac
        self.cfg = cfg
        self.store = store

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if len(args) >= 2:
            try:
                self._record(args[0], args[1])
            except Exception as exc:  # a statistic must never break generation
                rec = self.store.setdefault(self.layer_idx, {})
                rec.setdefault("errors", []).append(repr(exc)[:200])
        return args, kwargs

    @torch.no_grad()
    def _record(self, q: torch.Tensor, k: torch.Tensor) -> None:
        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
            return
        if q.ndim != 4 or k.ndim != 4:
            return

        _, Lq, H, D = q.shape
        Lk = k.shape[1]
        if Lq == 0 or Lk == 0:
            return

        rec = self.store.setdefault(self.layer_idx, {})
        rec.setdefault("layer_name", self.layer_name)
        rec.setdefault("stat_calls", 0)
        rec["calls"] = rec.get("calls", 0) + 1
        rec["q_len"] = int(Lq)
        rec["kv_len_max"] = max(rec.get("kv_len_max", 0), int(Lk))
        rec["num_heads"] = int(H)
        rec["head_dim"] = int(D)
        rec["local_attn_size"] = self.cfg.get("local_attn_size")
        rec["sink_size"] = self.cfg.get("sink_size")

        if rec["stat_calls"] >= self.cfg["max_calls"]:
            return

        n_h = min(self.cfg["sample_heads"], H)
        head_idx = torch.linspace(0, H - 1, n_h).round().long()
        n_q = min(self.cfg["sample_queries"], Lq)
        # bias toward the most recent queries -- those are the ones a future
        # eviction decision has to keep serving
        q_idx = torch.linspace(max(0, Lq - 4 * n_q), Lq - 1, n_q).round().long()

        qs = q[0, q_idx][:, head_idx].to(torch.float32)
        ks = k[0, :, head_idx].to(torch.float32)

        scores = torch.bmm(qs.permute(1, 0, 2), ks.permute(1, 2, 0)) / math.sqrt(D)
        probs = torch.softmax(scores, dim=-1)
        mass = probs.mean(dim=1)  # [n_h, Lk]

        nb = self.cfg["buckets"]
        edges = torch.linspace(0, Lk, nb + 1).round().long()
        head_age = torch.stack([mass[:, edges[i]:edges[i + 1]].sum(dim=-1) for i in range(nb)], dim=-1)

        n_sink = int(round(self.sink_frac * Lk))
        head_sink = mass[:, :n_sink].sum(dim=-1) if n_sink > 0 else torch.zeros(n_h, device=mass.device)
        # "recent frame" ~ the newest 1/local_attn_size of the window
        las = self.cfg.get("local_attn_size") or 0
        n_recent = int(round(Lk / las)) if las > 0 else 0
        head_recent = mass[:, -n_recent:].sum(dim=-1) if n_recent > 0 else torch.zeros(n_h, device=mass.device)

        ent = -(probs.clamp_min(1e-9).log() * probs).sum(dim=-1).mean(dim=1)
        kt = max(1, int(0.05 * Lk))
        top5 = mass.topk(kt, dim=-1).values.sum(dim=-1)

        for key, val in (("age_mass_sum", head_age.mean(dim=0)), ("head_age_mass_sum", head_age), ("head_sink_mass_sum",
                                                                                                   head_sink),
                         ("head_recent_mass_sum", head_recent), ("head_entropy_sum", ent), ("head_top5pct_sum", top5)):
            cur = rec.get(key)
            rec[key] = val.double().cpu() if cur is None else cur + val.double().cpu()
        rec["stat_calls"] += 1


class KVProbeManager:

    def __init__(self, managers: list[ModuleHookManager], store: dict[int, dict], path: Path) -> None:
        self.managers = managers
        self.store = store
        self.path = path
        self._dumped = False

    def dump(self) -> None:
        if self._dumped:
            return
        self._dumped = True
        layers = []
        total_bytes = 0
        for idx in sorted(self.store):
            r = self.store[idx]
            n = max(1, r.get("stat_calls", 0))
            H, D, Lk = r.get("num_heads"), r.get("head_dim"), r.get("kv_len_max", 0)
            kv_bytes = 2 * (Lk or 0) * (H or 0) * (D or 0) * 2  # k+v, bf16
            total_bytes += kv_bytes
            entry = {
                k: r.get(k)
                for k in ("layer_name", "calls", "stat_calls", "q_len", "kv_len_max", "num_heads", "head_dim",
                          "local_attn_size", "sink_size", "errors")
            }
            entry["layer"] = idx
            entry["kv_bytes_est"] = kv_bytes
            for key, out in (("age_mass_sum", "age_mass"), ("head_sink_mass_sum", "head_sink_mass"),
                             ("head_recent_mass_sum", "head_recent_mass"), ("head_entropy_sum", "head_entropy"),
                             ("head_top5pct_sum", "head_top5pct"), ("head_age_mass_sum", "head_age_mass")):
                v = r.get(key)
                entry[out] = (v / n).tolist() if v is not None else None
            layers.append(entry)

        report = {
            "pid": os.getpid(),
            "num_layers_seen": len(layers),
            "kv_cache_total_bytes_est": total_bytes,
            "kv_cache_total_gib_est": round(total_bytes / (1024**3), 3),
            "peak_mem_gib":
            round(torch.cuda.max_memory_allocated() / (1024**3), 3) if torch.cuda.is_available() else None,
            "layers": layers,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("KV probe wrote %d layers to %s", len(layers), self.path)


def attach_kv_probe(model: nn.Module | None) -> KVProbeManager | None:
    """Attach KV probe hooks to the causal self-attention blocks of *model*.

    Returns None when the probe is off or the model has no causal attention
    (i.e. it is not an AR/world model) -- both are normal, silent no-ops.
    """
    if not _enabled() or model is None:
        return None

    cfg = {
        "sample_heads": _int_env("FASTVIDEO_KV_PROBE_HEADS", 8),
        "sample_queries": _int_env("FASTVIDEO_KV_PROBE_QUERIES", 32),
        "buckets": _int_env("FASTVIDEO_KV_PROBE_BUCKETS", 12),
        "max_calls": _int_env("FASTVIDEO_KV_PROBE_MAX_CALLS", 24),
    }
    store: dict[int, dict] = {}
    managers: list[ModuleHookManager] = []

    idx = 0
    for name, module in model.named_modules():
        if not _is_causal_self_attn(module):
            continue
        inner = getattr(module, "attn", None)
        if not isinstance(inner, nn.Module):
            continue
        local_attn_size = int(getattr(module, "local_attn_size", -1) or -1)
        sink_size = int(getattr(module, "sink_size", 0) or 0)
        sink_frac = (sink_size / local_attn_size) if local_attn_size > 0 else 0.0
        layer_cfg = dict(cfg, local_attn_size=local_attn_size, sink_size=sink_size)

        manager = ModuleHookManager.get_from_or_default(inner)
        manager.append_forward_hook(KVProbeHook(f"{name}.attn", idx, sink_frac, layer_cfg, store))
        managers.append(manager)
        idx += 1

    if not managers:
        logger.warning("FASTVIDEO_KV_PROBE=1 but no causal self-attention modules found -- "
                       "not a causal model, or the matcher missed (try FASTVIDEO_KV_PROBE_CLS)")
        return None

    mgr = KVProbeManager(managers, store, _output_path())
    atexit.register(mgr.dump)
    logger.info("KV probe attached to %d causal attention layers (cfg=%s)", len(managers), cfg)
    return mgr

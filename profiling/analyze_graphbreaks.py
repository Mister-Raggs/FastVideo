"""Summarize the graph-break / recompile log from compile_graphbreaks.py.

Hardened parser. The raw stream is noisy: every torch record is wrapped
in a `(Worker pid=NNNN) [rankK]:V<ts> ... [__graph_breaks]` log prefix,
and each break is followed by boilerplate ("For more details ...",
"NOTE: the most recent ...") plus a code excerpt. The previous version
parsed those prefix/boilerplate lines as bogus "reasons" and only
flagged `dits/`-style frames as HOT — so it mis-classified the
per-layer layerwise-offload-hook cluster (the actual hottest one) as
cold. This version:

  - strips the torch-log prefix up to and including `[__graph_breaks]`
  - ignores boilerplate continuation lines
  - pairs each break's *reason* with the nearest `fastvideo/...:LINE`
    user frame within the same record
  - flags HOT by per-invocation cost location: anything in the model
    forward OR a per-layer hook / denoise path (paid layers x steps),
    not just `dits/`

Feeds the nsys-ai "profile finding -> source attribution" direction.

Usage:
    python profiling/analyze_graphbreaks.py /tmp/gb.log
"""

import collections
import re
import sys

# Frames whose break is paid per-layer and/or per-step (the expensive
# ones). Per-layer hooks and the denoise loop count as hot even though
# they are not under dits/.
HOT_HINTS = (
    "models/dits/", "attention", "transformer", "/block", "denois",
    "rotary", "norm", "modulation", "hooks/", "layerwise", "offload",
    "pipelines/stages/denoising",
)

# Lines that are torch/boilerplate, never a real break reason.
_BOILERPLATE = re.compile(
    r"(for more details|NOTE: the most recent|torch/_dynamo|traceback|"
    r"^\s*File \"|symbolic_convert|GRAPHBREAK-RUN|COMPILE-AND-RUN|"
    r"^\s*\^+\s*$)",
    re.IGNORECASE,
)
# Strip the worker/log prefix up to and including the [__graph_breaks]
# (or [__recompiles]) tag so we key on the message, not the log line.
_PREFIX = re.compile(r"^.*?\[__(?:graph_breaks|recompiles)\]\s*")
_FRAME = re.compile(r"(fastvideo/[\w/]+\.py:\d+)")
_REASON = re.compile(
    r"(Graph break.*|when attempting to trace \w+.*|"
    r"in user code at .*|Reason:.*)", re.IGNORECASE)


def _clean(ln: str) -> str:
    return _PREFIX.sub("", ln).strip()


def main(path: str) -> None:
    lines = open(path, errors="replace").read().splitlines()

    breaks: list[tuple[str, str]] = []
    recompiles: list[str] = []

    for i, raw in enumerate(lines):
        low = raw.lower()
        is_break = "graph break" in low or "[__graph_breaks]" in low
        is_recompile = "recompiling" in low or (
            "recompile" in low and "due to" in low)
        if not (is_break or is_recompile):
            continue

        msg = _clean(raw)
        if not msg or _BOILERPLATE.search(msg):
            continue

        if is_recompile:
            recompiles.append(msg[:160])
            continue

        # Pair the reason with the nearest fastvideo frame in this record
        # window (frame may be on a following line of the same record).
        window = " ".join(_clean(x) for x in lines[i:i + 6])
        frame_m = _FRAME.search(window)
        reason_m = _REASON.search(msg) or _REASON.search(window)
        reason = (reason_m.group(1) if reason_m else msg).strip()[:140]
        # Drop records that are only a bare frame line with no reason
        # text and no resolvable frame (continuation of a counted break).
        if reason.lower().startswith("in user code at") and not frame_m:
            continue
        breaks.append((reason, frame_m.group(1) if frame_m else "?"))

    by_reason = collections.Counter(r for r, _ in breaks)
    print(f"\n{'='*72}\nGRAPH BREAKS: {len(breaks)} total, "
          f"{len(by_reason)} distinct reasons\n{'='*72}")
    hot_total = 0
    for reason, n in by_reason.most_common():
        frames = sorted({f for r, f in breaks if r == reason and f != "?"})
        hot = any(any(h in f for h in HOT_HINTS) for f in frames)
        if hot:
            hot_total += n
        print(f"\n[{n}x]{'  <<< HOT (per-layer / per-step)' if hot else ''}"
              f"\n  reason: {reason}")
        for f in frames[:6]:
            print(f"  at:     {f}")

    print(f"\n{'='*72}\nRECOMPILES: {len(recompiles)} "
          f"(guard failures -> re-trace; many = dynamic-shape thrash)"
          f"\n{'='*72}")
    for r, c in collections.Counter(recompiles).most_common(10):
        print(f"  [{c}x] {r}")

    print(f"\n{'='*72}\nTRIAGE: {hot_total} break-instances are HOT "
          f"(model fwd / per-layer hook / denoise loop) — paid "
          f"layers x steps. Fix those first.\n{'='*72}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python profiling/analyze_graphbreaks.py <gb.log>")
        sys.exit(1)
    main(sys.argv[1])

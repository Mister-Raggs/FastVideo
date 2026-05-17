"""Summarize the graph-break / recompile log from compile_graphbreaks.py.

Turns the raw TORCH_LOGS stream into a triage table:
  - unique break reasons, with counts
  - the user-code frame (fastvideo/...) each break maps to
  - recompile events (guard failures) and their reason
  - a hot/cold hint: breaks whose frame is in a transformer block /
    attention / the denoise path matter far more than setup-time breaks

Usage:
    python profiling/analyze_graphbreaks.py /tmp/gb.log
"""

import collections
import re
import sys

HOT_HINTS = ("dits/", "attention", "transformer", "block", "denois",
             "rotary", "norm", "modulation")


def main(path: str) -> None:
    text = open(path, errors="replace").read()
    lines = text.splitlines()

    # torch logs a graph break as a line containing "Graph break"
    # (TORCHDYNAMO_VERBOSE adds the reason + a "due to:" / source frame).
    breaks = []
    recompiles = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "graph break" in low:
            ctx = " ".join(lines[i:i + 6])
            frame = re.search(r"(fastvideo/[\w/]+\.py:\d+)", ctx)
            reason = re.search(r"[Gg]raph break[:\s]+(.*)", ln)
            breaks.append((
                reason.group(1).strip()[:140] if reason else ln.strip()[:140],
                frame.group(1) if frame else "?",
            ))
        elif "recompiling" in low or "recompile" in low and "due to" in low:
            recompiles.append(ln.strip()[:160])

    print(f"\n{'='*70}\nGRAPH BREAKS: {len(breaks)} total\n{'='*70}")
    by_reason = collections.Counter(r for r, _ in breaks)
    for reason, n in by_reason.most_common():
        frames = sorted({f for r, f in breaks if r == reason and f != "?"})
        hot = any(any(h in f for h in HOT_HINTS) for f in frames)
        tag = "  <<< HOT (in model fwd)" if hot else ""
        print(f"\n[{n}x]{tag}\n  reason: {reason}")
        for f in frames[:6]:
            print(f"  at:     {f}")

    print(f"\n{'='*70}\nRECOMPILES: {len(recompiles)} "
          f"(guard failures -> re-trace; dynamic-shape thrash if many)"
          f"\n{'='*70}")
    for r in collections.Counter(recompiles).most_common(10):
        print(f"  [{r[1]}x] {r[0]}")

    hot_breaks = sum(
        n for reason, n in by_reason.items()
        for f in {f for r, f in breaks if r == reason}
        if any(h in f for h in HOT_HINTS))
    print(f"\n{'='*70}\nTRIAGE: {hot_breaks} break-instances are in the model "
          f"forward (hot). Fix those first — a break inside the per-layer "
          f"loop is paid every layer x every step.\n{'='*70}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python profiling/analyze_graphbreaks.py <gb.log>")
        sys.exit(1)
    main(sys.argv[1])

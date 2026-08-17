#!/usr/bin/env python3
"""Head-level analysis of a kv_probe JSON. No GPU, no torch.

  python kvpolicy_heads.py kvprobe_out/mg2_324498.json

Answers two questions that decide S2's granularity:
  1. Is the per-head newest-bucket mass BIMODAL (two clean groups -> a static
     static/dynamic split is enough) or CONTINUOUS (needs a real allocator)?
  2. Is a given head slot static in EVERY layer (-> allocate per head index,
     one profile) or does it vary by layer (-> allocate per (layer, head))?

NOTE: the probe samples `sample_heads` of the model's heads (default 8 of 12),
so column j is the j-th SAMPLED head, not model head j.
"""
import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <kvprobe.json>")
    d = json.load(open(sys.argv[1]))
    layers = [x for x in d["layers"] if x.get("head_age_mass")]
    if not layers:
        sys.exit("no head_age_mass in this JSON (older probe build?)")

    # newest-bucket mass per (layer, sampled head)
    grid = [[h[-1] for h in x["head_age_mass"]] for x in layers]
    n_l, n_h = len(grid), len(grid[0])
    flat = sorted(v for row in grid for v in row)
    n = len(flat)

    print(f"{n_l} layers x {n_h} sampled heads = {n} head-instances")
    print(f"newest-bucket mass: min {flat[0]:.3f}  med {flat[n//2]:.3f}  max {flat[-1]:.3f}")

    # --- Q1: bimodality -------------------------------------------------
    print("\nQ1 -- distribution (10 bins over [0,1]):")
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        c = sum(1 for v in flat if lo <= v < hi or (b == 9 and v >= hi))
        print(f"  {lo:.1f}-{hi:.1f} |{'#' * round(60 * c / n):<60}| {c:4d} ({100*c/n:4.1f}%)")

    # largest gap in the sorted values = natural split point
    gaps = [(flat[i + 1] - flat[i], i) for i in range(n - 1)]
    gap, gi = max(gaps)
    split = (flat[gi] + flat[gi + 1]) / 2
    lo_grp = [v for v in flat if v <= split]
    hi_grp = [v for v in flat if v > split]
    print(f"\n  largest gap {gap:.3f} at mass {split:.3f}")
    print(f"  -> below: {len(lo_grp):3d} heads (mean {sum(lo_grp)/max(1,len(lo_grp)):.3f})")
    print(f"  -> above: {len(hi_grp):3d} heads (mean {sum(hi_grp)/max(1,len(hi_grp)):.3f})")
    # separation: between-group distance vs within-group spread
    def sd(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return (sum((x - m)**2 for x in xs) / len(xs))**0.5
    sep = (sum(hi_grp)/max(1,len(hi_grp)) - sum(lo_grp)/max(1,len(lo_grp))) / max(1e-9, (sd(lo_grp) + sd(hi_grp)))
    minor = min(len(lo_grp), len(hi_grp)) / n
    print(f"  separation (between/within) = {sep:.2f}   >2 clean split | <1 continuous")
    if minor < 0.10:
        print(f"  !! smaller group is only {100*minor:.1f}% of heads -- this is a TAIL, not a mode.")
        print("     Ignore the separation number and read the histogram shape instead.")

    # --- Q2: is a head slot consistent across layers? -------------------
    print("\nQ2 -- per-head-slot consistency across layers:")
    print("  slot |  min    med    max   spread | verdict")
    consistent = 0
    for j in range(n_h):
        col = sorted(grid[i][j] for i in range(n_l))
        spread = col[-1] - col[0]
        verdict = "consistent" if spread < 0.25 else ("varies" if spread < 0.6 else "VARIES WIDELY")
        consistent += spread < 0.25
        print(f"  {j:4d} | {col[0]:.3f}  {col[len(col)//2]:.3f}  {col[-1]:.3f}  {spread:.3f} | {verdict}")
    print(f"\n  {consistent}/{n_h} head slots are consistent across layers")
    print("  most consistent -> a single per-head-index profile works")
    print("  mostly varying  -> allocation must be per (layer, head)")

    # within-layer spread, for contrast with the across-layer number
    ws = [max(r) - min(r) for r in grid]
    print(f"\n  within-layer head spread: min {min(ws):.3f}  mean {sum(ws)/len(ws):.3f}  max {max(ws):.3f}")


if __name__ == "__main__":
    main()

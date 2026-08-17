"""D1 step 3 -- score the KV window sweep for chunk discontinuity.

    python examples/inference/optimizations/kvpolicy_score.py kvsweep_out/*/output.mp4

Measures the failure mode that shrinking the KV window is expected to cause:
the model loses history at autoregressive chunk boundaries, so the seam between
chunks becomes visible as a jump. Forcing-KV (arXiv 2605.09681) scores exactly
this with an optical-flow "chunk discontinuity" metric; this is the cheap
frame-difference version of the same idea -- cv2 + numpy only, no CLIP/VBench
dependency tree to fight on aarch64.

  discontinuity = (mean frame-delta AT chunk boundaries)
                / (mean frame-delta WITHIN chunks)

1.0 means seams are indistinguishable from ordinary motion. Higher means the
boundaries are visibly jumpier than the content around them.

NOT SSIM against the w6 video. Changing the window changes the denoise
trajectory, so frames diverge even when quality is equal -- SSIM-vs-reference
would measure divergence and call it degradation. These are no-reference
metrics, computed per video independently.
"""

import argparse
import sys

import numpy as np


def frame_deltas(path: str) -> np.ndarray:
    import cv2  # lazy: keeps --help and the metric math usable without opencv
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    deltas, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev is not None:
            deltas.append(float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    return np.asarray(deltas)


def score(path: str, chunk: int) -> dict:
    d = frame_deltas(path)
    n = len(d)
    if n < chunk * 3:
        raise SystemExit(f"{path}: only {n+1} frames, too short for chunk={chunk}")

    # d[i] is the delta between frame i and i+1. A chunk boundary sits between
    # the last frame of one chunk and the first of the next.
    idx = np.arange(n)
    is_boundary = ((idx + 1) % chunk) == 0
    boundary, interior = d[is_boundary], d[~is_boundary]

    return {
        "frames": n + 1,
        "mean_delta": d.mean(),
        "boundary": boundary.mean(),
        "interior": interior.mean(),
        "discontinuity": boundary.mean() / max(1e-9, interior.mean()),
        "worst_boundary": boundary.max() / max(1e-9, interior.mean()),
        "delta_std": d.std(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--chunk", type=int, default=12,
                    help="pixel frames per AR chunk: num_frames_per_block(3) x VAE temporal ratio(4)")
    args = ap.parse_args()

    rows = []
    for v in args.videos:
        try:
            rows.append((v, score(v, args.chunk)))
        except SystemExit as e:
            print(f"  !! {e}", file=sys.stderr)

    print(f"\nchunk discontinuity (chunk = {args.chunk} frames)")
    print(f"{'video':<34} {'frames':>7} {'mean_d':>8} {'bound':>8} {'inter':>8} {'RATIO':>7} {'worst':>7}")
    for v, s in rows:
        short = "/".join(v.split("/")[-2:])
        print(f"{short:<34} {s['frames']:>7} {s['mean_delta']:>8.5f} {s['boundary']:>8.5f} "
              f"{s['interior']:>8.5f} {s['discontinuity']:>7.3f} {s['worst_boundary']:>7.2f}")

    if len(rows) > 1:
        ratios = [s["discontinuity"] for _, s in rows]
        print(f"\nspread across videos: {min(ratios):.3f} .. {max(ratios):.3f}  "
              f"(delta {max(ratios)-min(ratios):.3f})")
        print("Two runs of the SAME window are bit-identical, so the noise floor is 0 --")
        print("any spread here is caused by the window, not by run-to-run variance.")


if __name__ == "__main__":
    main()

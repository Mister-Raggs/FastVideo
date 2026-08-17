"""Is draft-init losing on quality, or is the metric measuring the wrong thing?

The sweep scored every arm by cosine distance to Wan's 50-step reference. But
the control starts from the SAME seeded noise as that reference, so it converges
into the same sample; draft-init starts from FastWan's sample, which is a
different video of the same prompt. If so, the score is measuring "did you
reproduce Wan's particular sample", not "is this good", and the control is
structurally advantaged.

This distinguishes the two explanations using only the saved tensors:

  If CONFOUND -- draft-init outputs sit close to the DRAFT and far from the
  reference, and cos(draft, reference) is itself ~0.55-0.6, matching the
  plateau the sweep reported.

  If BUG -- draft-init outputs sit far from BOTH, i.e. the re-noise produced
  something that is neither model's sample.

Run: python specdec_diagnose_basins.py --out ~/specdec
"""

import argparse
import json
import os
from pathlib import Path

import torch


def cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    a32, b32 = a.flatten().float(), b.flatten().float()
    return float(1.0 - torch.nn.functional.cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0), dim=1).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="specdec_out")
    args = ap.parse_args()
    out = Path(os.path.expanduser(args.out))

    draft = torch.load(out / "draft_latent.pt")["latent"]
    reference = torch.load(out / "reference_latent.pt")["latent"]
    arms = json.loads((out / "sweep_arms.json").read_text())

    d_ref = cos_dist(draft, reference)
    print(f"\ncos_dist(draft x0, reference)          = {d_ref:.6f}")
    print("   ^ if this is ~= the draft-init plateau (~0.55-0.59), the sweep was")
    print("     scoring basin identity, not quality.\n")

    print(f"{'strength':>9} {'->reference':>12} {'->draft':>10} {'closer to':>12}")
    rows = []
    for rec in arms:
        s = rec["strength"]
        arm = torch.load(out / f"arm_draftinit_s{s:.2f}.pt")
        to_ref, to_draft = cos_dist(arm, reference), cos_dist(arm, draft)
        rows.append({"strength": s, "to_reference": to_ref, "to_draft": to_draft})
        print(f"{s:>9.2f} {to_ref:>12.6f} {to_draft:>10.6f} {'DRAFT' if to_draft < to_ref else 'reference':>12}")

    near_draft = sum(r["to_draft"] < r["to_reference"] for r in rows)
    print(f"\ndraft-init output is closer to the draft than to the reference in "
          f"{near_draft}/{len(rows)} arms")
    if near_draft >= len(rows) // 2 and abs(d_ref - rows[0]["to_reference"]) < 0.15:
        print("=> CONFOUND: the arms are producing FastWan's sample, refined. The\n"
              "   comparison needs a no-reference quality metric (SDVG used\n"
              "   ImageReward for exactly this reason), not distance to Wan's sample.")
    else:
        print("=> NOT explained by basin identity -- suspect a real bug in the\n"
              "   re-noise path. Check the 'Partial denoise:' log lines fired.")

    (out / "basin_diagnosis.json").write_text(json.dumps({"draft_vs_reference": d_ref, "arms": rows}, indent=2))


if __name__ == "__main__":
    main()

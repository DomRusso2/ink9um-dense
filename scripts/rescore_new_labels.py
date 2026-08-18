"""Re-score existing predictions against a different label version.

Predictions do not depend on labels, so when the team revises the released
masks the whole comparison can be refreshed on CPU in minutes: no inference,
no retraining. Reads the (tag, segment) pairs already in step9_results.jsonl,
re-scores each prediction tif against a second label root, and prints old vs
new side by side.

  python vendor/rescore_new_labels.py --labels data/labels_v20260818/aligned-scrollprizeorg-21slices
"""
import argparse
import json
import os
import sys

import numpy as np
import tifffile
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor"))
from score_ink9um import score  # noqa: E402
from step9_eval import auc_masked  # noqa: E402

PRED = os.path.join(ROOT, "data/step9_preds")
OLD = os.path.join(ROOT, "data/ink9um/labels/aligned-scrollprizeorg-21slices")
REGIONS = ["pherc0139-w016", "pherc0814-46527", "pherc1667-w029"]


def plane(p, z=10):
    g = zarr.open(p, mode="r")
    a = g["0"] if hasattr(g, "array_keys") and "0" in list(g.array_keys()) else g
    return np.asarray(a[z])


def mask_diff(new_root):
    """How much each mask grew between the two label versions."""
    print("=== mask changes, old (2026-08-14) -> corrected (2026-08-18) ===")
    out = {}
    for seg in REGIONS:
        out[seg] = {}
        for kind in ("supervision_mask", "validation_mask", "inklabels"):
            o = os.path.join(OLD, seg, f"{seg}_{kind}.zarr")
            n = os.path.join(new_root, seg, f"{seg}_{kind}.zarr")
            if not (os.path.exists(o) and os.path.exists(n)):
                continue
            a, b = plane(o) > 0, plane(n) > 0
            if a.shape != b.shape:
                print(f"  {seg} {kind}: SHAPE CHANGED {a.shape} -> {b.shape}")
                continue
            gained, lost = int((~a & b).sum()), int((a & ~b).sum())
            out[seg][kind] = {"old_px": int(a.sum()), "new_px": int(b.sum()),
                              "gained": gained, "lost": lost}
            print(f"  {seg:<17} {kind:<17} {int(a.sum()):>9,} -> {int(b.sum()):>9,} px"
                  f"   (+{gained:,} / -{lost:,})")
    return out


def contamination(new_root):
    """THE check that matters: did the corrected validation regions grow into
    pixels our pseudo supervision covered? If so the training runs saw
    validation data and the zero-overlap guarantee no longer holds."""
    print("\n=== contamination check: corrected validation vs our pseudo supervision ===")
    pse = os.path.join(ROOT, "data/pseudo/aligned-scrollprizeorg-21slices")
    worst = 0
    res = {}
    for seg in REGIONS:
        sp = os.path.join(pse, seg, f"{seg}_supervision_mask.zarr")
        vp = os.path.join(new_root, seg, f"{seg}_validation_mask.zarr")
        if not (os.path.exists(sp) and os.path.exists(vp)):
            print(f"  {seg}: not part of the pseudo set, skipped")
            continue
        s, v = plane(sp) > 0, plane(vp) > 0
        ov = int((s & v).sum())
        worst = max(worst, ov)
        res[seg] = {"overlap_px": ov, "validation_px": int(v.sum())}
        flag = "CLEAN" if ov == 0 else f"CONTAMINATED by {ov:,} px"
        print(f"  {seg:<17} overlap {ov:>8,} px of {int(v.sum()):,} validation px   {flag}")
    print(f"  => worst case: {worst:,} px")
    return res, worst


def rescore(new_root):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/step9_results.jsonl")) if l.strip()]
    latest = {}
    for r in rows:
        latest[(r["tag"], r["segment"])] = r
    out = []
    print("\n=== re-scoring against corrected labels ===")
    for (tag, seg), r in sorted(latest.items()):
        if seg not in REGIONS:
            continue
        tif = os.path.join(PRED, f"{r['tag']}__{seg}.tif")
        if not os.path.exists(tif):
            continue
        d = os.path.join(new_root, seg)
        ink = os.path.join(d, f"{seg}_inklabels.zarr")
        rec = {"tag": tag, "segment": seg}
        for kind in ("supervision_mask", "validation_mask"):
            mp = os.path.join(d, f"{seg}_{kind}.zarr")
            if not os.path.exists(mp):
                continue
            s = score(tif, ink, mp)
            s["auc"] = auc_masked(tif, ink, mp, s["z"])
            rec[kind] = s
        out.append(rec)
        v = rec.get("validation_mask")
        ov = r.get("validation_mask")
        if v and ov:
            print(f"  {tag:<22} {seg:<17} val {ov['balanced_accuracy_THEIRS']:.4f}"
                  f" -> {v['balanced_accuracy_THEIRS']:.4f}"
                  f"  ({v['balanced_accuracy_THEIRS']-ov['balanced_accuracy_THEIRS']:+.4f})")
    p = os.path.join(ROOT, "data/step9_results_v20260818.jsonl")
    with open(p, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {os.path.relpath(p, ROOT)}  ({len(out)} records)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    a = ap.parse_args()
    root = os.path.join(ROOT, a.labels)
    md = mask_diff(root)
    cres, worst = contamination(root)
    rescore(root)
    with open(os.path.join(ROOT, "data/label_version_diff.json"), "w") as fh:
        json.dump({"mask_diff": md, "contamination": cres, "worst_overlap_px": worst}, fh, indent=1)
    print("\nwrote data/label_version_diff.json")

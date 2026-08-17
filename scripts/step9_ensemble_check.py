"""Diagnostic (zero GPU): what is the strongest available TEACHER?

STEP 9 showed the released family's best beats the banked 2.4 um teacher on 2 of
3 held-out regions, which undercuts the premise of the distillation plan. Before
spending ~10 h on STEP 8B it is worth knowing whether some combination of
already-computed predictions is a better teacher than any single model.

Uses only prediction tifs already on disk (data/step9_preds + data/teacher), so
it costs nothing but a few minutes of CPU.

  python vendor/step9_ensemble_check.py
"""
import glob
import json
import os
import sys

import numpy as np
import tifffile
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor"))
LABELS = os.path.join(ROOT, "data/ink9um/labels/aligned-scrollprizeorg-21slices")

REGIONS = ["pherc0139-w016", "pherc0814-46527", "pherc1667-w029"]
TEACHER = {
    "pherc0139-w016": "data/teacher/w016_teacher_fwd_9um.tif",
    "pherc0814-46527": "data/teacher/pherc0814-46527_teacher_fwd_9um.tif",
    "pherc1667-w029": "data/teacher/pherc1667-w029_teacher_fwd_9um.tif",
}
TSTAR = {"pherc0139-w016": 0.15, "pherc0814-46527": 0.53, "pherc1667-w029": 0.36}


def plane(path, z):
    a = zarr.open(path, mode="r")
    arr = a["0"] if hasattr(a, "array_keys") and "0" in list(a.array_keys()) else a
    return np.asarray(arr[z])


def auc(p, gt):
    n_pos, n_neg = int(gt.sum()), int((~gt).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    sv = p[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[gt].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bal_acc(p, gt, t):
    pb = p >= t
    tpr = float((pb & gt).sum()) / max(1, int(gt.sum()))
    tnr = float((~pb & ~gt).sum()) / max(1, int((~gt).sum()))
    return 0.5 * (tpr + tnr)


def best_t(p, gt):
    ts = np.arange(0.05, 0.96, 0.01)
    scores = [bal_acc(p, gt, t) for t in ts]
    i = int(np.argmax(scores))
    return float(ts[i]), float(scores[i])


def main():
    out = {}
    for seg in REGIONS:
        d = os.path.join(LABELS, seg)
        z = 10
        gt_full = plane(os.path.join(d, f"{seg}_inklabels.zarr"), z) > 0
        val_full = plane(os.path.join(d, f"{seg}_validation_mask.zarr"), z) > 0
        sup_full = plane(os.path.join(d, f"{seg}_supervision_mask.zarr"), z) > 0
        gt_v, gt_s = gt_full[val_full], gt_full[sup_full]

        print(f"\n=== {seg}  (val px {val_full.sum():,}) ===")
        rows = {}

        def add(name, arr_full):
            pv = arr_full[val_full].astype(np.float32) / 255.0
            ps = arr_full[sup_full].astype(np.float32) / 255.0
            # threshold calibrated ONLY on the supervision region, then applied to val
            t_cal, _ = best_t(ps, gt_s)
            rows[name] = {
                "val_auc": auc(pv, gt_v),
                "val_ba@0.5": bal_acc(pv, gt_v, 0.5),
                "val_ba@t_cal": bal_acc(pv, gt_v, t_cal),
                "t_cal": t_cal,
            }
            r = rows[name]
            print(f"  {name:<26} AUC {r['val_auc']:.4f}   ba@0.5 {r['val_ba@0.5']:.4f}"
                  f"   ba@t_cal {r['val_ba@t_cal']:.4f} (t={t_cal:.2f})")

        # individual released checkpoints -> ensembles
        preds = {}
        for f in sorted(glob.glob(os.path.join(ROOT, f"data/step9_preds/released_*__{seg}.tif"))):
            tag = os.path.basename(f).split("__")[0]
            preds[tag] = tifffile.imread(f)
        if preds:
            all14 = np.mean([p.astype(np.float32) for p in preds.values()], axis=0)
            add("ensemble_released_14", all14)
            s43 = [p.astype(np.float32) for t, p in preds.items() if "_s43_" in t]
            if s43:
                add("ensemble_released_s43_7", np.mean(s43, axis=0))
            late = [p.astype(np.float32) for t, p in preds.items()
                    if t.split("_")[-1] in ("040000", "050000", "060000", "075000")]
            if late:
                add("ensemble_released_late_8", np.mean(late, axis=0))

        tp = os.path.join(ROOT, TEACHER[seg])
        teacher = tifffile.imread(tp).astype(np.float32)
        if teacher.shape != gt_full.shape:
            teacher = teacher[:gt_full.shape[0], :gt_full.shape[1]]
        add("teacher_2p4um", teacher)

        if preds:
            # rank-normalise both before mixing: the teacher's probability scale is
            # not the students' (t* 0.15-0.53 vs 0.5)
            def rank_norm(a):
                flat = a.ravel()
                order = np.argsort(flat, kind="mergesort")
                r = np.empty(len(flat), dtype=np.float32)
                r[order] = np.linspace(0, 255, len(flat), dtype=np.float32)
                return r.reshape(a.shape)
            mix = 0.5 * rank_norm(all14) + 0.5 * rank_norm(teacher)
            add("teacher+ens14_rank_mix", mix)

        out[seg] = rows

    with open(os.path.join(ROOT, "data/step9_ensemble_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote data/step9_ensemble_check.json")


if __name__ == "__main__":
    main()

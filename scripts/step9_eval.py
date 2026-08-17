"""STEP 9 eval driver (HANDOFF.md).

For each (checkpoint, validation region): run the team's inference CLI with the
exact contract HANDOFF specifies, then score with vendor/score_ink9um.py -- once
against the supervision_mask and once against the validation_mask -- and add an
AUC computed over the same masked pixels.

Resumable: an existing prediction tif is reused, and results are appended to a
JSONL so an interrupted sweep continues where it stopped. Nothing here touches
the GPU except the inference subprocess, one at a time.

Usage:
  python vendor/step9_eval.py --ckpt data/train_baseline/ckpt_078125.pth --tag baseline_final
  python vendor/step9_eval.py --released                 # all 14 released checkpoints
  python vendor/step9_eval.py --ckpt A.pth --ckpt B.pth --tag ours
  python vendor/step9_eval.py --report                   # print the table only
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor"))

PYTHON = os.path.join(ROOT, ".venv/Scripts/python.exe")
PYPATH = (f"{ROOT}/vendor/villa-ink/ink-detection;{ROOT}/vendor/villa-ink/vesuvius/src"
          .replace("/", os.sep))
LABELS = os.path.join(ROOT, "data/ink9um/labels/aligned-scrollprizeorg-21slices")
PRED_DIR = os.path.join(ROOT, "data/step9_preds")
RESULTS = os.path.join(ROOT, "data/step9_results.jsonl")

# The three held-out validation regions -- the only genuinely held-out evidence
# in the release (memory vesuvius-05).
REGIONS = {
    "pherc0139-w016": "data/ink9um/volumes_w016/w016_9um_iso.zarr",
    "pherc0814-46527": "data/teacher/p0814_iso9.zarr",
    "pherc1667-w029": "data/teacher/p1667w029_iso9.zarr",
}


def auc_masked(pred_tif, inklabels, mask_zarr, z):
    """Rank-based AUC (Mann-Whitney U) over the masked pixels. No sklearn here --
    installing it risks the torch pin (memory vesuvius-03)."""
    import tifffile
    import zarr

    def plane(path):
        a = zarr.open(path, mode="r")
        arr = a["0"] if hasattr(a, "array_keys") and "0" in list(a.array_keys()) else a
        return np.asarray(arr[z])

    p = tifffile.imread(pred_tif).astype(np.float32)
    gt = plane(inklabels) > 0
    valid = plane(mask_zarr) > 0
    pv, gv = p[valid], gt[valid]
    n_pos, n_neg = int(gv.sum()), int((~gv).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(pv, kind="mergesort")
    ranks = np.empty(len(pv), dtype=np.float64)
    ranks[order] = np.arange(1, len(pv) + 1)
    # average ranks within ties
    sv = pv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[gv].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def run_inference(ckpt, seg, out_tif, log_path):
    """HANDOFF STEP 9 command, verbatim."""
    env = dict(os.environ, PYTHONPATH=PYPATH, WANDB_MODE="disabled")
    cmd = [PYTHON, "-u", "-m", "koine_machines.inference.infer",
           os.path.join(ROOT, REGIONS[seg]), ckpt, out_tif,
           "--overlap", "0.5", "--blend-mode", "hann",
           "--batch-size", "8", "--no-compile"]
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(" ".join(cmd) + "\n\n")
        lf.flush()
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
    return rc, time.time() - t0


def evaluate(ckpt, tag, regions, force=False):
    from score_ink9um import score

    os.makedirs(PRED_DIR, exist_ok=True)
    rows = []
    for seg in regions:
        name = f"{tag}__{seg}"
        out_tif = os.path.join(PRED_DIR, f"{name}.tif")
        log_path = os.path.join(ROOT, "data", f"step9_infer_{name}.log")
        if force or not os.path.exists(out_tif):
            print(f"  [infer] {name} ...", flush=True)
            rc, dt = run_inference(ckpt, seg, out_tif, log_path)
            if rc != 0 or not os.path.exists(out_tif):
                print(f"  FAILED rc={rc}; see {os.path.relpath(log_path, ROOT)}")
                continue
            print(f"  [infer] {name} done in {dt/60:.1f} min", flush=True)
        else:
            print(f"  [infer] {name} reusing existing prediction", flush=True)

        d = os.path.join(LABELS, seg)
        ink = os.path.join(d, f"{seg}_inklabels.zarr")
        row = {"tag": tag, "checkpoint": os.path.relpath(ckpt, ROOT), "segment": seg}
        for kind in ("supervision_mask", "validation_mask"):
            mask = os.path.join(d, f"{seg}_{kind}.zarr")
            if not os.path.exists(mask):
                continue
            s = score(out_tif, ink, mask)
            s["auc"] = auc_masked(out_tif, ink, mask, s["z"])
            row[kind] = s
            print(f"    {kind:<16} bal_acc {s['balanced_accuracy_THEIRS']:.4f}"
                  f"  AUC {s['auc']:.4f}  TPR {s['TPR']:.4f} TNR {s['TNR']:.4f}"
                  f"  px {s['supervised_px']:,}", flush=True)
        rows.append(row)
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    return rows


def report():
    if not os.path.exists(RESULTS):
        print("no results yet")
        return
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    latest = {}
    for r in rows:  # last write per (tag, segment) wins
        latest[(r["tag"], r["segment"])] = r
    tags = sorted({t for t, _ in latest})
    segs = list(REGIONS)
    print(f"\n{'':<28}" + "".join(f"{s:>26}" for s in segs))
    print(f"{'checkpoint':<28}" + "".join(f"{'val bal_acc / AUC':>26}" for _ in segs))
    print("-" * (28 + 26 * len(segs)))
    for t in tags:
        line = f"{t:<28}"
        for s in segs:
            r = latest.get((t, s))
            v = r.get("validation_mask") if r else None
            line += (f"{v['balanced_accuracy_THEIRS']:>18.4f} / {v['auc']:.4f}"
                     if v else f"{'-':>26}")
        print(line)
    rel = [t for t in tags if t.startswith("released_")]
    if rel:
        print("\nreleased bar (MAX over released checkpoints) per region:")
        for s in segs:
            vals = [(latest[(t, s)]["validation_mask"]["balanced_accuracy_THEIRS"], t)
                    for t in rel if (t, s) in latest and "validation_mask" in latest[(t, s)]]
            if vals:
                best = max(vals)
                spread = max(v for v, _ in vals) - min(v for v, _ in vals)
                print(f"  {s:<18} max {best[0]:.4f} ({best[1]})   spread over"
                      f" {len(vals)} ckpts = {spread:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--released", action="store_true", help="evaluate all 14 released checkpoints")
    ap.add_argument("--region", action="append", default=[], help="limit to these regions")
    ap.add_argument("--force", action="store_true", help="re-run inference even if the tif exists")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        report()
        return

    regions = args.region or list(REGIONS)
    jobs = []
    if args.released:
        base = os.path.join(ROOT, "models/ink9um/released")
        for seed in ("42", "43"):
            for step in ("010000", "020000", "030000", "040000", "050000", "060000", "075000"):
                p = os.path.join(base, f"hybrid_3d2d-seed{seed}", f"step-{step}.pth")
                if os.path.exists(p):
                    jobs.append((p, f"released_s{seed}_{step}"))
    for i, c in enumerate(args.ckpt):
        tag = args.tag if args.tag and len(args.ckpt) == 1 else f"{args.tag or 'ours'}_{i}"
        jobs.append((os.path.abspath(c), tag))

    if not jobs:
        print("nothing to do")
        return
    print(f"{len(jobs)} checkpoint(s) x {len(regions)} region(s)"
          f" = {len(jobs)*len(regions)} inference runs")
    for ckpt, tag in jobs:
        print(f"\n=== {tag}  ({os.path.relpath(ckpt, ROOT)}) ===", flush=True)
        evaluate(ckpt, tag, regions, force=args.force)
    report()


if __name__ == "__main__":
    main()

"""Full-segment 9 um inference on Scroll 1667, ours vs the best released checkpoint.

Answers the team's question: the validation crops are too small to judge how the
models behave over a whole segment. For each 1667 segment this produces

  <seg>_full.png    both models over the entire canvas, side by side, downsampled
  <seg>_detail.png  the densest text window at native 9.6 um resolution

Both models get identical treatment: same inference flags, same display rescale
(p-0.25)/0.5, same downsample factor, same crop coordinates.

  python vendor/full_segment_1667.py                # all six segments
  python vendor/full_segment_1667.py --only pherc1667-w029
  python vendor/full_segment_1667.py --render-only  # skip inference
"""
import argparse
import os
import subprocess
import time

import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, ".venv/Scripts/python.exe")
PYPATH = (f"{ROOT}/vendor/villa-ink/ink-detection;{ROOT}/vendor/villa-ink/vesuvius/src"
          .replace("/", os.sep))
PRED = os.path.join(ROOT, "data/full1667_preds")
OUT = os.path.join(ROOT, "release/ink9um-dense/figures/full_segments_1667")

VOLUMES = {
    "pherc1667-w013": "data/corpus/pherc1667-w013_iso9.zarr",
    "pherc1667-w018": "data/ink9um/volumes_pherc1667-w018/w018_9um_iso.zarr",
    "pherc1667-w023": "data/corpus/pherc1667-w023_iso9.zarr",
    "pherc1667-w028": "data/teacher/p1667w028_iso9.zarr",
    "pherc1667-w029": "data/teacher/p1667w029_iso9.zarr",
    "pherc1667-w031": "data/corpus/pherc1667-w031_iso9.zarr",
}
# ours, and the released checkpoint that scores best on 1667-w029 (s43 step-060000)
MODELS = [
    ("released_s43_060000", "models/ink9um/released/hybrid_3d2d-seed43/step-060000.pth",
     "released seed43 step-060000 (best released on 1667)"),
    ("dense_w016excl", "release/ink9um-dense/checkpoints/dense9um-w016excluded-step075000.pth",
     "dense pseudo-labels (ours)"),
]
# predictions already computed by the STEP 9 sweep, reused rather than recomputed
EXISTING = {
    ("released_s43_060000", "pherc1667-w029"): "data/step9_preds/released_s43_060000__pherc1667-w029.tif",
    ("dense_w016excl", "pherc1667-w029"): "data/step9_preds/diag_final__pherc1667-w029.tif",
}


def pred_path(tag, seg):
    if (tag, seg) in EXISTING:
        p = os.path.join(ROOT, EXISTING[(tag, seg)])
        if os.path.exists(p):
            return p
    return os.path.join(PRED, f"{tag}__{seg}.tif")


def infer(tag, ckpt, seg):
    out = pred_path(tag, seg)
    if os.path.exists(out):
        print(f"  [skip] {tag} {seg} already exists")
        return out
    os.makedirs(PRED, exist_ok=True)
    log = os.path.join(ROOT, "data", f"full1667_{tag}__{seg}.log")
    cmd = [PYTHON, "-u", "-m", "koine_machines.inference.infer",
           os.path.join(ROOT, VOLUMES[seg]), os.path.join(ROOT, ckpt), out,
           "--overlap", "0.5", "--blend-mode", "hann", "--batch-size", "8", "--no-compile"]
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT,
                             env=dict(os.environ, PYTHONPATH=PYPATH, WANDB_MODE="disabled"))
    if rc != 0:
        print(f"  FAILED {tag} {seg} rc={rc}, see {os.path.relpath(log, ROOT)}")
        return None
    print(f"  [infer] {tag} {seg} done in {(time.time()-t0)/60:.1f} min")
    return out


def resc(a):
    return np.clip((a.astype(np.float32) / 255.0 - 0.25) / 0.5, 0, 1)


def densest_window(p, size=1100):
    """Densest predicted-ink window on a coarse grid.

    `p` must be MODEL-NEUTRAL (the mean of both predictions). Selecting on one
    model's own density picks the place that model is most confident, which is
    exactly where it over-predicts, and hands it an unfair panel. Also avoids
    the centroid trap: a sparse pattern's centroid can land on empty canvas.
    """
    s = 16
    m = (p[::s, ::s] > 128).astype(np.float32)
    k = max(4, size // s)
    if m.shape[0] <= k or m.shape[1] <= k:
        return 0, 0
    best, by, bx = -1.0, 0, 0
    for y in range(0, m.shape[0] - k, max(1, k // 4)):
        for x in range(0, m.shape[1] - k, max(1, k // 4)):
            t = float(m[y:y + k, x:x + k].sum())
            if t > best:
                best, by, bx = t, y * s, x * s
    return by, bx


def render(seg, paths):
    imgs = []
    arrs = {}
    for tag, ckpt, label in MODELS:
        p = paths.get(tag)
        if not p or not os.path.exists(p):
            continue
        a = tifffile.imread(p)
        arrs[tag] = a
        imgs.append((label, a))
    if len(imgs) < 2:
        print(f"  {seg}: fewer than two predictions, skipping render")
        return

    h, w = imgs[0][1].shape
    # full canvas, both models side by side
    target_w = 1250
    f = max(1, int(np.ceil(w / target_w)))
    panels = []
    for label, a in imgs:
        small = resc(a[::f, ::f])
        im = Image.fromarray((small * 255).astype(np.uint8)).convert("RGB")
        panels.append((label, im))
    W = sum(i.width for _, i in panels) + 16
    H = max(i.height for _, i in panels) + 28
    sheet = Image.new("RGB", (W, H), "white")
    x = 0
    for label, im in panels:
        sheet.paste(im, (x, 28))
        ImageDraw.Draw(sheet).text((x + 4, 8), label, fill="black")
        x += im.width + 16
    ImageDraw.Draw(sheet).text((4, H - 14), f"{seg}  {w}x{h} px at 9.6 um, shown at 1/{f}",
                               fill="black")
    p1 = os.path.join(OUT, f"{seg}_full.png")
    sheet.save(p1)
    print(f"  wrote {os.path.relpath(p1, ROOT)}  ({W}x{H})")

    # native-resolution detail, window chosen on the MEAN of both predictions so
    # neither model picks its own best ground
    neutral = np.mean([a.astype(np.float32) for _, a in imgs], axis=0)
    by, bx = densest_window(neutral)
    size = 1100
    by, bx = min(by, h - size), min(bx, w - size)
    dpanels = []
    g = zarr.open(os.path.join(ROOT, VOLUMES[seg]), mode="r")
    vol = g["0"] if hasattr(g, "array_keys") and "0" in list(g.array_keys()) else g
    ct = np.asarray(vol[10, by:by + size, bx:bx + size]).astype(np.float32)
    ct = ct / max(1.0, float(ct.max()))
    dpanels.append(("9.6 um surface volume (input)",
                    Image.fromarray((ct * 255).astype(np.uint8)).convert("RGB")))
    for label, a in imgs:
        crop = resc(a[by:by + size, bx:bx + size])
        dpanels.append((label, Image.fromarray((crop * 255).astype(np.uint8)).convert("RGB")))
    W = sum(i.width for _, i in dpanels) + 16
    H = max(i.height for _, i in dpanels) + 28
    sheet = Image.new("RGB", (W, H), "white")
    x = 0
    for label, im in dpanels:
        sheet.paste(im, (x, 28))
        ImageDraw.Draw(sheet).text((x + 4, 8), label, fill="black")
        x += im.width + 16
    ImageDraw.Draw(sheet).text((4, H - 14),
                               f"{seg}  native 9.6 um, {size}x{size} px at y={by} x={bx}",
                               fill="black")
    p2 = os.path.join(OUT, f"{seg}_detail.png")
    sheet.save(p2)
    print(f"  wrote {os.path.relpath(p2, ROOT)}  ({W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    segs = args.only or list(VOLUMES)
    for seg in segs:
        print(f"\n=== {seg} ===", flush=True)
        paths = {}
        for tag, ckpt, _ in MODELS:
            paths[tag] = pred_path(tag, seg) if args.render_only else infer(tag, ckpt, seg)
        render(seg, paths)


if __name__ == "__main__":
    main()

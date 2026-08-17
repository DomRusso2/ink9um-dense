"""STEP 9 visual check (HANDOFF: mandatory before any claim).

For each held-out validation region, renders one side-by-side panel:

    best released checkpoint | our baseline (control) | our pseudo model | manual labels

cropped to the validation region, with HANDOFF's display rescale (p-0.25)/0.5
(the released models are trained with bce_label_smoothing 0.5, so confident
no-ink sits at 0.25, not 0).

Also answers the team's named failure mode -- "you gain clearness but lose
fainter signals" -- quantitatively: validation ink pixels are split into
difficulty quintiles by how confident the best released checkpoint is, and each
model's recall is reported per quintile. If our model wins overall but loses on
the faintest quintile, that shows up here instead of being hidden by the mean.

  python vendor/step9_figures.py
"""
import json
import os

import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(ROOT, "data/ink9um/labels/aligned-scrollprizeorg-21slices")
PREDS = os.path.join(ROOT, "data/step9_preds")
OUT = os.path.join(ROOT, "data/step9_figures")
RESULTS = os.path.join(ROOT, "data/step9_results.jsonl")
Z = 10
REGIONS = ["pherc0139-w016", "pherc0814-46527", "pherc1667-w029"]


def plane(path, z=Z):
    g = zarr.open(path, mode="r")
    a = g["0"] if hasattr(g, "array_keys") and "0" in list(g.array_keys()) else g
    return np.asarray(a[z])


def rescale(p):
    """HANDOFF display rescale: (p-0.25)/0.5, clipped."""
    return np.clip((p.astype(np.float32) / 255.0 - 0.25) / 0.5, 0, 1)


def best_released_per_region():
    rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
    latest = {}
    for r in rows:
        latest[(r["tag"], r["segment"])] = r
    out = {}
    for seg in REGIONS:
        cands = [(v["validation_mask"]["balanced_accuracy_THEIRS"], t)
                 for (t, s), v in latest.items()
                 if s == seg and t.startswith("released_") and "validation_mask" in v]
        if cands:
            out[seg] = max(cands)
    return out


def bbox_of(mask, pad=64):
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    return y0, y1, x0, x1


def to_img(a, max_w=900):
    im = Image.fromarray((a * 255).astype(np.uint8))
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    return im.convert("RGB")


def main():
    os.makedirs(OUT, exist_ok=True)
    best_rel = best_released_per_region()
    report = {}

    for seg in REGIONS:
        d = os.path.join(LABELS, seg)
        gt = plane(os.path.join(d, f"{seg}_inklabels.zarr")) > 0
        val = plane(os.path.join(d, f"{seg}_validation_mask.zarr")) > 0
        y0, y1, x0, x1 = bbox_of(val)

        rel_tag = best_rel[seg][1]
        panels = [
            (f"best released\n{rel_tag}", f"{rel_tag}__{seg}.tif"),
            ("our baseline (control)", f"baseline_final__{seg}.tif"),
            ("dense all-7 model", f"pseudo_final__{seg}.tif"),
            ("dense w016-excluded (shipped)", f"diag_final__{seg}.tif"),
        ]
        imgs, arrays = [], {}
        for title, fname in panels:
            p = os.path.join(PREDS, fname)
            if not os.path.exists(p):
                print(f"  missing {fname}, skipping panel")
                continue
            a = tifffile.imread(p)
            arrays[title] = a
            crop = rescale(a[y0:y1, x0:x1])
            crop[~val[y0:y1, x0:x1]] *= 0.25   # dim outside the scored region
            imgs.append((title, to_img(crop)))
        lab = gt[y0:y1, x0:x1].astype(np.float32)
        imgs.append(("manual ink labels", to_img(lab)))

        w = sum(i.width for _, i in imgs) + 12 * (len(imgs) - 1)
        h = max(i.height for _, i in imgs) + 26
        sheet = Image.new("RGB", (w, h), "white")
        xoff = 0
        for title, im in imgs:
            sheet.paste(im, (xoff, 26))
            ImageDraw.Draw(sheet).text((xoff + 4, 6), title.replace("\n", " "), fill="black")
            xoff += im.width + 12
        path = os.path.join(OUT, f"{seg}_validation_panels.png")
        sheet.save(path)
        print(f"{seg}: wrote {os.path.relpath(path, ROOT)}  crop {y1-y0}x{x1-x0}")

        # faint-stroke check: quintiles of difficulty by best-released confidence
        rel_a = arrays.get(f"best released\n{rel_tag}")
        if rel_a is None:
            continue
        ink_v = gt & val
        conf = rel_a[ink_v].astype(np.float32) / 255.0
        qs = np.quantile(conf, [0, .2, .4, .6, .8, 1.0])
        rows = {}
        for title, a in arrays.items():
            pv = a[ink_v].astype(np.float32) / 255.0 >= 0.5
            rec = []
            for i in range(5):
                m = (conf >= qs[i]) & (conf <= qs[i + 1] if i == 4 else conf < qs[i + 1])
                rec.append(float(pv[m].mean()) if m.sum() else float("nan"))
            rows[title.replace("\n", " ")] = rec
        report[seg] = {"quintile_edges": [float(q) for q in qs], "recall_by_quintile": rows}
        print(f"  recall on validation ink, by difficulty (Q1 = faintest per the released model):")
        print(f"    {'model':<34}" + "".join(f"{'Q'+str(i+1):>8}" for i in range(5)))
        for k, v in rows.items():
            print(f"    {k:<34}" + "".join(f"{x:>8.3f}" for x in v))

    with open(os.path.join(OUT, "faint_stroke_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}/faint_stroke_report.json")


if __name__ == "__main__":
    main()

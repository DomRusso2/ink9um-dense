"""Build every figure for the release (HANDOFF picture list + the checkpoint spread plot).

  fig1  released checkpoint spread, all 14 + our models, per region   [the lead]
  fig2  w016 metric-vs-legibility, four models + manual labels
  fig3  0814 and w029 validation panels
  fig4  dense vs sparse supervision on w016
  fig5  teacher quality, four panels on a supervision crop
  fig6  render fidelity, our render vs the official training input

  python vendor/build_release_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(ROOT, "data/ink9um/labels/aligned-scrollprizeorg-21slices")
PSE = os.path.join(ROOT, "data/pseudo/aligned-scrollprizeorg-21slices")
PRED = os.path.join(ROOT, "data/step9_preds")
OUT = os.path.join(ROOT, "release/ink9um-dense/figures")
REGIONS = ["pherc0139-w016", "pherc0814-46527", "pherc1667-w029"]
SHORT = {"pherc0139-w016": "PHerc0139 w016", "pherc0814-46527": "PHerc0814 46527",
         "pherc1667-w029": "PHerc1667 w029"}


def plane(p, z=10):
    g = zarr.open(p, mode="r")
    a = g["0"] if hasattr(g, "array_keys") and "0" in list(g.array_keys()) else g
    return np.asarray(a[z])


def resc(p):
    return np.clip((p.astype(np.float32) / 255.0 - 0.25) / 0.5, 0, 1)


def results():
    L = {}
    for line in open(os.path.join(ROOT, "data/step9_results.jsonl")):
        if line.strip():
            r = json.loads(line)
            L[(r["tag"], r["segment"])] = r
    return L


def val(L, tag, seg, key="balanced_accuracy_THEIRS"):
    r = L.get((tag, seg))
    return r["validation_mask"][key] if r and "validation_mask" in r else None


def strip(im, w=1200):
    im = Image.fromarray((im * 255).astype(np.uint8))
    if im.width != w:
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    return im.convert("RGB")


def stack(rows, path, w=1200):
    ims = [(t, strip(a, w)) for t, a in rows]
    H = sum(i.height + 24 for _, i in ims)
    sh = Image.new("RGB", (w, H), "white")
    y = 0
    for t, im in ims:
        ImageDraw.Draw(sh).text((6, y + 6), t, fill="black")
        sh.paste(im, (0, y + 24))
        y += im.height + 24
    sh.save(path)
    print("wrote", os.path.relpath(path, ROOT), sh.size)


def bbox(mask, pad=64):
    ys, xs = np.where(mask)
    return (max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad),
            max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad))


def fig1(L):
    steps = ["010000", "020000", "030000", "040000", "050000", "060000", "075000"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    ours = [("control (manual labels only)", "baseline_final", "o", "#888888"),
            ("dense, w016 excluded  [shipped]", "diag_final", "*", "#0b6e4f"),
            ("dense, all 7 segments", "pseudo_final", "D", "#b3541e")]
    for ax, seg in zip(axes, REGIONS):
        r42 = [val(L, f"released_s42_{s}", seg) for s in steps]
        r43 = [val(L, f"released_s43_{s}", seg) for s in steps]
        ax.scatter(r42, [1] * 7, s=42, c="#4a6fa5", label="released seed42 (7 steps)", zorder=3)
        ax.scatter(r43, [1] * 7, s=42, c="#8fb3e0", marker="s",
                   label="released seed43 (7 steps)", zorder=3)
        bar = max([x for x in r42 + r43 if x is not None])
        ax.axvline(bar, color="#4a6fa5", ls="--", lw=1, zorder=1)
        for i, (lbl, tag, mk, col) in enumerate(ours):
            v = val(L, tag, seg)
            if v is not None:
                ax.scatter([v], [1.9 + i * 0.42], s=110 if mk == "*" else 60, marker=mk,
                           c=col, label=lbl if seg == REGIONS[0] else None, zorder=4)
        ax.set_yticks([])
        ax.set_ylim(0.5, 3.3)
        ax.set_ylabel(SHORT[seg], rotation=0, ha="right", va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        ax.text(bar, 0.62, f" released best {bar:.3f}", fontsize=7, color="#4a6fa5")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=8, loc="lower center", ncol=3, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.07))
    axes[-1].set_xlabel("balanced accuracy on the held-out validation region (threshold 0.5)")
    axes[0].set_title("The 14 released 9 um checkpoints vary by up to 0.14 on held-out regions.\n"
                      "No metrics were published for any of them.", fontsize=10, loc="left")
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_released_checkpoint_spread.png")
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def fig2(L):
    seg = "pherc0139-w016"
    gt = plane(f"{LAB}/{seg}/{seg}_inklabels.zarr") > 0
    v = plane(f"{LAB}/{seg}/{seg}_validation_mask.zarr") > 0
    y0, y1, x0, x1 = bbox(v)
    rows = []
    for lbl, tag in [("best released checkpoint (s43 step-060000)", "released_s43_060000"),
                     ("control, manual labels only", "baseline_final"),
                     ("dense pseudo-labels, all 7 segments", "pseudo_final"),
                     ("dense pseudo-labels, w016 excluded  [shipped]", "diag_final")]:
        a = tifffile.imread(f"{PRED}/{tag}__{seg}.tif")
        c = resc(a[y0:y1, x0:x1])
        c[~v[y0:y1, x0:x1]] *= 0.25
        ba = val(L, tag, seg)
        auc = val(L, tag, seg, "auc")
        rows.append((f"{lbl}    balanced accuracy {ba:.4f}   AUC {auc:.4f}", c))
    rows.append(("manual ink labels (ground truth)", gt[y0:y1, x0:x1].astype(np.float32)))
    stack(rows, os.path.join(OUT, "fig2_w016_metric_vs_legibility.png"))


def fig3(L):
    for seg in ["pherc0814-46527", "pherc1667-w029"]:
        gt = plane(f"{LAB}/{seg}/{seg}_inklabels.zarr") > 0
        v = plane(f"{LAB}/{seg}/{seg}_validation_mask.zarr") > 0
        y0, y1, x0, x1 = bbox(v)
        relbest = max((val(L, f"released_s{s}_{st}", seg), f"s{s}_{st}")
                      for s in (42, 43)
                      for st in ("010000", "020000", "030000", "040000", "050000",
                                 "060000", "075000"))
        rows = []
        for lbl, tag in [(f"best released checkpoint ({relbest[1]})",
                          f"released_{relbest[1][:3]}_{relbest[1][4:]}"),
                         ("control, manual labels only", "baseline_final"),
                         ("dense pseudo-labels, w016 excluded  [shipped]", "diag_final")]:
            a = tifffile.imread(f"{PRED}/{tag}__{seg}.tif")
            c = resc(a[y0:y1, x0:x1])
            c[~v[y0:y1, x0:x1]] *= 0.25
            rows.append((f"{lbl}    balanced accuracy {val(L, tag, seg):.4f}", c))
        rows.append(("manual ink labels (ground truth)", gt[y0:y1, x0:x1].astype(np.float32)))
        stack(rows, os.path.join(OUT, f"fig3_{seg}_validation_panels.png"))


def fig4():
    seg = "pherc0139-w016"
    man = plane(f"{LAB}/{seg}/{seg}_supervision_mask.zarr") > 0
    pse = plane(f"{PSE}/{seg}/{seg}_supervision_mask.zarr") > 0
    sub = (slice(None, None, 4), slice(None, None, 4))
    m, p = man[sub], pse[sub]
    rgb_m = np.stack([m * 0.9, m * 0.2, m * 0.2], -1) + 0.08
    rgb_p = np.stack([p * 0.15, p * 0.75, p * 0.5], -1) + 0.08
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(np.clip(rgb_m, 0, 1))
    ax[0].set_title(f"manual supervision: {100*man.mean():.2f}% of canvas\n"
                    f"{man.sum():,} labelled pixels", fontsize=9)
    ax[1].imshow(np.clip(rgb_p, 0, 1))
    ax[1].set_title(f"teacher pseudo-labels: {100*pse.mean():.2f}% of canvas\n"
                    f"{pse.sum():,} labelled pixels ({pse.sum()/man.sum():.0f}x)", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("PHerc0139 w016, the same segment. Held-out validation region is excluded "
                 "from both.", fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_dense_vs_sparse_supervision.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def fig5():
    seg = "pherc0139-w016"
    sup = plane(f"{LAB}/{seg}/{seg}_supervision_mask.zarr") > 0
    gt = plane(f"{LAB}/{seg}/{seg}_inklabels.zarr") > 0
    # densest supervision window by box-sum, never the centroid (memory: sparse
    # ring masks have empty centroids)
    s = sup[::8, ::8].astype(np.float32)
    k = 40
    best, by, bx = -1, 0, 0
    for yy in range(0, s.shape[0] - k, 8):
        for xx in range(0, s.shape[1] - k, 8):
            t = s[yy:yy + k, xx:xx + k].sum()
            if t > best:
                best, by, bx = t, yy * 8, xx * 8
    y0, y1, x0, x1 = by, by + k * 8, bx, bx + k * 8
    ct = plane(os.path.join(ROOT, "data/ink9um/volumes_w016/w016_9um_iso.zarr"))[y0:y1, x0:x1]
    te = tifffile.imread(os.path.join(ROOT, "data/teacher/w016_teacher_fwd_9um.tif"))[y0:y1, x0:x1]
    st = tifffile.imread(f"{PRED}/released_s42_075000__{seg}.tif")[y0:y1, x0:x1]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.9))
    for a, img, t in zip(ax,
                         [ct / max(1, ct.max()), te / 255.0, resc(st), gt[y0:y1, x0:x1]],
                         ["9.6 um surface volume (input)", "2.4 um teacher, pooled to 9.6 um",
                          "released 9 um model (s42 step-075000)", "manual ink labels"]):
        a.imshow(img, cmap="gray")
        a.set_title(t, fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Supervision-region crop on PHerc0139 w016. The teacher has never seen "
                 "these labels.", fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_teacher_quality.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def fig6():
    d = os.path.join(ROOT, "data/ink9um/volumes_w016")
    off = plane(os.path.join(d, "w016_9um_iso.zarr"))
    ours = plane(os.path.join(d, "w016_ref_render.zarr"))
    h = min(off.shape[0], ours.shape[0]); w = min(off.shape[1], ours.shape[1])
    y0, x0 = h // 2 - 400, w // 2 - 400
    a, b = off[y0:y0 + 800, x0:x0 + 800], ours[y0:y0 + 800, x0:x0 + 800]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.7))
    ax[0].imshow(a, cmap="gray"); ax[0].set_title("official training input (prepare_9um)", fontsize=9)
    ax[1].imshow(b, cmap="gray"); ax[1].set_title("our renderer from tifxyz + CT", fontsize=9)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle("Render fidelity on PHerc0139 w016, plane z=10. Pearson r 0.925 pooled over "
                 "all 21 planes, 0.908 on this plane alone.", fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "fig6_render_fidelity.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    L = results()
    fig1(L); fig2(L); fig3(L); fig4(); fig5(); fig6()

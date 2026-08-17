"""Teacher run on any aligned segment: canonical 2.4um ink model over a public
2.4um surface volume, pooled 4x, scored vs aligned ink_9um labels with
calibration (threshold fit on supervision region, applied to validation) + AUC.

Usage:
  python teacher_seg.py <seg> <sv_zarr> --pilot
  python teacher_seg.py <seg> <sv_zarr> --full fwd|rev
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import tifffile
import torch
import zarr

ROOT = r"C:\Users\nikox\Documents\Vesuvius"
sys.path.insert(0, os.path.join(ROOT, "vendor", "inkdet"))
LBLROOT = os.path.join(ROOT, "data", "ink9um", "labels", "aligned-scrollprizeorg-21slices")
OUTDIR = os.path.join(ROOT, "data", "teacher")
NL, TILE, STRIDE, BATCH = 62, 256, 128, 8


def load_model():
    from model_resnet3d_3d_decoder import RegressionModel
    mdl = RegressionModel(with_norm=True)
    ckp = torch.load(os.path.join(ROOT, "models", "ink_canonical_2um.ckpt"),
                     map_location="cpu", weights_only=False)
    mdl.load_state_dict(ckp.get("state_dict", ckp), strict=False)
    return mdl.eval().cuda().half()


def blend_weight():
    g = np.exp(-0.5 * ((np.arange(TILE) - (TILE - 1) / 2) / (TILE / 2.5)) ** 2)
    return (g[:, None] * g[None, :]).astype(np.float32)


@torch.no_grad()
def infer_region(mdl, vol, z0, y0, x0, hh, ww, reverse):
    st = np.asarray(vol[z0:z0 + NL, y0:y0 + hh, x0:x0 + ww], dtype=np.float32)
    np.clip(st, 0, 200, out=st)
    st /= 200.0
    if reverse:
        st = np.ascontiguousarray(st[::-1])
    wk = blend_weight()
    acc = np.zeros((hh, ww), np.float32)
    ws = np.zeros((hh, ww), np.float32)

    def starts(L):
        s = list(range(0, max(1, L - TILE + 1), STRIDE))
        if s[-1] != L - TILE:
            s.append(L - TILE)
        return s

    coords = [(a, b) for a in starts(hh) for b in starts(ww)]
    t0 = time.time()
    for bi in range(0, len(coords), BATCH):
        cs = coords[bi:bi + BATCH]
        tiles = np.stack([st[:, a:a + TILE, b:b + TILE] for a, b in cs])
        ten = torch.from_numpy(tiles).cuda().half()
        o = mdl(ten)
        if isinstance(o, (list, tuple)):
            o = o[0]
        o = torch.sigmoid(o.float())
        if o.ndim == 4 and o.shape[1] == 1:
            o = o[:, 0]
        o = torch.nn.functional.interpolate(o[:, None], size=(TILE, TILE),
                                            mode="bilinear", align_corners=False)[:, 0]
        o = o.cpu().numpy()
        for (a, b), m in zip(cs, o):
            acc[a:a + TILE, b:b + TILE] += m * wk
            ws[a:a + TILE, b:b + TILE] += wk
    return acc / np.maximum(ws, 1e-6), len(coords) / (time.time() - t0)


def pool4(x):
    h, w = x.shape
    h4, w4 = h // 4 * 4, w // 4 * 4
    return x[:h4, :w4].reshape(h4 // 4, 4, w4 // 4, 4).mean(axis=(1, 3))


def load_mask(path, z=10):
    a = zarr.open(path, mode="r")
    arr = a["0"] if hasattr(a, "array_keys") and "0" in list(a.array_keys()) else a
    return np.asarray(arr[z]) > 0


def bal_acc(pv, gv, t=0.5):
    pb = pv >= t
    tpr = float((pb & gv).sum()) / max(1, int(gv.sum()))
    tnr = float((~pb & ~gv).sum()) / max(1, int((~gv).sum()))
    return 0.5 * (tpr + tnr), tpr, tnr


def auc_of(pv, gv):
    order = np.argsort(pv)
    r = np.empty(len(pv))
    r[order] = np.arange(len(pv))
    n1, n0 = int(gv.sum()), int((~gv).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[gv].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def best_threshold(pv, gv):
    ts = np.linspace(0.02, 0.9, 89)
    bas = [bal_acc(pv, gv, t)[0] for t in ts]
    i = int(np.argmax(bas))
    return float(ts[i]), float(bas[i])


def main():
    seg, sv = sys.argv[1], sys.argv[2]
    mode = sys.argv[3]
    lbl = os.path.join(LBLROOT, seg)
    vol = zarr.open(sv, mode="r")["0"]
    Z, H, W = vol.shape
    z0 = (Z - NL) // 2
    print(f"{seg}: volume {vol.shape}, layers {z0}..{z0+NL}", flush=True)

    gt = load_mask(os.path.join(lbl, f"{seg}_inklabels.zarr"))
    sup = load_mask(os.path.join(lbl, f"{seg}_supervision_mask.zarr"))
    vpath = os.path.join(lbl, f"{seg}_validation_mask.zarr")
    val = load_mask(vpath) if os.path.exists(vpath) else None
    mdl = load_model()

    if mode == "--pilot":
        # densest 512x512 (9.6um) supervision window, not the centroid — a
        # sparse ring-shaped mask has an empty centroid
        k = 512
        c = np.cumsum(np.cumsum(sup.astype(np.int64), 0), 1)
        cp = np.pad(c, ((1, 0), (1, 0)))
        box = cp[k:, k:] - cp[:-k, k:] - cp[k:, :-k] + cp[:-k, :-k]
        by, bx = np.unravel_index(np.argmax(box), box.shape)
        cy, cx = (by + k // 2) * 4, (bx + k // 2) * 4
        hh = ww = min(2048, H, W)
        y0 = min(max(0, cy - hh // 2), H - hh)
        x0 = min(max(0, cx - ww // 2), W - ww)
        res = {}
        for rev in (False, True):
            m, tps = infer_region(mdl, vol, z0, y0, x0, hh, ww, rev)
            m9 = pool4(m)
            g = gt[y0 // 4:y0 // 4 + hh // 4, x0 // 4:x0 // 4 + ww // 4]
            s = sup[y0 // 4:y0 // 4 + hh // 4, x0 // 4:x0 // 4 + ww // 4]
            pv, gv = m9[s], g[s]
            a = auc_of(pv, gv)
            res["rev" if rev else "fwd"] = a
            print(f"  {'rev' if rev else 'fwd'}: AUC {a:.4f} ({int(s.sum())} px) "
                  f"| {tps:.1f} tiles/s", flush=True)
        best = max(res, key=lambda k: (res[k] if res[k] == res[k] else -9))
        ny = len(range(0, max(1, H - TILE + 1), STRIDE)) + 1
        nx = len(range(0, max(1, W - TILE + 1), STRIDE)) + 1
        print(f"DIRECTION: {best} (AUC {res[best]:.4f}); full ~{ny*nx} tiles "
              f"~{ny*nx/tps/60:.0f} min", flush=True)
        json.dump(res, open(os.path.join(OUTDIR, f"pilot_{seg}.json"), "w"))
        return

    rev = sys.argv[4] == "rev"
    BAND = 2048
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    t0 = time.time()
    band_starts = list(range(0, max(1, H - TILE + 1), BAND - STRIDE))
    for i, by in enumerate(band_starts):
        bh = min(BAND, H - by)
        m, tps = infer_region(mdl, vol, z0, by, 0, bh, W, rev)
        acc[by:by + bh] += m
        cnt[by:by + bh] += 1
        el = time.time() - t0
        print(f"band {i+1}/{len(band_starts)} {tps:.1f} t/s {el:.0f}s "
              f"ETA {el/(i+1)*(len(band_starts)-i-1):.0f}s", flush=True)
    pred = acc / np.maximum(cnt, 1)
    dtag = sys.argv[4]
    tifffile.imwrite(os.path.join(OUTDIR, f"{seg}_teacher_{dtag}.tif"),
                     (pred * 255).astype(np.uint8))
    p9 = pool4(pred)
    tifffile.imwrite(os.path.join(OUTDIR, f"{seg}_teacher_{dtag}_9um.tif"),
                     (p9 * 255).astype(np.uint8))
    gh, gw = gt.shape
    p9 = p9[:gh, :gw]

    out = {}
    spv, sgv = p9[sup], gt[sup]
    t_star, sup_ba = best_threshold(spv, sgv)
    out["supervised"] = {"AUC": auc_of(spv, sgv), "bal_acc_at_t*": sup_ba,
                         "t*": t_star, "px": int(sup.sum())}
    print(f"TEACHER supervised: AUC {out['supervised']['AUC']:.4f} "
          f"bal_acc {sup_ba:.4f} at t*={t_star:.2f}", flush=True)
    if val is not None:
        vpv, vgv = p9[val], gt[val]
        vba, vtpr, vtnr = bal_acc(vpv, vgv, t_star)
        out["validation"] = {"AUC": auc_of(vpv, vgv), "bal_acc_at_t*": vba,
                             "TPR": vtpr, "TNR": vtnr, "px": int(val.sum())}
        print(f"TEACHER validation: AUC {out['validation']['AUC']:.4f} "
              f"bal_acc {vba:.4f} (t* from supervision)", flush=True)
    json.dump(out, open(os.path.join(OUTDIR, f"teacher_{seg}_scores.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()

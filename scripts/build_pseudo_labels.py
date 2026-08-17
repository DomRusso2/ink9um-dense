"""STEP 8B step 1: build the dense pseudo-label dataset (HANDOFF.md, verbatim).

For each of the 7 teacher segments, writes into
`data/pseudo/aligned-scrollprizeorg-21slices/<seg>/`:

  <seg>_inklabels.zarr        manual label where manual supervision exists,
                              else teacher_9um >= t*
  <seg>_supervision_mask.zarr (render valid, iso z10 > 0) AND NOT validation_mask
  <seg>_validation_mask.zarr  copied verbatim where one exists

Values 0/255 uint8 as HANDOFF specifies. (Provably equivalent to 0/1 here: the
flat-mode loss binarizes with `torch.amax(inklabels, dim=2) > 0`, train.py:740.)

The validation_mask copy is not in HANDOFF's file list but is required for the
baseline and pseudo runs to stay comparable: the trainer draws its online
validation patches from that mask, and it is also the loader's own safety net
(ink_dataset.py:1983 zeroes supervision wherever validation is set). Without it
the pseudo run would report a val metric computed over a different set.

Every array is created with the SAME shape/chunks/dtype/fill_value/compressor as
the real label zarr it shadows, and three gates are checked per segment before
the segment counts as built.

  python vendor/build_pseudo_labels.py            # all 7, skips completed
  python vendor/build_pseudo_labels.py --only pherc0814-46527
  python vendor/build_pseudo_labels.py --verify-only
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import tifffile
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data/ink9um/labels/aligned-scrollprizeorg-21slices")
PSEUDO = os.path.join(ROOT, "data/pseudo/aligned-scrollprizeorg-21slices")
Z = 10  # annotated plane, confirmed by err and by vendor/step0_verify.py

# t* per segment. w016's scores JSON predates the calibration format and has no
# t*; HANDOFF and memory both record 0.15 (AUC 0.8607 / bal_acc 0.7985).
TSTAR = {
    "pherc0139-w016": 0.15,
    "pherc0139-w017": 0.33,
    "pherc0139-w028": 0.27,
    "pherc0139-w029": 0.25,
    "pherc0814-46527": 0.53,
    "pherc1667-w028": 0.45,
    "pherc1667-w029": 0.36,
}
TEACHER_TIF = {s: f"data/teacher/{s}_teacher_fwd_9um.tif" for s in TSTAR}
TEACHER_TIF["pherc0139-w016"] = "data/teacher/w016_teacher_fwd_9um.tif"
ISO = {
    "pherc0139-w016": "data/ink9um/volumes_w016/w016_9um_iso.zarr",
    "pherc0139-w017": "data/teacher/w017_iso9.zarr",
    "pherc0139-w028": "data/teacher/w028_iso9.zarr",
    "pherc0139-w029": "data/teacher/w029_0139_iso9.zarr",
    "pherc0814-46527": "data/teacher/p0814_iso9.zarr",
    "pherc1667-w028": "data/teacher/p1667w028_iso9.zarr",
    "pherc1667-w029": "data/teacher/p1667w029_iso9.zarr",
}
BAND = 1024  # rows per write, keeps peak memory bounded


def arr_of(path):
    g = zarr.open(path, mode="r")
    return g["0"] if hasattr(g, "array_keys") and "0" in list(g.array_keys()) else g


def plane_of(path, z=Z):
    return np.asarray(arr_of(path)[z])


def write_like(src_path, dst_path, plane):
    """Create a zarr v2 group holding array '0' with the source's exact
    parameters, then write `plane` at z=Z and nothing else."""
    src = arr_of(src_path)
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    g = zarr.open_group(dst_path, mode="w", zarr_format=2)
    dst = g.create_array(
        "0",
        shape=tuple(int(v) for v in src.shape),
        chunks=tuple(int(v) for v in src.chunks),
        dtype=src.dtype,
        fill_value=src.fill_value,
        compressors=src.compressors,
        chunk_key_encoding={"name": "v2",
                            "separator": src.metadata.dimension_separator},
    )
    h = int(src.shape[1])
    for y0 in range(0, h, BAND):
        y1 = min(y0 + BAND, h)
        dst[Z, y0:y1, :] = plane[y0:y1, :]
    # mirror the source's .zattrs so provenance stays readable
    try:
        src_attrs = dict(zarr.open(src_path, mode="r")["0"].attrs)
    except Exception:  # noqa: BLE001
        src_attrs = {}
    if src_attrs:
        dst.attrs.update(src_attrs)
    return dst


def build(seg, force=False):
    d_real = os.path.join(REAL, seg)
    d_out = os.path.join(PSEUDO, seg)
    os.makedirs(d_out, exist_ok=True)

    real_ink = os.path.join(d_real, f"{seg}_inklabels.zarr")
    real_sup = os.path.join(d_real, f"{seg}_supervision_mask.zarr")
    real_val = os.path.join(d_real, f"{seg}_validation_mask.zarr")
    out_ink = os.path.join(d_out, f"{seg}_inklabels.zarr")
    out_sup = os.path.join(d_out, f"{seg}_supervision_mask.zarr")
    out_val = os.path.join(d_out, f"{seg}_validation_mask.zarr")

    if not force and os.path.exists(out_ink) and os.path.exists(out_sup):
        print(f"[{seg}] already built, verifying only")
        return verify(seg)

    canvas = tuple(int(v) for v in arr_of(real_ink).shape[1:])
    print(f"[{seg}] canvas {canvas}  t*={TSTAR[seg]}")

    manual_ink = plane_of(real_ink) > 0
    manual_sup = plane_of(real_sup) > 0
    val = plane_of(real_val) > 0 if os.path.exists(real_val) else np.zeros(canvas, bool)

    teacher = tifffile.imread(os.path.join(ROOT, TEACHER_TIF[seg]))
    if teacher.shape != canvas:  # pooling floor can leave a few extra px
        print(f"  teacher tif {teacher.shape} -> cropping to {canvas}")
        teacher = teacher[:canvas[0], :canvas[1]]
        if teacher.shape != canvas:
            raise SystemExit(f"  teacher smaller than canvas: {teacher.shape} < {canvas}")
    teacher_ink = teacher >= (TSTAR[seg] * 255.0)

    render_valid = plane_of(os.path.join(ROOT, ISO[seg])) > 0

    # manual labels win wherever manual supervision exists (err: the team's
    # conservative negatives are more trustworthy than teacher negatives)
    pseudo_ink = np.where(manual_sup, manual_ink, teacher_ink)
    pseudo_sup = render_valid & ~val

    print(f"  manual sup {manual_sup.mean()*100:5.2f}%  ->  pseudo sup"
          f" {pseudo_sup.mean()*100:5.2f}%  ({pseudo_sup.sum()/max(1,manual_sup.sum()):.1f}x)")
    print(f"  ink: manual {manual_ink.mean()*100:5.2f}%  teacher@t* {teacher_ink.mean()*100:5.2f}%"
          f"  -> pseudo {pseudo_ink.mean()*100:5.2f}%")

    write_like(real_ink, out_ink, (pseudo_ink * 255).astype(np.uint8))
    write_like(real_sup, out_sup, (pseudo_sup * 255).astype(np.uint8))
    if os.path.exists(real_val):
        if os.path.exists(out_val):
            shutil.rmtree(out_val)
        shutil.copytree(real_val, out_val)
        print("  copied validation_mask verbatim")
    return verify(seg)


def verify(seg):
    """HANDOFF's three gates, printed per segment; any failure aborts."""
    d_real, d_out = os.path.join(REAL, seg), os.path.join(PSEUDO, seg)
    real_ink = arr_of(os.path.join(d_real, f"{seg}_inklabels.zarr"))
    ink = arr_of(os.path.join(d_out, f"{seg}_inklabels.zarr"))
    sup = arr_of(os.path.join(d_out, f"{seg}_supervision_mask.zarr"))
    ok = True

    # (a) shapes identical to the real label zarrs
    a_ok = tuple(ink.shape) == tuple(real_ink.shape) == tuple(sup.shape)
    print(f"  (a) shape {tuple(ink.shape)} == real {tuple(real_ink.shape)}: {'OK' if a_ok else 'FAIL'}")
    ok &= a_ok

    ink_p = np.asarray(ink[Z]) > 0
    sup_p = np.asarray(sup[Z]) > 0
    ink_f, sup_f = float(ink_p.mean()), float(sup_p.mean())
    b_ok = (0.30 <= sup_f <= 0.99) and (0.01 <= ink_f <= 0.60)
    print(f"  (b) z10 fractions: supervision {sup_f:.4f} (want 0.30-0.99),"
          f" inklabels {ink_f:.4f} (want 0.01-0.60): {'OK' if b_ok else 'FAIL'}")
    ok &= b_ok

    # (c) zero overlap with the held-out validation region
    real_val = os.path.join(d_real, f"{seg}_validation_mask.zarr")
    if os.path.exists(real_val):
        val_p = np.asarray(arr_of(real_val)[Z]) > 0
        overlap = int((sup_p & val_p).sum())
        c_ok = overlap == 0
        print(f"  (c) supervision AND validation = {overlap} px"
              f" (val region {int(val_p.sum()):,} px): {'OK' if c_ok else 'FAIL — CONTAMINATED'}")
        ok &= c_ok
    else:
        print("  (c) no validation_mask for this segment (nothing to exclude)")

    # other planes must be empty
    other = int(np.asarray(ink[0]).sum()) + int(np.asarray(ink[20]).sum())
    d_ok = other == 0
    print(f"  (d) planes 0/20 empty: {'OK' if d_ok else 'FAIL'}")
    ok &= d_ok

    print(f"  => {seg}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    segs = args.only or list(TSTAR)
    os.makedirs(PSEUDO, exist_ok=True)
    results = {}
    for seg in segs:
        print(f"\n=== {seg} ===")
        results[seg] = verify(seg) if args.verify_only else build(seg, force=args.force)

    print("\n=== SUMMARY ===")
    for seg, ok in results.items():
        print(f"  {seg:<18} {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        sys.exit(1)
    with open(os.path.join(ROOT, "data/pseudo_build_report.json"), "w") as fh:
        json.dump({"segments": list(results), "tstar": TSTAR, "z": Z}, fh, indent=1)
    print("\nALL SEGMENTS PASS")


if __name__ == "__main__":
    main()

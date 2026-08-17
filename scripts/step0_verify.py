"""STEP 0 inventory verification (HANDOFF.md).

Checks, for all 24 aligned + 5 native9 segments:
  * the training-input zarr opens, and its annotated plane is nonzero
  * the inklabels / supervision_mask (/ validation_mask) zarrs open and are
    annotated, and which plane carries the annotation
  * label canvas (y,x) matches the training-input canvas

Cheap by construction: these label zarrs write EVERY chunk (an all-zero
21x128x128 chunk compresses to ~78 bytes), so chunk presence carries no signal
but chunk FILE SIZE does. For sparse arrays (labels/masks) we read every chunk
larger than the modal empty size -> exact canvas-wide counts. For dense arrays
(training inputs) we read an even spread of chunks. Each read is one chunk,
i.e. all planes at once.

Usage:  .venv/Scripts/python.exe -u vendor/step0_verify.py
"""
import json
import os
import sys

import numpy as np
import zarr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIGNED_LABELS = os.path.join(ROOT, "data/ink9um/labels/aligned-scrollprizeorg-21slices")
NATIVE9_LABELS = os.path.join(ROOT, "data/ink9um/labels/native9-scrollprizeorg-21slices")

# segment -> training input zarr (paths exactly as listed in HANDOFF.md)
ALIGNED_INPUTS = {
    "pherc0139-w016": "data/ink9um/volumes_w016/w016_9um_iso.zarr",
    "pherc0139-w017": "data/teacher/w017_iso9.zarr",
    "pherc0139-w028": "data/teacher/w028_iso9.zarr",
    "pherc0139-w029": "data/teacher/w029_0139_iso9.zarr",
    "pherc0814-46527": "data/teacher/p0814_iso9.zarr",
    "pherc1667-w018": "data/ink9um/volumes_pherc1667-w018/w018_9um_iso.zarr",
    "pherc1667-w028": "data/teacher/p1667w028_iso9.zarr",
    "pherc1667-w029": "data/teacher/p1667w029_iso9.zarr",
}
for _seg in ["pherc0139-w035", "pherc0139-w039", "pherc0139-w040", "pherc0139-w041",
             "pherc0139-w043", "pherc1667-w013", "pherc1667-w023", "pherc1667-w031",
             "phercparis4-w00", "phercparis4-w01", "phercparis4-w02", "phercparis4-w03",
             "phercparis4-w05", "phercparis4-w06", "phercparis4-w07", "phercparis4-w09"]:
    ALIGNED_INPUTS[_seg] = f"data/corpus/{_seg}_iso9.zarr"

NATIVE9_INPUTS = {
    "w035": "data/corpus/native9/w035.zarr",
    "w039": "data/corpus/native9/w039.zarr",
    "w040": "data/ink9um/volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr",
    "w041": "data/corpus/native9/w041.zarr",
    "w044": "data/corpus/native9/w044.zarr",
}

ALIGNED_PLANE = 10
NATIVE9_PLANE = 14
MAX_CHUNK_SAMPLE = 48


def open_arr(path):
    """Return (zarr array, array-directory-on-disk) for a group holding '0', or a bare array."""
    g = zarr.open(path, mode="r")
    if hasattr(g, "shape"):
        return g, path
    return g["0"], os.path.join(path, "0")


def present_chunks(arr_dir):
    """(key, size_bytes) for every chunk file on disk."""
    out = []
    for entry in os.scandir(arr_dir):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():  # nested dimension_separator "/"
            for sub in os.scandir(entry.path):
                for sub2 in os.scandir(sub.path):
                    out.append(((int(entry.name), int(sub.name), int(sub2.name)),
                                sub2.stat().st_size))
            continue
        parts = entry.name.split(".")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            out.append((tuple(int(p) for p in parts), entry.stat().st_size))
    return out


def plane_profile(arr, arr_dir, sparse, sample=MAX_CHUNK_SAMPLE, cap=4000):
    """Per-plane nonzero pixel counts.

    sparse=True : read EVERY chunk bigger than the modal (all-zero) size, so the
                  counts are exact canvas-wide totals.
    sparse=False: read an even spread of `sample` chunks (dense array).
    Returns (counts_per_plane, px_seen, n_chunks_total, n_chunks_read, exact, values)
    """
    chunks = present_chunks(arr_dir)
    n_total = len(chunks)
    if n_total == 0:
        return None, 0, 0, 0, False, None
    _, cy, cx = arr.chunks
    if sparse:
        sizes = np.array([s for _, s in chunks])
        empty = int(np.bincount(sizes).argmax()) if sizes.max() < 1 << 20 else int(np.median(sizes))
        sel = [k for (k, s) in chunks if s > empty]
        exact = len(sel) <= cap
        if not exact:
            sel = [sel[i] for i in np.unique(np.linspace(0, len(sel) - 1, cap).astype(int))]
    else:
        keys = sorted(k for k, _ in chunks)
        sel = [keys[i] for i in np.unique(np.linspace(0, n_total - 1, min(sample, n_total)).astype(int))]
        exact = False
    counts = np.zeros(arr.shape[0], dtype=np.int64)
    seen = 0
    values = set()
    for _, ky, kx in sel:
        y0, x0 = ky * cy, kx * cx
        block = np.asarray(arr[:, y0:y0 + cy, x0:x0 + cx])
        counts += (block > 0).reshape(block.shape[0], -1).sum(axis=1)
        seen += block.shape[1] * block.shape[2]
        if len(values) < 8:
            values |= set(np.unique(block).tolist())
    return counts, seen, n_total, len(sel), exact, sorted(values)[:8]


def check(tag, path, expect_plane, expect_yx=None, require_frac=0.05, sparse=False):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        print(f"  FAIL {tag}: MISSING {path}")
        return None, False
    try:
        arr, arr_dir = open_arr(full)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {tag}: cannot open ({type(exc).__name__}: {exc})")
        return None, False
    counts, seen, n_chunks, n_read, exact, values = plane_profile(arr, arr_dir, sparse)
    if counts is None:
        print(f"  FAIL {tag}: shape={arr.shape} NO CHUNKS ON DISK (all-zero)")
        return arr, False
    ann = int(np.argmax(counts))
    canvas = int(arr.shape[1]) * int(arr.shape[2])
    ok = True
    msgs = []
    n_at = int(counts[expect_plane]) if expect_plane < len(counts) else 0
    # sparse -> fraction of the whole canvas (exact); dense -> fraction of sampled px
    frac = n_at / canvas if sparse else n_at / max(seen, 1)
    if frac <= require_frac:
        ok = False
        msgs.append(f"plane {expect_plane} frac {frac:.5f} <= {require_frac}")
    if expect_yx is not None and tuple(arr.shape[1:]) != tuple(expect_yx):
        ok = False
        msgs.append(f"canvas {tuple(arr.shape[1:])} != input {tuple(expect_yx)}")
    nz_planes = int((counts > 0).sum())
    extra = ""
    if sparse:
        extra = (f" px@{expect_plane}={n_at:,}{'' if exact else '~'} of {canvas:,}"
                 f" ({frac * 100:.2f}%) vals={values}")
    else:
        extra = f" frac@{expect_plane}={frac:.4f}"
    print(f"  {'ok  ' if ok else 'FAIL'} {tag}: shape={tuple(arr.shape)}"
          f" chunks={n_read}/{n_chunks} argmax_plane={ann} nz_planes={nz_planes}{extra}"
          + ("  <-- " + "; ".join(msgs) if msgs else ""))
    return arr, ok


def main():
    print(f"zarr {zarr.__version__}  numpy {np.__version__}\n")
    failures = []

    print("=== ALIGNED (24) — training input @z=%d, labels ===" % ALIGNED_PLANE)
    for seg in sorted(ALIGNED_INPUTS):
        print(f"[{seg}]")
        arr, ok = check("input   ", ALIGNED_INPUTS[seg], ALIGNED_PLANE)
        if not ok:
            failures.append(f"{seg}/input")
        yx = tuple(arr.shape[1:]) if arr is not None else None
        d = os.path.join(ALIGNED_LABELS, seg)
        for kind, req in (("inklabels", 0.0), ("supervision_mask", 0.0), ("validation_mask", 0.0)):
            p = os.path.join(d, f"{seg}_{kind}.zarr")
            if kind == "validation_mask" and not os.path.exists(p):
                print(f"  --   {kind:<16}: (none)")
                continue
            _, ok2 = check(f"{kind:<8}", p, ALIGNED_PLANE, expect_yx=yx, require_frac=req,
                           sparse=True)
            if not ok2:
                failures.append(f"{seg}/{kind}")

    print("\n=== NATIVE9 (5) — training input @z=%d, labels ===" % NATIVE9_PLANE)
    for seg in sorted(NATIVE9_INPUTS):
        print(f"[{seg}]")
        arr, ok = check("input   ", NATIVE9_INPUTS[seg], NATIVE9_PLANE)
        if not ok:
            failures.append(f"native9:{seg}/input")
        yx = tuple(arr.shape[1:]) if arr is not None else None
        d = os.path.join(NATIVE9_LABELS, seg)
        for kind in ("inklabels", "supervision_mask", "validation_mask"):
            p = os.path.join(d, f"{seg}_{kind}.zarr")
            if not os.path.exists(p):
                if kind == "validation_mask":
                    print(f"  --   {kind:<16}: (none)")
                    continue
                print(f"  FAIL {kind}: MISSING {p}")
                failures.append(f"native9:{seg}/{kind}")
                continue
            _, ok2 = check(f"{kind:<8}", p, NATIVE9_PLANE, expect_yx=yx, require_frac=0.0,
                           sparse=True)
            if not ok2:
                failures.append(f"native9:{seg}/{kind}")

    print("\n=== TEACHER ASSETS (7) ===")
    teacher = {
        "pherc0139-w016": "data/teacher/w016_teacher_fwd_9um.tif",
        "pherc0139-w017": "data/teacher/pherc0139-w017_teacher_fwd_9um.tif",
        "pherc0139-w028": "data/teacher/pherc0139-w028_teacher_fwd_9um.tif",
        "pherc0139-w029": "data/teacher/pherc0139-w029_teacher_fwd_9um.tif",
        "pherc0814-46527": "data/teacher/pherc0814-46527_teacher_fwd_9um.tif",
        "pherc1667-w028": "data/teacher/pherc1667-w028_teacher_fwd_9um.tif",
        "pherc1667-w029": "data/teacher/pherc1667-w029_teacher_fwd_9um.tif",
    }
    scores = {
        "pherc0139-w016": "data/teacher/teacher_w016_scores.json",
        "pherc0139-w017": "data/teacher/teacher_pherc0139-w017_scores.json",
        "pherc0139-w028": "data/teacher/teacher_pherc0139-w028_scores.json",
        "pherc0139-w029": "data/teacher/teacher_pherc0139-w029_scores.json",
        "pherc0814-46527": "data/teacher/teacher_pherc0814-46527_scores.json",
        "pherc1667-w028": "data/teacher/teacher_pherc1667-w028_scores.json",
        "pherc1667-w029": "data/teacher/teacher_pherc1667-w029_scores.json",
    }
    try:
        import tifffile
    except ImportError:
        tifffile = None
        print("  (tifffile not installed — teacher tif shapes not checked)")
    for seg, rel in teacher.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"  FAIL {seg}: MISSING {rel}")
            failures.append(f"teacher:{seg}")
            continue
        shape = dtype = "?"
        if tifffile is not None:
            with tifffile.TiffFile(p) as tf:
                shape = tf.series[0].shape
                dtype = tf.series[0].dtype
        sp = os.path.join(ROOT, scores[seg])
        tstar = None
        if os.path.exists(sp):
            with open(sp) as fh:
                js = json.load(fh)
            tstar = js.get("supervised", {}).get("t*")
        lab = os.path.join(ALIGNED_LABELS, seg, f"{seg}_inklabels.zarr")
        lyx = None
        if os.path.exists(lab):
            a, _ = open_arr(lab)
            lyx = tuple(a.shape[1:])
        print(f"  ok   {seg}: tif shape={shape} {dtype} label_canvas={lyx} t*={tstar}")

    print("\n=== SUMMARY ===")
    if failures:
        print(f"FAILURES ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

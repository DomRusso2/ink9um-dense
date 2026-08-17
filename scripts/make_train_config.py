"""Build the STEP 8A / 8B training configs (HANDOFF.md).

Starts from the team's own two files and changes ONLY what HANDOFF says to set:
  base contract : configs/aligned21_hybrid_3d2d.json  (all hyperparameters)
  corpus contract: configs/aligned21_fixed_scroll_prior.json
                   (scroll / physical_segment_key / representation_key per segment)

Everything else -- optimizer, lr, loss, bce_label_smoothing, normalization,
patch size, jitter, seed, iterations, batch size, scroll quotas -- is copied
verbatim from the base config. The script prints the exact diff vs the base
config so HANDOFF STEP 8A debug item (d) can be checked at any time.

`sampling_scroll` is a single string per dataset entry (samplers.py:95), so the
29 representations are split into one entry per (scroll, source_family):
4 aligned entries + 1 native9 entry. This is semantically identical to a single
entry -- the sampler groups patches by representation_key and maps each to
(scroll, physical_key) through its dataset contract, never by dataset index.

Usage:
  python vendor/make_train_config.py --out data/train_baseline_config.json \
      --out-dir data/train_baseline
  python vendor/make_train_config.py --out data/train_smoke_config.json \
      --out-dir data/train_smoke --iters 200 --warmup 20
  # STEP 8B: swap the 7 teacher segments' labels to the pseudo root
  python vendor/make_train_config.py --out data/train_pseudo_config.json \
      --out-dir data/train_pseudo --pseudo-root data/pseudo/aligned-scrollprizeorg-21slices
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIR = os.path.join(ROOT, "vendor/villa-ink/ink-detection/configs")
BASE_CFG = os.path.join(CFG_DIR, "aligned21_hybrid_3d2d.json")
PRIOR_CFG = os.path.join(CFG_DIR, "aligned21_fixed_scroll_prior.json")

ALIGNED_LABELS = "data/ink9um/labels/aligned-scrollprizeorg-21slices"
NATIVE9_LABELS = "data/ink9um/labels/native9-scrollprizeorg-21slices"

# The 7 segments with banked teacher maps (STEP 8B swaps these to the pseudo root).
TEACHER_SEGMENTS = [
    "pherc0139-w016", "pherc0139-w017", "pherc0139-w028", "pherc0139-w029",
    "pherc0814-46527", "pherc1667-w028", "pherc1667-w029",
]

# Training inputs, all verified by vendor/step0_verify.py (STEP 0).
VOLUMES = {
    ("public_2p4_level2_zmean4", "pherc0139-w016"): "data/ink9um/volumes_w016/w016_9um_iso.zarr",
    ("public_2p4_level2_zmean4", "pherc0139-w017"): "data/teacher/w017_iso9.zarr",
    ("public_2p4_level2_zmean4", "pherc0139-w028"): "data/teacher/w028_iso9.zarr",
    ("public_2p4_level2_zmean4", "pherc0139-w029"): "data/teacher/w029_0139_iso9.zarr",
    ("public_2p4_level2_zmean4", "pherc0814-46527"): "data/teacher/p0814_iso9.zarr",
    ("public_2p4_level2_zmean4", "pherc1667-w018"): "data/ink9um/volumes_pherc1667-w018/w018_9um_iso.zarr",
    ("public_2p4_level2_zmean4", "pherc1667-w028"): "data/teacher/p1667w028_iso9.zarr",
    ("public_2p4_level2_zmean4", "pherc1667-w029"): "data/teacher/p1667w029_iso9.zarr",
    ("native_9p362_level0", "w040"): "data/ink9um/volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr",
}
for _s in ["pherc0139-w035", "pherc0139-w039", "pherc0139-w040", "pherc0139-w041",
           "pherc0139-w043", "pherc1667-w013", "pherc1667-w023", "pherc1667-w031",
           "phercparis4-w00", "phercparis4-w01", "phercparis4-w02", "phercparis4-w03",
           "phercparis4-w05", "phercparis4-w06", "phercparis4-w07", "phercparis4-w09"]:
    VOLUMES[("public_2p4_level2_zmean4", _s)] = f"data/corpus/{_s}_iso9.zarr"
for _s in ["w035", "w039", "w041", "w044"]:
    VOLUMES[("native_9p362_level0", _s)] = f"data/corpus/native9/{_s}.zarr"

LABEL_ROOTS = {
    "public_2p4_level2_zmean4": ALIGNED_LABELS,
    "native_9p362_level0": NATIVE9_LABELS,
}


def abspath(rel):
    return os.path.join(ROOT, rel).replace("\\", "/")


def build_datasets(pseudo_root=None, exclude=()):
    with open(PRIOR_CFG) as fh:
        prior = json.load(fh)
    groups = {}
    for rep in prior["representations"]:
        fam, seg, scroll = rep["source_family"], rep["segment"], rep["scroll"]
        label_root = LABEL_ROOTS[fam]
        # STEP 8B: the teacher segments read their labels from the pseudo root
        if (pseudo_root and fam == "public_2p4_level2_zmean4"
                and seg in TEACHER_SEGMENTS and seg not in exclude):
            label_root = pseudo_root
        key = (scroll, fam, label_root)
        g = groups.setdefault(key, {
            "segments_path": abspath(label_root),
            "segments": [],
            "surface_volume_paths": {},
            "volume_scale": 0,
            "source_family": fam,
            "sampling_scroll": scroll,
            "sampling_physical_segment_keys": {},
            "sampling_representation_keys": {},
        })
        vol = VOLUMES[(fam, seg)]
        if not os.path.exists(os.path.join(ROOT, vol)):
            raise SystemExit(f"missing training input for {fam}/{seg}: {vol}")
        if not os.path.exists(os.path.join(ROOT, label_root, seg)):
            raise SystemExit(f"missing label dir for {fam}/{seg}: {label_root}/{seg}")
        g["segments"].append(seg)
        g["surface_volume_paths"][seg] = abspath(vol)
        g["sampling_physical_segment_keys"][seg] = rep["physical_segment_key"]
        g["sampling_representation_keys"][seg] = rep["representation_key"]
    order = {"0139": 0, "1667": 1, "Paris4": 2, "0814": 3}
    return [groups[k] for k in sorted(groups, key=lambda k: (k[1] != "public_2p4_level2_zmean4",
                                                            order.get(k[0], 9)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=None, help="override num_iterations (smoke only)")
    ap.add_argument("--warmup", type=int, default=None, help="override warmup_steps (smoke only)")
    ap.add_argument("--pseudo-root", default=None, help="STEP 8B pseudo-label dataset root")
    ap.add_argument("--exclude-pseudo", action="append", default=[],
                    help="segment(s) kept on REAL labels (STEP 8B stop-rule diagnostic)")
    args = ap.parse_args()

    with open(BASE_CFG) as fh:
        cfg = json.load(fh)
    base = json.loads(json.dumps(cfg))

    cfg["out_dir"] = abspath(args.out_dir)
    cfg["datasets"] = build_datasets(args.pseudo_root, tuple(args.exclude_pseudo))
    if args.iters is not None:
        cfg["num_iterations"] = args.iters
    if args.warmup is not None:
        cfg["warmup_steps"] = args.warmup

    out = os.path.join(ROOT, args.out)
    with open(out, "w") as fh:
        json.dump(cfg, fh, indent=1)

    print(f"wrote {args.out}")
    print("\n--- diff vs aligned21_hybrid_3d2d.json (must be only out_dir/datasets"
          " [+ iters/warmup for the smoke copy]) ---")
    for k in sorted(set(base) | set(cfg)):
        if base.get(k) != cfg.get(k):
            if k == "datasets":
                print(f"  datasets: 1 placeholder entry -> {len(cfg['datasets'])} entries")
            else:
                print(f"  {k}: {base.get(k)!r} -> {cfg.get(k)!r}")

    n_seg = sum(len(d["segments"]) for d in cfg["datasets"])
    print(f"\n--- corpus: {n_seg} representations in {len(cfg['datasets'])} dataset entries ---")
    for d in cfg["datasets"]:
        swapped = [s for s in d["segments"]
                   if args.pseudo_root and args.pseudo_root in d["segments_path"]]
        print(f"  scroll={d['sampling_scroll']:<6} family={d['source_family']:<24}"
              f" n={len(d['segments']):<2} labels={os.path.relpath(d['segments_path'], ROOT)}"
              + (f"  [PSEUDO: {len(swapped)}]" if swapped else ""))
    quotas = cfg["fixed_scroll_prior"]["target_batch_counts"]
    print(f"\n  target_batch_counts={quotas} sum={sum(quotas.values())} batch_size={cfg['batch_size']}")
    print(f"  num_iterations={cfg['num_iterations']} warmup={cfg['warmup_steps']}"
          f" seed={cfg['seed']} bce_label_smoothing={cfg['loss']['bce_label_smoothing']}")


if __name__ == "__main__":
    main()

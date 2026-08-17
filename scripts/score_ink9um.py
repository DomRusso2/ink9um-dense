"""Score a 9 um ink prediction with the TEAM'S OWN metric.

Why it is written this way:
  * `BalancedAccuracy` is their selection metric (`best_checkpoint_metric:
    val_balanced_accuracy` in the released checkpoint config), so it is the
    number to report. Ours can go alongside, never instead.
  * `Confusion._counts_from_tensors` does `logits[valid_mask]`, i.e. it EXCLUDES
    masked pixels rather than counting them as true negatives. Labels exist only
    inside `_supervision_mask`, so scoring the whole canvas would grade the model
    against absent labels over ~88% of the segment. The mask is mandatory.
  * Their metric applies `torch.sigmoid` internally, but `infer.py` writes
    uint8 probability*255. Feeding p straight in would double-apply sigmoid, so
    p is inverted to a logit first. A direct threshold-on-p computation is run
    alongside as a cross-check; the two must agree.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import tifffile
import torch
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ink9um_env  # noqa: E402

ink9um_env.install(verbose=False)

from koine_machines.evaluation.metrics.balanced_accuracy import BalancedAccuracy  # noqa: E402
from koine_machines.evaluation.metrics.confusion import Confusion  # noqa: E402


@dataclass
class _Batch:
    logits: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor | None

    def require_targets(self):
        return self.targets


def load_plane(zarr_path: str, z: int) -> np.ndarray:
    a = zarr.open(zarr_path, mode="r")
    arr = a["0"] if hasattr(a, "array_keys") and "0" in list(a.array_keys()) else a
    return np.asarray(arr[z])


def annotated_z(zarr_path: str) -> int:
    """Find the single annotated plane rather than assuming 10 (aligned) or 14
    (native9).

    Scans a STRIDED SUBSAMPLE OF THE FULL EXTENT, not a centre window. An
    earlier version probed one 512x512 block at the centre of the canvas and
    returned z=0 for w016, whose annotated region is off-centre - which scored
    0 supervised pixels and a balanced accuracy of 0.0. A centre window is not
    a safe proxy for where the annotation lives.
    """
    a = zarr.open(zarr_path, mode="r")
    arr = a["0"] if hasattr(a, "array_keys") and "0" in list(a.array_keys()) else a
    sub = np.asarray(arr[:, ::8, ::8])
    counts = [int((sub[i] > 0).sum()) for i in range(sub.shape[0])]
    if max(counts) == 0:
        raise ValueError(f"no annotated plane found in {zarr_path}")
    return int(np.argmax(counts))


def score(pred_tif: str, inklabels: str, supervision: str, z: int | None = None) -> dict:
    if z is None:
        z = annotated_z(supervision)
    p = tifffile.imread(pred_tif).astype(np.float32) / 255.0
    gt = load_plane(inklabels, z) > 0
    valid = load_plane(supervision, z) > 0

    if p.shape != gt.shape:
        raise ValueError(f"pred {p.shape} != label {gt.shape}")

    # ---- their metric ----
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    batch = _Batch(
        logits=torch.from_numpy(logit),
        targets=torch.from_numpy(gt.astype(np.float32)),
        valid_mask=torch.from_numpy(valid),
    )
    theirs = BalancedAccuracy(threshold=0.5).compute_batch(batch)
    counts = Confusion(threshold=0.5).compute_batch(batch)
    tp, fp, fn, tn = (float(counts.tp), float(counts.fp), float(counts.fn), float(counts.tn))

    # ---- independent cross-check: threshold p directly, mask by hand ----
    pv, gv = p[valid], gt[valid]
    pb = pv >= 0.5
    tpr = float((pb & gv).sum()) / max(1, int(gv.sum()))
    tnr = float((~pb & ~gv).sum()) / max(1, int((~gv).sum()))
    direct = 0.5 * (tpr + tnr)

    return {
        "z": z,
        "supervised_px": int(valid.sum()),
        "supervised_frac": float(valid.mean()),
        "ink_frac_in_mask": float(gv.mean()),
        "balanced_accuracy_THEIRS": theirs,
        "balanced_accuracy_direct": direct,
        "agree": abs(theirs - direct) < 1e-4,
        "TPR": tpr,
        "TNR": tnr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pred_mean_in_mask": float(pv.mean()),
    }


if __name__ == "__main__":
    import json

    r = score(sys.argv[1], sys.argv[2], sys.argv[3],
              z=int(sys.argv[4]) if len(sys.argv) > 4 else None)
    print(json.dumps(r, indent=2))

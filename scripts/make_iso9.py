"""Build the official-recipe 21-slice ~9.6um training input from a level-0
2.4um surface volume: 84 centered z planes mean-pooled 4x, XY mean-pooled 4x.
Banded so GPU blocks stay ~2 GB. Verifies each plane is nonzero.

Usage: python make_iso9.py <src_sv_zarr> <dest_iso_zarr>
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch
import zarr
from numcodecs import Blosc

BAND = 4096  # source rows per GPU block (multiple of 4)


def main() -> int:
    src_path, dst_path = sys.argv[1], sys.argv[2]
    src = zarr.open(src_path, mode="r")["0"]
    Z, H, W = src.shape
    z0 = math.ceil((Z - 84) / 2)
    g = zarr.open_group(dst_path, mode="w", zarr_format=2)
    out = g.create_array("0", shape=(21, H // 4, W // 4), chunks=(21, 128, 128),
                         dtype=np.uint8, fill_value=0,
                         compressors=[Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)])
    bad = 0
    for zi in range(21):
        rows = []
        for y0 in range(0, H // 4 * 4, BAND):
            y1 = min(H // 4 * 4, y0 + BAND)
            blk = np.asarray(src[z0 + zi * 4:z0 + zi * 4 + 4, y0:y1, :W // 4 * 4],
                             dtype=np.float32)
            t = torch.from_numpy(blk).cuda()
            t = torch.nn.functional.avg_pool2d(t.mean(dim=0, keepdim=True)[None], 4)
            rows.append(t[0, 0].round().clamp(0, 255).to(torch.uint8).cpu().numpy())
            del t
        plane = np.concatenate(rows, axis=0)
        out[zi] = plane
        nz = float((plane > 0).mean())
        print(f"plane {zi} nonzero {nz:.3f}", flush=True)
        if nz == 0.0:
            bad += 1
    if bad:
        print(f"!! {bad} all-zero planes - DO NOT USE", flush=True)
        return 1
    print(f"done {out.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

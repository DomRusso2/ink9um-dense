"""Render the full w016 canvas (21 x 7020 x 7220 @ 9.6um) from local CT L2
with the validated reference sampler (grid_sample fp32, FD normals, +-10
planes at 1 L2-voxel step). Tiled 1024^2 with per-tile CT sub-reads.

Writes a zarr the koine inference consumes directly, plus per-tile progress.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import tifffile
import torch
import zarr
from numcodecs import Blosc

ROOT = r"C:\Users\nikox\Documents\Vesuvius"
OUT = os.path.join(ROOT, "data", "ink9um", "volumes_w016", "w016_ref_render.zarr")
TILE = 1024
NZ, H, W = 21, 7020, 7220


def main():
    t0 = time.time()
    xs = tifffile.imread(rf"{ROOT}\data\ink9um\tifxyz\PUBLIC_w016\x.tif")
    ys = tifffile.imread(rf"{ROOT}\data\ink9um\tifxyz\PUBLIC_w016\y.tif")
    zs = tifffile.imread(rf"{ROOT}\data\ink9um\tifxyz\PUBLIC_w016\z.tif")
    grid = np.stack([zs, ys, xs], -1).astype(np.float32)
    valid = (xs > 0) & (ys > 0) & (zs > 0)
    # invalid cells are -1: NaN them out so bilinear interpolation cannot
    # blend valid coords with -1 into garbage positions (inflates the CT
    # bbox by orders of magnitude and samples the wrong place)
    grid[~valid] = np.nan
    gh, gw = valid.shape

    a = zarr.open(rf"{ROOT}\data\ct0139_L2.zarr", mode="r")["2"]
    zdim, ydim, xdim = a.shape

    part = OUT + ".partial"
    g = zarr.open_group(part, mode="a", zarr_format=2)
    if "0" not in g:
        arr = g.create_array("0", shape=(NZ, H, W), chunks=(NZ, 128, 128),
                             dtype=np.uint8, fill_value=0,
                             compressors=[Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)])
    else:
        arr = g["0"]
    donefile = part + ".done.json"
    done = set(json.load(open(donefile))) if os.path.exists(donefile) else set()

    dev = "cuda"
    tgrid = torch.from_numpy(grid).permute(2, 0, 1)[None]

    tiles = [(y0, x0) for y0 in range(0, H, TILE) for x0 in range(0, W, TILE)]
    for idx, (y0, x0) in enumerate(tiles):
        key = f"{y0}_{x0}"
        if key in done:
            continue
        th, tw = min(TILE, H - y0), min(TILE, W - x0)
        # grid validity for this tile (any valid cell?)
        vy0, vy1 = y0 // 5, min(gh, (y0 + th) // 5 + 2)
        vx0, vx1 = x0 // 5, min(gw, (x0 + tw) // 5 + 2)
        if not valid[vy0:vy1, vx0:vx1].any():
            done.add(key)
            if idx % 10 == 0:
                json.dump(sorted(done), open(donefile, "w"))
            continue

        gy = (torch.arange(th, dtype=torch.float32) + y0) / 5.0
        gx = (torch.arange(tw, dtype=torch.float32) + x0) / 5.0
        yy = (gy / (gh - 1) * 2 - 1)[:, None].expand(th, tw)
        xx = (gx / (gw - 1) * 2 - 1)[None, :].expand(th, tw)
        gsamp = torch.stack([xx, yy], -1)[None]
        coords = torch.nn.functional.grid_sample(
            tgrid, gsamp, mode="bilinear", align_corners=True)[0].permute(1, 2, 0).numpy()

        vmask = np.isfinite(coords).all(axis=-1) & (coords.min(axis=-1) > 0)
        if not vmask.any():
            done.add(key)
            continue

        dyv = np.gradient(coords, axis=0)
        dxv = np.gradient(coords, axis=1)
        n = np.cross(dxv.reshape(-1, 3), dyv.reshape(-1, 3)).reshape(th, tw, 3)
        n /= (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)

        base = coords / 4.0
        bm = base[vmask]
        lo = np.floor(bm.min(axis=0)).astype(int) - 12
        hi = np.ceil(bm.max(axis=0)).astype(int) + 13
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, [zdim, ydim, xdim])
        if (hi - lo).min() <= 0:
            done.add(key)
            continue
        sub_bytes = 4 * int(np.prod(hi - lo))
        if sub_bytes > 8e9:
            raise MemoryError(f"tile {key}: CT sub would be {sub_bytes/1e9:.1f} GB "
                              f"({lo}..{hi}) - coordinate outliers not cleaned?")
        sub = np.asarray(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], dtype=np.float32)

        v = torch.from_numpy(sub)[None, None].to(dev)
        b0 = torch.from_numpy((base - lo).astype(np.float32)).to(dev)
        nrm = torch.from_numpy(n.astype(np.float32)).to(dev)
        vm = torch.from_numpy(vmask).to(dev)
        sz = torch.tensor([sub.shape[0] - 1, sub.shape[1] - 1, sub.shape[2] - 1],
                          dtype=torch.float32, device=dev)
        out = torch.zeros((NZ, th, tw), dtype=torch.uint8, device=dev)
        for k in range(NZ):
            p = b0 + float(k - 10) * nrm
            gn = (p / sz * 2 - 1)
            gs = torch.stack([gn[..., 2], gn[..., 1], gn[..., 0]], -1)[None, None]
            s = torch.nn.functional.grid_sample(v, gs, mode="bilinear",
                                                align_corners=True)[0, 0, 0]
            s = torch.where(vm, s, torch.zeros_like(s))
            out[k] = s.nan_to_num(0).round().clamp(0, 255).to(torch.uint8)
        arr[:, y0:y0 + th, x0:x0 + tw] = out.cpu().numpy()
        done.add(key)
        json.dump(sorted(done), open(donefile, "w"))
        el = time.time() - t0
        n_done = len(done)
        print(f"tile {n_done}/{len(tiles)} ({key})  {el:.0f}s  "
              f"ETA {el / max(1, n_done) * (len(tiles) - n_done):.0f}s", flush=True)

    if os.path.exists(OUT):
        raise FileExistsError(OUT)
    os.replace(part, OUT)
    print(f"DONE -> {OUT}  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

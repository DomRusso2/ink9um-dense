"""Fetch aligned-21slice label folders for given segments from the ink_9um HF
bucket. Tree API is rate limited (500 req / 300 s) so listing is per-directory
and shallow; files go through resolve/ (not rate limited).
Usage: python fetch_labels.py [--dest=DIR] pherc0814-46527 pherc1667-w029 ...
NOTE: fetch_parallel skips files that already exist, so re-fetching a
corrected label version MUST go to a fresh --dest or nothing downloads."""
from __future__ import annotations

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_parallel import run  # noqa: E402

TREE = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
RESOLVE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
BASES = {
    "aligned": ("ink_9um/labels/aligned-scrollprizeorg-21slices",
                r"C:\Users\nikox\Documents\Vesuvius\data\ink9um\labels\aligned-scrollprizeorg-21slices"),
    "native9": ("ink_9um/labels/native9-scrollprizeorg-21slices",
                r"C:\Users\nikox\Documents\Vesuvius\data\ink9um\labels\native9-scrollprizeorg-21slices"),
}
UA = {"User-Agent": "curl/8"}


def list_dir(path):
    """Yield (type, path) entries for one directory, following pagination."""
    url = TREE + urllib.parse.quote(path)
    while url:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            items = json.load(r)
            link = r.headers.get("Link", "")
        for it in items:
            yield it["type"], it["path"]
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1:part.find(">")]


def walk(path):
    files = []
    for t, p in list_dir(path):
        if t == "directory":
            files += walk(p)
        else:
            files.append(p)
    return files


def main() -> int:
    tasks = []
    args = list(sys.argv[1:])
    dest_override = None
    if args and args[0].startswith("--dest="):
        dest_override = args.pop(0).split("=", 1)[1]
    for arg in args:
        kind, seg = arg.split(":") if ":" in arg else ("aligned", arg)
        base, dest = BASES[kind]
        if dest_override:
            dest = os.path.join(dest_override, base.split("/")[-1])
        root = f"{base}/{seg}"
        files = walk(root)
        print(f"{arg}: {len(files)} files", flush=True)
        for f in files:
            rel = f[len(base) + 1:]
            tasks.append((RESOLVE + f, os.path.join(dest, rel.replace("/", os.sep))))
    st = run(tasks, threads=8, label="labels")
    missing = [d for _, d in tasks if not (os.path.exists(d) and os.path.getsize(d) > 0)]
    print(f"completeness: {len(tasks)-len(missing)}/{len(tasks)}", flush=True)
    if st["failed"] or missing:
        print("!! INCOMPLETE - rerun to resume", flush=True)
        return 1
    print("GATED COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

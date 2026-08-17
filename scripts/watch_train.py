"""Progress / stall monitor for a koine_machines training run (HANDOFF rule 4).

The trainer writes a tqdm bar with carriage returns, so the log is one enormous
line and `tail` is useless on it. This reads the tail bytes, splits on \\r as
well as \\n, and reports:

  phase (patch discovery vs training), step/total, %, it/s now and mean,
  loss trend, elapsed, ETA, age of the last log write (STALL detector),
  GPU utilisation + which PIDs hold GPU memory, and the checkpoints on disk.

A run is called STALLED when the log has not been written for --stall-seconds
(default 300) -- during patch discovery the bar only ticks once per segment, so
the threshold is relaxed automatically in that phase.

Usage:
  python vendor/watch_train.py data/train_smoke.log
  python vendor/watch_train.py data/train_baseline.log --interval 300   # loop
"""
import argparse
import os
import re
import subprocess
import time

BAR = re.compile(
    r"(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.]+)(it/s|s/it)"
    r"(?:,\s*loss=([\d.eE+-]+))?(?:,\s*lr=([\d.eE+-]+))?"
)
FINDING = re.compile(r"Finding patches:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")


def tail_text(path, nbytes=400_000):
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - nbytes))
        return fh.read().decode("utf-8", errors="replace"), size


def hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def gpu_status():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        util, used, total, temp = [x.strip() for x in q.stdout.strip().split(",")]
        a = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        apps = [l.split(",")[0].strip() for l in a.stdout.strip().splitlines() if l.strip()]
        apps = [p for p in apps if p.isdigit()]
        return f"GPU {util}% util, {int(used)/1024:.1f}/{int(total)/1024:.1f} GiB, {temp}C", apps
    except Exception as exc:  # noqa: BLE001
        return f"GPU status unavailable ({type(exc).__name__})", []


def snapshot(log_path, out_dir, stall_seconds):
    if not os.path.exists(log_path):
        print(f"log does not exist yet: {log_path}")
        return
    text, size = tail_text(log_path)
    age = time.time() - os.path.getmtime(log_path)
    parts = re.split(r"[\r\n]", text)

    bars = [m for m in (BAR.search(p) for p in parts) if m]
    finds = [m for m in (FINDING.search(p) for p in parts) if m]
    audit = [p for p in parts if p.startswith("sampling_audit=")]
    observed = [p for p in parts if p.startswith("sampling_observed=")]
    errors = [p for p in parts if re.search(r"Traceback|Error|error:|CUDA out of memory|Killed", p)]

    print(f"=== {os.path.basename(log_path)}  ({size/1e6:.2f} MB, last write {hms(age)} ago) ===")

    # the training bar carries loss=; the patch-finding bar never does
    training = next((m for m in reversed(bars) if m.group(7)), None)

    if training is None:
        phase = "PATCH DISCOVERY"
        if finds:
            f = finds[-1]
            print(f"phase: {phase} -- {f.group(2)}/{f.group(3)} segments ({f.group(1)}%)")
        else:
            print(f"phase: {phase} -- starting up (no progress line yet)")
        stall_seconds = max(stall_seconds, 900)  # discovery ticks once per segment
    else:
        cur, tot = int(training.group(1)), int(training.group(2))
        elapsed_s, eta_s = training.group(3), training.group(4)
        rate = float(training.group(5))
        if training.group(6) == "s/it":
            rate = 1.0 / rate if rate else 0.0
        loss, lr = training.group(7), training.group(8)
        losses = [float(m.group(7)) for m in bars[-400:] if m.group(7)]
        trend = ""
        if len(losses) > 20:
            first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
            trend = f"  (mean {first:.4f} -> {last:.4f} over last {len(losses)} logged)"
        pct = 100.0 * cur / tot
        remain = (tot - cur) / rate if rate else 0
        print(f"phase: TRAINING  step {cur:,}/{tot:,} ({pct:.2f}%)")
        print(f"  rate   {rate:.2f} it/s   elapsed {elapsed_s}   tqdm ETA {eta_s}"
              f"   -> finish ~{time.strftime('%H:%M', time.localtime(time.time() + remain))}"
              f" ({hms(remain)} left)")
        print(f"  loss   {loss}   lr {lr}{trend}")

    gpu, apps = gpu_status()
    print(f"  {gpu}" + (f"   compute pids: {', '.join(apps)}" if apps else "   NO GPU PROCESS"))

    if out_dir and os.path.isdir(out_dir):
        ck = [(f, os.path.getmtime(os.path.join(out_dir, f)))
              for f in os.listdir(out_dir) if f.endswith((".pth", ".pt"))]
        if ck:
            ck.sort(key=lambda t: t[1])
            print(f"  checkpoints: {len(ck)}  latest {ck[-1][0]} ({hms(time.time()-ck[-1][1])} ago)")
        else:
            print(f"  checkpoints: none yet in {os.path.relpath(out_dir)}")

    vpath = os.path.join(out_dir, "validation_metrics.jsonl") if out_dir else None
    if vpath and os.path.exists(vpath):
        import json as _json
        recs = [_json.loads(l) for l in open(vpath) if l.strip()]
        if recs:
            best = max(recs, key=lambda r: r["val_balanced_accuracy"])
            last = recs[-1]
            print(f"  val    {len(recs)} passes | last step {last['step']:,}:"
                  f" bal_acc {last['val_balanced_accuracy']:.4f}"
                  f" bce_unsmoothed {last['val_bce_unsmoothed']:.4f}"
                  f" | best {best['val_balanced_accuracy']:.4f} @ step {best['step']:,}")

    if audit:
        counts = re.search(r'"source_patches_by_representation":\s*(\{[^}]*\})', audit[-1])
        total = re.search(r'"source_patches":\s*(\d+)', audit[-1])
        if total:
            print(f"  source_patches total: {int(total.group(1)):,}")
        if counts:
            import json as _json
            per = _json.loads(counts.group(1))
            zeros = [k for k, v in per.items() if v == 0]
            print(f"  per-representation patches: {len(per)} representations,"
                  f" min={min(per.values()):,} max={max(per.values()):,}"
                  + (f"  ZERO-PATCH: {zeros}" if zeros else "  (none at zero)"))
    if observed:
        step = re.search(r'"training_step_completed":\s*(\d+)', observed[-1])
        if step:
            print(f"  last sampling audit at step {int(step.group(1)):,}")
    if errors:
        print(f"  !! {len(errors)} error-like lines, last: {errors[-1][:200]}")

    if age > stall_seconds:
        print(f"  !! STALLED: no log write for {hms(age)} (threshold {hms(stall_seconds)})")
    return training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stall-seconds", type=float, default=300)
    ap.add_argument("--interval", type=float, default=0, help="loop forever every N seconds")
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir is None:  # infer from the sibling config, e.g. train_smoke.log -> train_smoke
        guess = args.log[:-4] if args.log.endswith(".log") else args.log
        out_dir = guess if os.path.isdir(guess) else None

    while True:
        snapshot(args.log, out_dir, args.stall_seconds)
        if not args.interval:
            return
        print()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

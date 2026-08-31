#!/usr/bin/env python3
"""Turn a saved run trace into a reference line.

This is the handover between the two training stages. Stage one drives a
provisional line straight through the map's checkpoints, using the lidar to
find the road; the route it actually took is recorded as a trace. This turns
that trace into a `Centerline` - the same file format a recorded human lap
produces - so stage two can refine it with nothing downstream changing.

    python3 tools/line_from_trace.py runs/<uid>/traces/<file>.json lines/foo.json

With no trace named, the best one for that map is chosen: a finished run if
there is one (fastest wins), otherwise the one that got furthest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.centerline import Centerline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_trace(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def best_trace(map_uid: str) -> str:
    d = os.path.join(ROOT, "runs", map_uid, "traces")
    if not os.path.isdir(d):
        raise SystemExit(f"no traces for map {map_uid} ({d} missing)")
    best, best_key = None, None
    for fn in os.listdir(d):
        if not fn.endswith(".json"):
            continue
        full = os.path.join(d, fn)
        try:
            t = load_trace(full)
        except (OSError, json.JSONDecodeError):
            continue
        # A finished run always beats an unfinished one, however far the
        # unfinished one got - only a finished lap proves the route works.
        key = (1 if t.get("finished") else 0,
               -(t.get("race_time") or 0) if t.get("finished")
               else t.get("distance") or 0)
        if best_key is None or key > best_key:
            best, best_key = full, key
    if best is None:
        raise SystemExit(f"no readable traces in {d}")
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", help="trace json; omit to pick the best")
    ap.add_argument("out", help="reference line json to write")
    ap.add_argument("--map", help="map uid, when picking the best trace")
    ap.add_argument("--spacing", type=float, default=2.0,
                    help="resample spacing in metres")
    ap.add_argument("--min-speed", type=float, default=1.0,
                    help="drop samples slower than this (m/s); a car sitting "
                         "on the start line would otherwise pile hundreds of "
                         "identical points at the first metre of the line")
    args = ap.parse_args()

    path = args.trace or best_trace(args.map or "")
    t = load_trace(path)
    samples = t.get("samples") or []
    if not samples:
        raise SystemExit(f"{path} has no samples")

    # fields: t, x, y, z, speed, steer, gas, brake, cp
    pts = np.asarray([[s[1], s[2], s[3]] for s in samples
                      if (s[4] or 0) >= args.min_speed], dtype=np.float64)
    if len(pts) < 2:
        raise SystemExit(
            f"{path} has {len(pts)} samples above {args.min_speed} m/s - "
            "the car barely moved, so there is no line to build")

    line = Centerline(pts, spacing=args.spacing)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    line.save(args.out)

    kind = "FINISHED" if t.get("finished") else "unfinished"
    rt = t.get("race_time")
    print(f"from {os.path.relpath(path, ROOT)}")
    print(f"  {kind}, episode {t.get('episode')}, "
          f"{t.get('checkpoints', 0)} checkpoints"
          + (f", {rt/1000:.3f}s" if rt else ""))
    print(f"  {len(samples)} samples -> {len(pts)} moving -> "
          f"{len(line.points)} resampled, {line.length:.0f}m")
    print(f"wrote {args.out}")
    print(f"\nnow: python3 train/train_sac.py --line {args.out} --resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())

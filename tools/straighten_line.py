#!/usr/bin/env python3
"""Take the wobble out of a recorded reference line.

A line recorded by driving - by you or by the explore stage - contains every
correction the driver made. On a real track that is mostly fine: the racing
line genuinely curves. On a straight road it is a disaster, and it is the
reason the policy snakes.

`lines/ExploreMode.json` is the worked example. It is a recording of a *dead
straight* road, and it measures:

    length 1534.2m   chord 1482.9m   wanders up to 11.0m sideways
    836 degrees of cumulative heading change over a straight road

Two things follow from that, and both punish driving straight:

  * the six lookahead points fed to the policy swing sideways by up to 11m, so
    the thing it is being pointed at is a slalom, and
  * `soft_offset` is 8m, so a car driving perfectly straight down the middle of
    the road is *outside the line's own wobble* for much of the map and is
    charged the off-line penalty for it.

The policy was not misbehaving. It was doing what the line asked.

    tools/straighten_line.py lines/ExploreMode.json --smooth 60
    tools/straighten_line.py lines/ExploreMode.json --chord      # perfectly straight
    tools/straighten_line.py lines/foo.json --smooth 25 -o lines/foo-clean.json

`--smooth W` is a moving average over W metres of arc length, run with the
ends pinned so the start and finish do not drift off the road. Bigger W is
straighter. `--chord` replaces the line with the straight segment between its
endpoints, which is what a straight-road map actually wants.

Nothing here invents geometry: the output stays inside the envelope of the
recording, so it cannot smooth the line through a wall. Check the report it
prints - if `max shift` is larger than the road is wide, the window is too big.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.centerline import Centerline


def smooth(points: np.ndarray, s: np.ndarray, window_m: float) -> np.ndarray:
    """Moving average over arc length, with the two ends pinned.

    Pinned because the endpoints are the spawn and the finish: dragging either
    of them sideways moves the line off the start block, and every projection
    after that is measured from the wrong place.
    """
    n = len(points)
    out = np.array(points, dtype=np.float64)
    for i in range(n):
        lo = np.searchsorted(s, s[i] - window_m * 0.5)
        hi = np.searchsorted(s, s[i] + window_m * 0.5)
        if hi - lo >= 2:
            out[i] = points[lo:hi].mean(axis=0)
    # Ramp the correction in over the first and last window so the pinned ends
    # meet the smoothed middle without a kink.
    ramp = np.clip(np.minimum(s, s[-1] - s) / max(window_m * 0.5, 1e-6), 0.0, 1.0)
    return points + (out - points) * ramp[:, None]


def _lateral_to(polyline: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perpendicular distance from each of `pts` to the `polyline`.

    Point-to-point distance is the wrong measure here: smoothing shortens the
    path, so sample i slides *backwards along the road* as well as sideways,
    and a naive p_new - p_old reports tens of metres of "shift" for a line that
    never leaves the tarmac. What matters is only how far sideways it moved.
    """
    a = polyline[:-1]
    ab = polyline[1:] - a
    denom = np.maximum((ab * ab).sum(1), 1e-12)
    out = np.empty(len(pts))
    for i, q in enumerate(pts):
        t = np.clip(((q - a) * ab).sum(1) / denom, 0.0, 1.0)
        foot = a + ab * t[:, None]
        out[i] = np.sqrt(((foot - q) ** 2).sum(1).min())
    return out


def report(before: np.ndarray, after: np.ndarray) -> None:
    def stats(p):
        length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        chord_v = p[-1] - p[0]
        chord = float(np.linalg.norm(chord_v))
        u = chord_v / max(chord, 1e-9)
        d = p - p[0]
        lat = np.linalg.norm(d - np.outer(d @ u, u), axis=1)
        return length, chord, lat

    lb, cb, latb = stats(before)
    la, ca, lata = stats(after)
    shift = _lateral_to(before, after)
    print(f"  length          {lb:8.1f}m -> {la:8.1f}m   (chord {cb:.1f}m)")
    print(f"  wasted          {lb - cb:8.1f}m -> {la - ca:8.1f}m")
    print(f"  wander sideways {latb.max():8.1f}m -> {lata.max():8.1f}m max, "
          f"{latb.mean():.1f} -> {lata.mean():.1f}m mean")
    print(f"  moved sideways from the recording: {shift.max():.2f}m max "
          f"({shift.mean():.2f}m mean)")
    if shift.max() > 8.0:
        print("  ^ that is wider than a road block. Check it still fits the "
              "track, or use a smaller --smooth.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("-o", "--out", help="default: overwrite in place")
    ap.add_argument("--smooth", type=float, default=40.0,
                    help="moving-average window in metres (default 40)")
    ap.add_argument("--chord", action="store_true",
                    help="replace the line with the straight segment between "
                         "its endpoints - for a straight-road map")
    ap.add_argument("--axis", choices=("x", "z", "auto"),
                    help="force a perfectly straight line down the dominant "
                         "axis, at the recording's MEDIAN lateral position. "
                         "Better than --chord on a straight road: the chord "
                         "inherits whatever sideways error the recording "
                         "happened to end on, and drifts across the road by "
                         "that much over the whole map.")
    ap.add_argument("--passes", type=int, default=2,
                    help="how many smoothing passes (default 2)")
    ap.add_argument("--map", help="stamp this map uid into the output")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    line = Centerline.load(a.line)
    before = line.points.copy()
    print(f"{a.line}: {len(before)} points, map uid "
          f"{line.map_uid or '(none recorded)'}")

    if a.axis:
        span = before.max(0) - before.min(0)
        axis = (0 if span[0] >= span[2] else 2) if a.axis == "auto" \
            else (0 if a.axis == "x" else 2)
        other = 2 if axis == 0 else 0
        # Median, not mean and not the endpoints: a recording that wandered
        # spends most of its samples near the middle of the road, and the
        # median ignores the excursions entirely.
        lat = float(np.median(before[:, other]))
        y = float(np.median(before[:, 1]))
        lo, hi = float(before[:, axis].min()), float(before[:, axis].max())
        # Keep the direction of travel the recording had.
        if before[0, axis] > before[-1, axis]:
            lo, hi = hi, lo
        start = np.zeros(3); end = np.zeros(3)
        start[axis], end[axis] = lo, hi
        start[other] = end[other] = lat
        start[1] = end[1] = y
        after = Centerline(np.array([start, end]), spacing=2.0).points
        print(f"  straightened along {'xz'[axis // 2]} at "
              f"{'xz'[other // 2]}={lat:.1f}, y={y:.1f}")
    elif a.chord:
        # Two points and let the resampler lay them out: spacing the original
        # sample count evenly along a *shorter* segment would slide every point
        # backwards along the road as well as sideways.
        after = Centerline(np.array([before[0], before[-1]]), spacing=2.0).points
    else:
        after = before.copy()
        for _ in range(max(1, a.passes)):
            seg = np.concatenate([[0.0], np.cumsum(
                np.linalg.norm(np.diff(after, axis=0), axis=1))])
            after = smooth(after, seg, a.smooth)

    report(before, after)

    if a.dry_run:
        print("  (dry run, nothing written)")
        return 0

    out = a.out or a.line
    uid = a.map or line.map_uid
    Centerline(after, spacing=2.0).save(out, map_uid=uid)
    print(f"  wrote {out}" + ("" if uid else
          "  -- no map uid: this line can be loaded against the WRONG track "
          "without complaint. Pass --map <uid> to fix that."))
    return 0


if __name__ == "__main__":
    sys.exit(main())

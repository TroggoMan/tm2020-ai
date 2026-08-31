"""Explore finds the finish; the racer is handed the map, not the driving.

The explore stage drives toward the map's own checkpoint landmarks with the
lidar telling it where the road is. When it finishes, it has produced two
things worth keeping, and they are not the same thing:

  1. **A path to the finish.** Proof that a route exists, and roughly where it
     goes. It is *not* a racing line - it is the wandering of a car that was
     finding the road for the first time.
  2. **The geometry of the track.** The occupancy grid from the map dump, which
     is what the lidar beams are cast against. This is objective: it is the
     map, not one car's opinion of it.

The racer is given both, and is deliberately given the first one *loosely*.

    the explored path  ->  smoothed  ->  the progress axis and the lookahead
    the occupancy grid ->  unchanged ->  the lidar
    the landmarks      ->  unchanged ->  the checkpoint gates

Why smoothed, and why loose: a line recorded by driving contains every
correction the driver made, and the racer's lookahead points and off-line
penalty both come off that line. Handing over the raw explore trace makes the
explorer's wandering the *target* - the racer would be paid to reproduce it.
So the handover smooths the path, widens the off-line tolerance and drops its
weight to near nothing. The line then does what it should: it says which way
round the track goes and how far along you are. Where the road actually is
comes from the lidar, and how fast you get round is left to the racer.

The racer also starts from **fresh weights**. Nothing is copied from the
explore policy - it was trained on a different reward for a different job, and
seeding from it is the other way to accidentally inherit the wandering.
"""
from __future__ import annotations

import json
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from env.centerline import Centerline

# What the race stage's line-following is relaxed to at handover. Wide and
# weak: the explored line is a suggestion about the route, and charging the
# usual per-metre penalty against it would make "drive where the explorer
# drove" the objective.
RACE_PROFILE = {
    "max_offset": 60.0,
    "soft_offset": 28.0,
    "w_soft": 0.002,
}


class HandoverWatch(BaseCallback):
    """Stop the explore stage once it has found the finish and stopped
    improving.

    Two conditions, because either alone is wrong. "Finished N times" alone
    hands over a route the car stumbled into once and cannot repeat; "stopped
    improving" alone never triggers on a stage that has not finished at all.

    :param finishes: how many finishes before handover is even considered.
    :param patience: episodes without a new best time, after that many
        finishes, before calling it done.
    """

    def __init__(self, finishes: int = 3, patience: int = 25, verbose: int = 0):
        super().__init__(verbose)
        self.target = int(finishes)
        self.patience = int(patience)
        self.finishes = 0
        self.best_ms: int | None = None
        self.since_best = 0
        self.done = False

    def _on_step(self) -> bool:
        for info in (self.locals.get("infos") or []):
            if not info or not info.get("finished"):
                continue
            self.finishes += 1
            t = info.get("race_time")
            if t is not None and (self.best_ms is None or int(t) < self.best_ms):
                self.best_ms = int(t)
                self.since_best = 0
                print(f"  explore: finish #{self.finishes} in "
                      f"{self.best_ms / 1000.0:.3f}s (new best)", flush=True)
            else:
                self.since_best += 1
                print(f"  explore: finish #{self.finishes}, no improvement "
                      f"({self.since_best}/{self.patience})", flush=True)

        if (not self.done and self.finishes >= self.target
                and self.since_best >= self.patience):
            self.done = True
            print(f"\nexplore stage done: {self.finishes} finishes, best "
                  f"{self.best_ms / 1000.0:.3f}s, no improvement in "
                  f"{self.patience} finishes. Handing over.\n", flush=True)
            return False
        return True


def best_trace(root: str, map_uid: str) -> str | None:
    """The finished trace with the fastest race time, else the longest one.

    Preferring a finished run matters more than it looks: an unfinished trace
    stops wherever the car died, and a line built from it sends the racer
    confidently into whatever killed the explorer.
    """
    d = os.path.join(root, "runs", map_uid, "traces")
    if not os.path.isdir(d):
        return None
    best, best_key = None, None
    for name in os.listdir(d):
        if not name.endswith(".json"):
            continue
        path = os.path.join(d, name)
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        finished = bool(data.get("finished"))
        t = data.get("race_time") or 0
        dist = data.get("distance") or len(data.get("samples") or [])
        # Sort key: finished first, then fastest, then furthest.
        key = (1 if finished else 0, -float(t) if finished else 0.0, float(dist))
        if best_key is None or key > best_key:
            best, best_key = path, key
    return best


def smooth_points(points: np.ndarray, window_m: float = 30.0,
                  passes: int = 2) -> np.ndarray:
    """Moving average over arc length, ends pinned.

    Same operation as tools/straighten_line.py --smooth, inlined so the
    handover cannot silently drift from it. Ends are pinned because they are
    the spawn and the finish: move either sideways and every projection after
    that is measured from the wrong place.
    """
    pts = np.asarray(points, dtype=np.float64)
    for _ in range(max(1, passes)):
        s = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        out = pts.copy()
        for i in range(len(pts)):
            lo = np.searchsorted(s, s[i] - window_m * 0.5)
            hi = np.searchsorted(s, s[i] + window_m * 0.5)
            if hi - lo >= 2:
                out[i] = pts[lo:hi].mean(axis=0)
        ramp = np.clip(np.minimum(s, s[-1] - s) / max(window_m * 0.5, 1e-6),
                       0.0, 1.0)
        pts = pts + (out - pts) * ramp[:, None]
    return pts


def build_race_line(root: str, map_uid: str, smooth_m: float = 30.0,
                    out: str | None = None) -> str | None:
    """Turn the best explore trace into the race stage's reference line."""
    trace = best_trace(root, map_uid)
    if not trace:
        print(f"handover: no traces for {map_uid} - nothing to hand over",
              flush=True)
        return None
    with open(trace) as f:
        data = json.load(f)
    # The trace names its own columns, so read the header rather than assuming
    # x/y/z are at 1,2,3. They are today; a new column at the front would
    # otherwise silently build the line out of the wrong three numbers, and a
    # line that is subtly wrong is far worse than one that fails loudly.
    rows = data.get("samples") or data.get("rows") or data.get("trace") or []
    fields = data.get("fields") or ["t", "x", "y", "z"]
    try:
        ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError:
        print(f"handover: {os.path.basename(trace)} has no x/y/z columns "
              f"({fields})", flush=True)
        return None
    width = max(ix, iy, iz) + 1
    pts = np.asarray([[r[ix], r[iy], r[iz]] for r in rows if len(r) >= width],
                     dtype=np.float64)
    if len(pts) < 8:
        print(f"handover: {os.path.basename(trace)} has too few points",
              flush=True)
        return None

    raw = Centerline(pts, spacing=2.0)
    line = Centerline(smooth_points(raw.points, smooth_m), spacing=2.0)
    out = out or os.path.join(root, "lines", f"{map_uid}-explored.json")
    line.save(out, map_uid=map_uid)

    turn_before = _turning(raw.points)
    turn_after = _turning(line.points)
    print(f"handover: {os.path.basename(trace)} -> {out}", flush=True)
    print(f"  {raw.length:.0f}m of driving, smoothed to {line.length:.0f}m; "
          f"{turn_before:.0f} degrees of steering became {turn_after:.0f}",
          flush=True)
    print(f"  the racer gets this as a ROUTE, not a line to trace: "
          f"soft_offset {RACE_PROFILE['soft_offset']:.0f}m at "
          f"w_soft {RACE_PROFILE['w_soft']}", flush=True)
    return out


def _turning(pts: np.ndarray) -> float:
    t = np.diff(pts, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return float(np.degrees(np.arccos(
        np.clip((t[:-1] * t[1:]).sum(1), -1, 1))).sum())

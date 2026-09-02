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

    the ROUTE MODEL   ->  unchanged ->  the progress axis and the lookahead
    the occupancy grid ->  unchanged ->  the lidar
    the landmarks      ->  unchanged ->  the checkpoint gates

WHICH LINE, and why it matters more than it looks. 32 of the 148 observation
dims are line-relative: six lookahead points (xyz each), `offset`,
`sides_ahead`, `margin`, `width_ahead`. Hand the racer a DIFFERENT line from
the one the explorer trained against and all 32 shift at once, so the policy
has to rediscover the spatial relationship - it visibly relearns the track,
which is the opposite of the point of an explore stage.

So the racer is handed the route model itself (roadtrace plus any panel edits)
whenever roadtrace covered the map, and only falls back to a smoothed
best-trace line when it did not. Measured on Summer 2026-06: the trace-derived
line sat a median 4.0m from the roadtrace line (18.8m at worst) and the racer
crawled; on the same geometry (0.21m) it reached CP3 in its first three
episodes and finished on its fourth.

Looseness is a separate lever, and still applied: RACE_PROFILE widens the
off-line tolerance and drops its weight to near nothing, so the line says which
way round the track goes and how far along you are, without the racer being
paid to trace it. Where the road actually is comes from the lidar, and how fast
you get round is left to the racer.

The racer INHERITS the explore weights: do_handover execs the race stage with
`--init-from <explore model>`. This docstring claimed the opposite - "starts
from fresh weights, nothing is copied" - for long enough to mislead a reading
of the code, so check train_sac.do_handover rather than trusting prose here.

The point of the loose line above is what stops the wandering being inherited,
and it does not depend on the weights being fresh: the explore policy's basic
car control (throttle, not spinning, staying on a road) is worth keeping, and
retraining it from scratch on the race reward wastes it. What must not be
inherited is the explored PATH as a target, which the smoothing and the
near-zero `w_soft` handle.
"""
from __future__ import annotations

import json
import os
import time

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

    #: Where the live countdown is published for the panel and the stream
    #: overlays. A file rather than log-scraping: the "will stop after N" line
    #: is printed once at startup and scrolls out of the stdout tail within
    #: minutes, so anything parsing the tail loses the target and can only
    #: guess at it.
    STATE_FILE = "handover.json"

    def __init__(self, finishes: int = 3, patience: int = 25, verbose: int = 0):
        super().__init__(verbose)
        self.target = int(finishes)
        self.patience = int(patience)
        self.finishes = 0
        self.best_ms: int | None = None
        self.since_best = 0
        self.done = False
        self.state_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", self.STATE_FILE)
        self._publish()

    def _publish(self) -> None:
        """Write the countdown to disk. Never allowed to break a run."""
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "finishes": self.finishes,
                    "target": self.target,
                    "since_best": self.since_best,
                    "patience": self.patience,
                    "best_ms": self.best_ms,
                    "done": self.done,
                    # What is still outstanding, so a reader does not have to
                    # re-derive the two-condition rule.
                    "finishes_left": max(0, self.target - self.finishes),
                    "stall_left": max(0, self.patience - self.since_best),
                    "updated": time.time(),
                }, f)
            os.replace(tmp, self.state_path)
        except (OSError, TypeError):
            pass

    def _on_step(self) -> bool:
        saw_finish = False
        for info in (self.locals.get("infos") or []):
            if not info or not info.get("finished"):
                continue
            saw_finish = True
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

        # Only on a finish: this runs every control step, and rewriting the
        # file 40 times a second would be pure churn for a number that cannot
        # have changed.
        if saw_finish:
            self._publish()

        if (not self.done and self.finishes >= self.target
                and self.since_best >= self.patience):
            self.done = True
            self._publish()
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


def route_model_line(root: str, map_uid: str, min_coverage: float = 0.75,
                     out: str | None = None) -> str | None:
    """Hand the racer the line the EXPLORER ACTUALLY TRAINED AGAINST.

    This is preferred over re-deriving one from a trace, and the reason is
    measurable. 32 of the 148 observation dims are line-relative - the six
    lookahead points (xyz each), `offset`, `sides_ahead`, `margin` and
    `width_ahead`. Hand the policy a different line and all 32 shift at once.

    Measured on Summer 2026-06: the smoothed best-trace line sat a median 4.0m
    from the roadtrace line the explorer trained on, 9.6m at p90 and 18.8m at
    worst, and was 1713m against 1835m. The racer visibly relearned the track
    from scratch - which is the exact opposite of what the explore stage is
    for. On the same geometry (median separation 0.21m) it reached CP3 on its
    first three episodes and finished on its fourth.

    build_race_line's assumption - that an explorer's line is provisional junk
    cutting through scenery - holds for the PROVISIONAL layer, not for
    roadtrace, which follows the map's own block ribbon and carries half_width
    and sides natively. When roadtrace covered the map well, re-deriving a line
    from one driven lap throws away better geometry AND breaks the transfer.

    Returns None when there is no good-enough roadtrace, so the caller can fall
    back to the trace-derived line.
    """
    src = os.path.join(root, "maps", f"{map_uid}.roadtrace.json")
    try:
        with open(src) as f:
            rt = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    cover = float(rt.get("coverage") or 0.0)
    pts = rt.get("points") or []
    if cover < min_coverage or len(pts) < 8:
        print(f"handover: roadtrace coverage {cover:.0%} (<{min_coverage:.0%}) "
              f"- falling back to the explored trace", flush=True)
        return None

    # Splice the panel's route edits in, exactly as env.routemodel does, so the
    # racer gets what the explorer had - hand-drawn corrections included.
    arr = np.asarray([p[:3] for p in pts], dtype=np.float64)
    try:
        from env.routemodel import load_edits, _apply_patches
        patches = load_edits(root, map_uid)
        if patches:
            arr, _ = _apply_patches(arr, patches)
            print(f"handover: spliced {len(patches)} route edit(s)", flush=True)
    except Exception as ex:                                # noqa: BLE001
        print(f"handover: could not splice route edits ({ex})", flush=True)

    line = Centerline(arr, spacing=2.0)
    out = out or os.path.join(root, "lines", f"{map_uid}-roadtrace.json")
    line.save(out, map_uid=map_uid)
    print(f"handover: route model -> {out}", flush=True)
    print(f"  {line.length:.0f}m, {len(line.points)} pts, roadtrace coverage "
          f"{cover:.0%} - the SAME geometry the explorer trained against, so "
          f"the driving transfers instead of being relearned", flush=True)
    return out


def build_race_line(root: str, map_uid: str, smooth_m: float = 30.0,
                    out: str | None = None) -> str | None:
    """The race stage's reference line.

    Prefers the route model (see route_model_line); only smooths a driven trace
    when there is no usable roadtrace for the map.
    """
    from_route = route_model_line(root, map_uid, out=out)
    if from_route:
        return from_route
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

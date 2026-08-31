#!/usr/bin/env python3
"""Record a reference line - either by driving, or by watching a replay.

Both work through the same path. Openplanet's VehicleState API reads whatever
vehicle is currently being *viewed*, and its own docs note the state stays valid
while spectating, so watching a replay produces exactly the same telemetry as
driving. That means a saved replay is a first-class source and you never have to
drive a lap just to get a line.

    # from a replay (best): load the replay in game, hit play, then
    python3 tools/record_line.py lines/spring2026-03.json

    # from your own driving
    python3 tools/record_line.py lines/spring2026-03.json --mode live

Modes:
    auto    (default) record whatever is being viewed once it starts moving -
            works for a replay, a ghost, or your own driving
    live    only record when YOU are spawned and racing
    replay  only record when it is not your own live run

It also captures the viewed car's inputs, so a replay recording doubles as
demonstration data for seeding SAC's replay buffer later.

Stops on Ctrl-C, when the run finishes, or after --idle seconds without movement.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.centerline import Centerline
from env.tm_env import TelemetryLink

UI_PLAYING = 1
SPAWN_SPAWNED = 2


def pick_vehicle(rec: dict) -> dict | None:
    """Fall back to the scene's vehicle list.

    When you watch a ghost on the track there is no viewing player, and
    GetSingularVis() only works when the scene holds exactly one car - so
    ViewingPlayerState() returns null and the top-level fields are empty even
    though the ghost is right there. GetAllVis still sees it, so take the
    fastest distinct vehicle: that is the one actually running the lap.
    """
    vs = rec.get("vehicles") or []
    seen, uniq = set(), []
    for v in vs:
        key = tuple(round(c, 3) for c in (v.get("pos") or []))
        if key and key not in seen:
            seen.add(key)
            uniq.append(v)
    if not uniq:
        return None
    return max(uniq, key=lambda v: v.get("speed") or 0.0)


def as_sample(rec: dict) -> dict | None:
    """One telemetry record -> the fields we record, from whichever source has
    them: the player/viewed car if present, else the scene vehicle list."""
    if rec.get("car"):
        return rec
    v = pick_vehicle(rec)
    if v is None:
        return None
    merged = dict(v)
    merged["race_time"] = rec.get("race_time")
    merged["ui"] = rec.get("ui")
    merged["finished"] = rec.get("finished")
    merged["from_vehicles"] = True
    return merged


def usable(rec: dict, mode: str) -> bool:
    if not rec:
        return False
    if not rec.get("car") and not (rec.get("vehicles") or []):
        return False
    # SpawnStatus is unreliable (reads NotSpawned while driving), so "live"
    # means the UI says we are playing and the car is real.
    live = rec.get("ui") == UI_PLAYING and rec.get("source") != "viewed"
    if mode == "live":
        return live
    if mode == "replay":
        return not live
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="where to write the line, e.g. lines/track.json")
    ap.add_argument("--mode", choices=("auto", "live", "replay"), default="auto")
    ap.add_argument("--min-step", type=float, default=0.5,
                    help="metres between kept points")
    ap.add_argument("--idle", type=float, default=0.0,
                    help="stop after this many seconds without movement (0 = never)")
    ap.add_argument("--lap-gap", type=float, default=30.0,
                    help="metres of positional jump that counts as a restart")
    ap.add_argument("--laps", type=int, default=1,
                    help="stop after this many complete laps (0 = never)")
    ap.add_argument("--expect-time", type=float, default=None,
                    help="the replay's time in seconds, to sanity-check the capture")
    ap.add_argument("--demo", metavar="FILE",
                    help="also write (state, action) samples here for demo seeding")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # A background process started from a non-interactive shell inherits
    # SIGINT ignored, so Ctrl-C style stops never arrive. Handle SIGTERM too,
    # and turn it into the same clean exit that writes the file out.
    def _term(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    link = TelemetryLink()
    print(f"Connected ({args.mode} mode). Drive, or play a replay. Ctrl-C to stop.\n",
          flush=True)

    pts: list[np.ndarray] = []
    demo: list[dict] = []
    last_t = None
    moving_since = None
    last_move = time.time()
    t_first = t_last = None
    laps = 0
    seen: dict[tuple, int] = {}
    last_report = 0.0

    try:
        while True:
            time.sleep(0.01)
            raw_rec = link.get()
            rec = as_sample(raw_rec) if raw_rec else None
            if rec is not None:
                rec = dict(rec)
                rec["t"] = raw_rec.get("t")
            if not usable(raw_rec, args.mode) or rec is None:
                rec = raw_rec
                # Say why nothing is being captured, rather than sitting
                # silently - the mode filter is the usual culprit.
                if rec:
                    key = (bool(rec.get("car")), rec.get("ui"),
                           len(rec.get("vehicles") or []))
                    seen[key] = seen.get(key, 0) + 1
                if time.time() - last_report > 3.0:
                    last_report = time.time()
                    states = ", ".join(
                        f"car={c} ui={u} vehicles={s} (x{n})"
                        for (c, u, s), n in sorted(
                            seen.items(), key=lambda kv: -kv[1])[:3])
                    print(f"\rwaiting [{args.mode}] - seeing: {states or 'nothing'}",
                          end="", flush=True)
                continue
            if rec.get("t") == last_t:
                continue          # same frame, plugin hasn't ticked yet
            last_t = rec.get("t")
            # race_time comes from the player API, which does not exist when
            # watching a ghost. Fall back to the plugin's own clock, which for
            # a replay is the same elapsed time.
            clock = rec.get("race_time")
            if clock is None:
                clock = rec.get("t")
            if clock is not None:
                if t_first is None:
                    t_first = clock
                t_last = clock

            speed = rec.get("speed", 0.0)
            if moving_since is None:
                if speed < 1.0:
                    continue      # don't start on a stationary car
                moving_since = time.time()

            p = rec.get("pos")
            if not p:
                continue
            p = np.asarray(p, dtype=float)

            # A big jump between consecutive samples means the ghost was
            # teleported back to the start - i.e. the run restarted. That is
            # the only lap boundary available when there is no player API to
            # give us a "finished" flag.
            if pts and np.linalg.norm(p - pts[-1]) > args.lap_gap:
                laps += 1
                print(f"\n  lap {laps} complete: {len(pts)} points")
                if args.laps and laps >= args.laps:
                    break
                pts.clear()
                if args.demo:
                    demo.clear()
                t_first = None

            if not pts or np.linalg.norm(p - pts[-1]) > args.min_step:
                pts.append(p)
                last_move = time.time()
                if args.demo:
                    demo.append({
                        "pos": rec.get("pos"), "vel": rec.get("vel"),
                        "dir": rec.get("dir"), "up": rec.get("up"),
                        "left": rec.get("left"), "speed": speed,
                        "gear": rec.get("gear"), "rpm": rec.get("rpm"),
                        "slip": rec.get("slip"), "adherence": rec.get("adherence"),
                        "ground": rec.get("ground"),
                        "steer": rec.get("in_steer"), "gas": rec.get("in_gas"),
                        "brake": rec.get("in_brake"),
                        "race_time": rec.get("race_time"),
                    })
                print(f"\r{len(pts):5d} points   t={(clock or 0)/1000:7.2f}s"
                      f"   {speed*3.6:5.0f} km/h"
                      f"   {'ghost' if rec.get('from_vehicles') else 'player'}",
                      end="", flush=True)

            if rec.get("finished"):
                print("\nRun finished.")
                break
            # Only arm the idle timeout once something has actually moved, or
            # it fires while you are still loading the replay.
            if args.idle and pts and time.time() - last_move > args.idle:
                print(f"\nNo movement for {args.idle:.0f}s - stopping.")
                break
    except KeyboardInterrupt:
        print()
    finally:
        link.close()

    if len(pts) < 10:
        print(f"Only {len(pts)} points - not enough for a line. "
              f"Was anything actually moving?")
        return 1

    line = Centerline(np.asarray(pts))
    line.save(args.out)
    dur = ((t_last - t_first) / 1000.0) if (t_first is not None
                                            and t_last is not None) else None
    print(f"Wrote {args.out}: {len(line.points)} samples, {line.length:.0f}m")
    if dur is not None:
        avg = line.length / dur * 3.6 if dur > 0 else 0
        print(f"Captured {dur:.3f}s of running, avg {avg:.0f} km/h")
    if args.expect_time and dur:
        diff = dur - args.expect_time
        verdict = "looks right" if abs(diff) < max(1.5, args.expect_time * 0.1) \
            else "MISMATCH - did it capture the whole run?"
        print(f"Replay time {args.expect_time:.3f}s vs captured {dur:.3f}s "
              f"({diff:+.3f}s) - {verdict}")

    if args.demo and demo:
        os.makedirs(os.path.dirname(args.demo) or ".", exist_ok=True)
        with open(args.demo, "w") as f:
            json.dump(demo, f)
        print(f"Wrote {args.demo}: {len(demo)} demonstration samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())

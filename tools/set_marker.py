#!/usr/bin/env python3
"""Place a reward marker where the car is standing.

Markers are one-off bonuses at hand-picked distances along the reference line,
paid once per episode when the car's furthest progress passes them. They exist
to shorten CREDIT-ASSIGNMENT DISTANCE across a section the policy will not
commit to: measured on Summer 2026-06, half the episodes reaching CP4 stopped
dead at the entrance to a 194m unbarriered platform run, because for entering
to look worthwhile the critic had to carry CP5's bonus back across all 194m
while every sample of that stretch ended badly.

Driving to the spot is the only sane way to choose one - a distance along a
traced line is not something you can eyeball off a map.

    tools/set_marker.py                     where am I? (does not write)
    tools/set_marker.py --add --bonus 200   add a marker here
    tools/set_marker.py --list              show the markers on this map
    tools/set_marker.py --clear             remove them all

The trainer must be STOPPED (it owns the pads), and you drive the car yourself.
Marker positions are read through the config hot-reload, so a running trainer
picks up an edit within a second - but the trainer cannot be driving while you
are.

PER-TRACK SCAFFOLDING. Markers belong to one hard section of one map. Remove
them once the section is learned; the goal is a driver that does not need them.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def live_position(timeout: float = 8.0):
    """{seat: pos} for every spawned car, and the map uid."""
    from env.ports import broker_addr
    host, port = broker_addr(0)
    try:
        sock = socket.create_connection((host, port), timeout=5)
    except OSError as ex:
        sys.exit(f"no broker on {host}:{port}: {ex}\n"
                 f"Is the game up and the plugin loaded?")
    sock.settimeout(timeout)
    f = sock.makefile("r")
    try:
        for _ in range(400):
            line = f.readline()
            if not line:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Prefer the seat the camera is on; in splitscreen the top-level
            # record describes the camera's car.
            players = rec.get("players") or []
            seats = {}
            for p in players:
                pos = p.get("pos")
                if pos and any(abs(float(c)) > 1e-6 for c in pos):
                    slot = p.get("slot")
                    if slot is None:
                        continue
                    seats[int(slot)] = pos
            if not seats and rec.get("pos"):
                seats[0] = rec["pos"]
            if not seats:
                continue
            return seats, rec.get("map")
    finally:
        sock.close()
    sys.exit("no position in the telemetry - is a map loaded and a car spawned?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="store_true", help="add a marker here")
    ap.add_argument("--bonus", type=float, default=200.0,
                    help="reward for reaching it (default 200)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--seat", type=int, default=None,
                    help="which car to take the position from. Default: the "
                         "one FURTHEST along the line, which is the one you "
                         "drove to the spot.")
    a = ap.parse_args()

    from env.config import config_path

    seats, uid = live_position()
    if not uid:
        sys.exit("telemetry carried no map uid - reload the plugin")

    cfg_path = config_path(ROOT, uid, "explore")
    try:
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        cfg = {}
    markers = cfg.get("markers") or []

    if a.list or a.clear:
        if a.clear:
            cfg["markers"] = []
            with open(cfg_path, "w") as fh:
                json.dump(cfg, fh, indent=2)
            print(f"cleared {len(markers)} marker(s) from "
                  f"{os.path.basename(cfg_path)}")
            return 0
        if not markers:
            print("no markers on this map")
        for i, m in enumerate(sorted(markers, key=lambda x: x.get("s", 0))):
            print(f"  {i}: {m.get('s'):>7.0f}m  +{m.get('bonus', 0):.0f}")
        return 0

    # Where is that, along the line the TRAINER actually uses?
    #
    # This must be the roadtrace cache, not routemodel.merged_line(): without
    # the occupancy dump and the gates - neither of which this tool has - that
    # function silently falls back to the `learned` layer, which on this map is
    # 626m against roadtrace's 1836m. A marker measured against the wrong line
    # is not slightly off, it is meaningless.
    import numpy as np
    from env.centerline import Centerline
    trace = os.path.join(ROOT, "maps", f"{uid}.roadtrace.json")
    if not os.path.isfile(trace):
        sys.exit(f"no roadtrace cache for {uid} ({trace}).\n"
                 f"The trainer builds it from the occupancy dump - run a "
                 f"training session on this map first, or survey it.")
    with open(trace) as fh:
        rm = json.load(fh)
    pts = rm.get("points")
    if not pts:
        sys.exit(f"{trace} has no points")
    line = Centerline(np.asarray(pts, dtype=np.float64)[:, :3])
    print(f"line     roadtrace, {len(pts)} pts, {float(line.s[-1]):.0f}m "
          f"(coverage {rm.get('coverage', '?')})")
    # Project every car, so the choice is visible rather than implicit.
    proj = {}
    for slot, p in sorted(seats.items()):
        i, dist, off = line.project_near(np.asarray(p, dtype=np.float64), None)
        proj[slot] = (dist, off, p)

    print(f"map      {uid}")
    print("cars:")
    for slot, (dist, off, p) in proj.items():
        print(f"   seat {slot}: {dist:7.0f}m  offset {off:5.1f}m  "
              f"{[round(float(c), 1) for c in p]}")

    if a.seat is not None:
        if a.seat not in proj:
            sys.exit(f"seat {a.seat} has no spawned car "
                     f"(have {sorted(proj)})")
        pick = a.seat
        why = f"--seat {a.seat}"
    else:
        # Furthest along: the car you drove to the spot. Picking "the first
        # player with a position" instead would silently choose whichever seat
        # happened to come first in the telemetry array.
        pick = max(proj, key=lambda k: proj[k][0])
        why = "furthest along"
    s, offset, pos = proj[pick][0], proj[pick][1], proj[pick][2]
    print(f"\nusing seat {pick} ({why})")
    print(f"on the line at {s:.0f}m of {float(line.s[-1]):.0f}m "
          f"(offset {offset:.1f}m)")

    sides = rm.get("sides")
    if sides is not None:
        try:
            # Truncate to len(line.s): Centerline's cumulative-distance array
            # is one shorter than the point list it was built from, and
            # np.interp refuses mismatched lengths ("fp and xp are not of the
            # same length"). The trainer never hits this because merged_line
            # resamples sides to match the Centerline it builds.
            sd = np.asarray(sides, dtype=np.float64)[:len(line.s)]
            here = float(np.interp(s, line.s, sd))
            print(f"containment here: {here:.2f} "
                  f"({'UNBARRIERED - drive off the edge' if here < 0.25 else 'barriered'})")
        except Exception:                                  # noqa: BLE001
            pass

    if not a.add:
        print("\n(nothing written - pass --add to place a marker here)")
        return 0

    markers.append({"s": round(float(s), 1), "bonus": float(a.bonus)})
    markers.sort(key=lambda m: m["s"])
    cfg["markers"] = markers
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"\nadded marker at {s:.0f}m (+{a.bonus:.0f}) -> "
          f"{os.path.basename(cfg_path)}")
    print(f"markers now: " + ", ".join(f"{m['s']:.0f}m+{m['bonus']:.0f}"
                                      for m in markers))
    return 0


if __name__ == "__main__":
    sys.exit(main())

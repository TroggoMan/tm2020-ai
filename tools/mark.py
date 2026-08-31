#!/usr/bin/env python3
"""Record where the car is right now, under a name you can use in a hint.

    tools/mark.py before-the-jump          # drive there first, then run this
    tools/mark.py --list
    tools/mark.py --delete before-the-jump

The point of this is that you should not have to guess arc lengths. Drive to
the place you care about, name it, and then a hint can say

    {"name": "tap-into-the-left", "mark_from": "before-the-jump",
     "mark_to": "after-the-jump", "hold_ms": 120, "w": 0.05}

The mark is stored as a **world position**, not as a distance along the line,
because arc length belongs to whichever reference line happens to be loaded
and lines get re-recorded. Storing the place means the mark still points at
the same corner after the line is straightened or rebuilt by the explore
stage - which is exactly when a hardcoded `s_from: 812.5` would quietly start
pointing somewhere else.

Marks live in the map's tuning config, so they are per map and they hot-reload
like everything else there.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.ports import broker_addr   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def telemetry(instance: int, timeout: float = 5.0) -> dict:
    """One telemetry record from the broker."""
    with socket.create_connection(broker_addr(instance), timeout=timeout) as s:
        s.settimeout(timeout)
        buf = b""
        while b"\n" in buf or len(buf) < 1_000_000:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "pos" in rec:
                    return rec
    raise SystemExit("no telemetry with a position - is a map loaded?")


def config_for(uid: str | None, profile: str = "") -> str:
    from env.config import config_path
    return config_path(ROOT, uid, profile)


def load(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("-i", "--instance", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--delete")
    ap.add_argument("--map", help="map uid (default: whatever is loaded)")
    a = ap.parse_args()

    uid = a.map
    rec = None
    if uid is None or a.name:
        rec = telemetry(a.instance)
        uid = uid or rec.get("map")
    path = config_for(uid)
    data = load(path)
    marks = data.setdefault("marks", {})

    if a.list:
        if not marks:
            print(f"no marks in {os.path.basename(path)}")
            return 0
        print(f"{os.path.basename(path)}:")
        for k, v in sorted(marks.items()):
            print(f"  {k:<24} {v[0]:8.1f} {v[1]:6.1f} {v[2]:8.1f}")
        return 0

    if a.delete:
        if marks.pop(a.delete, None) is None:
            print(f"no mark called {a.delete!r}")
            return 1
        save(path, data)
        print(f"removed {a.delete}")
        return 0

    if not a.name:
        ap.error("give the mark a name, or use --list / --delete")

    pos = [round(float(x), 2) for x in rec.get("pos", [0, 0, 0])]
    speed = float(rec.get("speed", 0.0)) * 3.6
    if speed > 5.0:
        print(f"note: the car is doing {speed:.0f}km/h - the mark is where it "
              f"was when you ran this, not where you meant to stop.")
    marks[a.name] = pos
    save(path, data)
    print(f"{a.name} = {pos}  (written to {os.path.basename(path)})")
    print(f"use it as  \"mark_from\": \"{a.name}\"  in a hint")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Live surface readout - what the wheels are actually on, as you drive.

    tools/watch_surfaces.py --port 8767        report each new surface
    tools/watch_surfaces.py --all              every sample (very noisy)
    tools/watch_surfaces.py --hold 0.4         how long a surface must persist

Prints a line when the set of surfaces under the wheels genuinely changes, and
a running tally on Ctrl-C.

Two pieces of noise this deliberately suppresses, both learned the hard way
watching a real drive:

  * `XXX_Null` is the enum's "nothing" entry - a wheel that is OFF THE GROUND,
    not a surface. On anything bumpy (dirt especially) wheels leave the ground
    constantly, so treating null as a surface fired a "change" on every micro
    bounce: 40 lines in 4 seconds, all of them "Dirt" vs "Dirt + XXX_Null".
    Airborne wheels are now ignored for the purpose of naming the surface, and
    reported separately as an airborne fraction.

  * A genuine surface still flickers at a boundary - straddling a road edge
    alternates Asphalt / Asphalt+Grass many times a second. A new reading has
    to persist for `--hold` seconds before it is announced, so a crossing
    prints once rather than fifteen times.
"""
import argparse
import collections
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.surfaces import material_name  # noqa: E402

# The enum's null entry: this wheel is touching nothing.
AIRBORNE = "XXX_Null"


def wheels(rec):
    """(materials, speed, slot) for whichever record carries them."""
    if rec.get("mat"):
        return rec["mat"], rec.get("speed"), rec.get("slot")
    for p in (rec.get("players") or []):
        if p.get("mat"):
            return p["mat"], p.get("speed"), p.get("slot")
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767,
                    help="8767 = broker (fans out to many clients), "
                         "8766 = the plugin directly (ONE client only)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seat", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--hold", type=float, default=0.4,
                    help="seconds a reading must persist before it is printed")
    a = ap.parse_args()

    s = socket.create_connection((a.host, a.port), timeout=10)
    s.settimeout(60)
    print(f"watching surfaces on {a.host}:{a.port}  "
          f"(airborne wheels ignored, {a.hold}s debounce)\n", flush=True)

    tally = collections.Counter()
    air = [0, 0]
    buf = b""
    shown = None            # last reading actually printed
    cand, cand_since = None, 0.0
    since = time.time()
    t0 = time.time()
    try:
        while True:
            d = s.recv(65536)
            if not d:
                break
            buf += d
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1)
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                m, spd, slot = wheels(rec)
                if not m:
                    continue
                if a.seat is not None and slot != a.seat:
                    continue

                names = [material_name(v) for v in m]
                air[1] += len(names)
                air[0] += sum(1 for n in names if n == AIRBORNE)
                grounded = [n for n in names if n != AIRBORNE]
                for n in grounded:
                    tally[n] += 1
                if not grounded:
                    continue                     # fully airborne: not a surface

                key = tuple(sorted(set(grounded)))
                now = time.time()
                if a.all:
                    cand, cand_since = key, now
                elif key != cand:
                    cand, cand_since = key, now
                    continue
                elif now - cand_since < a.hold or key == shown:
                    continue

                kmh = (spd or 0) * 3.6
                tail = f"   (previous held {now - since:5.1f}s)" if shown else ""
                print(f"{now-t0:8.1f}  {kmh:6.0f} km/h   {' + '.join(key)}{tail}",
                      flush=True)
                shown, since = key, now
    except KeyboardInterrupt:
        pass
    finally:
        s.close()
        total = sum(tally.values()) or 1
        print("\n--- wheel-contact tally (grounded wheels only) ---")
        for name, n in tally.most_common():
            print(f"  {name:18s} {100*n/total:5.1f}%   ({n} contacts)")
        if air[1]:
            print(f"  wheels airborne:   {100*air[0]/air[1]:5.1f}% of wheel-samples")


if __name__ == "__main__":
    sys.exit(main())

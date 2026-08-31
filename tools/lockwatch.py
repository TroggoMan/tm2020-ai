#!/usr/bin/env python3
"""Log the telemetry rate over time, so a screen lock can be tested.

The plugin's send rate is FRAME-bound, not physics-bound - the game ticks
physics at 100Hz regardless, but the plugin emits from a render callback. So
this number tracks the game's frame rate, and the frame rate is what a
compositor throttles when a window is unfocused or occluded.

That matters because TM2020 ramps analogue steering per FRAME. Fewer frames
means the ramp never reaches full deflection, so the policy's actions land
softened - we measured ~90% applied merely from losing focus. A locked screen
occludes the window completely, which is a stronger version of the same thing.

    tools/lockwatch.py --minutes 5      then lock the screen and wait

Writes one line per sample to logs/lockwatch.log with a wall-clock timestamp,
so the drop (if any) can be lined up against when the lock happened.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.ports import broker_addr   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--window", type=float, default=3.0,
                    help="seconds per sample")
    a = ap.parse_args()

    out = os.path.join(ROOT, "logs", "lockwatch.log")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sock = socket.create_connection(broker_addr(0), 5)
    sock.settimeout(a.window + 2)
    buf = b""
    end = time.time() + a.minutes * 60
    with open(out, "a", buffering=1) as f:
        f.write(f"\n--- lockwatch started {time.strftime('%F %T')} ---\n")
        while time.time() < end:
            n = 0
            t0 = time.time()
            while time.time() - t0 < a.window:
                try:
                    c = sock.recv(65536)
                except socket.timeout:
                    break
                if not c:
                    break
                buf += c
                lines = buf.split(b"\n")
                buf = lines[-1]
                for line in lines[:-1]:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "car" in d:
                        n += 1
            hz = n / max(time.time() - t0, 1e-6)
            f.write(f"{time.strftime('%H:%M:%S')}  {hz:6.1f} Hz\n")
    sock.close()
    print(f"done - see {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

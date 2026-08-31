#!/usr/bin/env python3
"""Which virtual pad drives which splitscreen seat?

Nothing tells us. The game assigns controllers to seats in whatever order it
saw them, and that order is not the order we created the uinput devices in.
Guessing produces a training run where four policies steer the wrong cars, and
the symptom - every car "stuck" - looks exactly like a policy that cannot
drive.

So we ask the game. Full lock AND full throttle on one pad at a time, and
whichever seat moves is the seat that pad owns.

The throttle is not decoration. A stationary Trackmania car does not turn, and
the game reports the steering it *applied* rather than the steering it was
handed - so a working pad reads `in_steer 0.0` against a parked car and the
test says "nothing happened" about a pad that is fine. Movement is checked as
well as steering, and distance is the most reliable of the three because it is
cumulative and so survives a sample landing between telemetry frames.

    tools/calibrate_seats.py            report the mapping
    tools/calibrate_seats.py --json     just the mapping, for a script

A pad that moves nothing is not bound to any seat. That is an in-game step:
each seat has to have a controller assigned to it in the game's own controller
settings before it will spawn a car that answers to anything.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.ports import MAX_SEATS, seat_ports   # noqa: E402


def read_players(sock, sock_file, timeout: float = 2.0) -> list[dict]:
    """The MOST RECENT players array, not the next one in the buffer.

    This is the bug that made a working pad report "nothing happened". The
    plugin streams at ~100Hz, so holding a pad down for 1.6 seconds leaves
    about 160 records queued in the socket. Reading "the next record with
    players" then returns the state from the *start* of the hold - before the
    car had moved - and the comparison is against itself.

    So: drain everything already buffered and keep the last complete record.
    """
    latest: list[dict] = []
    deadline = time.time() + timeout
    while True:
        ready = select.select([sock], [], [], 0.0)[0]
        if not ready:
            break
        line = sock_file.readline()
        if not line:
            break
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("players"):
            latest = rec["players"]
    # Nothing buffered yet - wait for one.
    while not latest and time.time() < deadline:
        line = sock_file.readline()
        if not line:
            break
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("players"):
            latest = rec["players"]
    return latest


# One connection per pad, held open for the whole test.
#
# Reconnecting per command is what made this tool report "nothing happened"
# about pads that were working: the pad server used to release the pad when a
# client disconnected, so each command undid itself the moment the socket
# closed. That release is gone now, but holding the connection is what a real
# client does anyway - the trainer never reconnects mid-episode.
_PADS: dict[int, socket.socket] = {}


def _pad(port: int) -> socket.socket | None:
    if port not in _PADS:
        try:
            _PADS[port] = socket.create_connection(("127.0.0.1", port), 2)
        except OSError:
            return None
    return _PADS[port]


def close_pads() -> None:
    for s in _PADS.values():
        try:
            s.close()
        except OSError:
            pass
    _PADS.clear()


def drive(pad_port: int, steer: float, gas: float = 0.0) -> None:
    """Steering ALONE is not enough to detect a pad.

    A stationary Trackmania car does not turn, and the game reports the
    steering it applied rather than the steering it was handed - so a pad that
    is working perfectly reads `in_steer 0.0` while the car sits still. The
    test has to accelerate, and then watch for movement as well as steering.
    """
    sock = _pad(pad_port)
    if sock is None:
        return
    try:
        sock.sendall(f"act {steer} {gas} 0\n".encode())
        sock.recv(64)
    except OSError:
        _PADS.pop(pad_port, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seats", type=int, default=MAX_SEATS)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-save", action="store_true",
                    help="do not record the result for the panel to read")
    ap.add_argument("--hold", type=float, default=2.5,
                    help="seconds to hold each pad at full lock and throttle")
    a = ap.parse_args()

    broker = seat_ports(0)["broker"]
    try:
        sock = socket.create_connection(("127.0.0.1", broker), 5)
    except OSError as ex:
        print(f"no broker on {broker}: {ex}", file=sys.stderr)
        return 1
    sock.settimeout(4.0)
    f = sock.makefile("rb")

    mapping: dict[int, int | None] = {}
    for pad_index in range(a.seats):
        pad = seat_ports(pad_index)["pad"]
        # Neutral first, so the previous pad's input is not still being read.
        for i in range(a.seats):
            drive(seat_ports(i)["pad"], 0.0, 0.0)
        time.sleep(0.5)
        base = {}
        for p in read_players(sock, f):
            base[p["slot"]] = (float(p.get("in_steer") or 0.0),
                               float(p.get("dist") or 0.0),
                               float(p.get("speed") or 0.0))

        # Full lock AND full throttle: the car has to actually move before the
        # game will report any steering at all.
        # Keep writing for the whole window. One command would be enough now,
        # but a steady stream is what training looks like and it survives a
        # deadman release if one is configured aggressively.
        t_end = time.time() + a.hold
        while time.time() < t_end:
            drive(pad, -1.0, 1.0)
            time.sleep(0.05)
        moved = read_players(sock, f)
        drive(pad, 0.0, 0.0)

        best, best_score = None, 0.0
        for p in moved:
            slot = p.get("slot")
            b = base.get(slot, (0.0, 0.0, 0.0))
            d_steer = abs(float(p.get("in_steer") or 0.0) - b[0])
            d_dist = abs(float(p.get("dist") or 0.0) - b[1])
            d_speed = abs(float(p.get("speed") or 0.0) - b[2])
            # Any of the three is evidence. Distance is the most reliable -
            # it is cumulative, so it survives the sample landing between
            # telemetry frames.
            score = max(d_steer / 0.05, d_dist / 0.25, d_speed / 0.25)
            if score > 1.0 and score > best_score:
                best, best_score = slot, score
        mapping[pad_index] = best
        if not a.json:
            where = (f"seat {best}" if best is not None
                     else "NOTHING - this pad is not bound to a seat")
            print(f"pad {pad} -> {where}")

    for i in range(a.seats):
        drive(seat_ports(i)["pad"], 0.0, 0.0)
    close_pads()
    f.close()
    sock.close()

    if not a.no_save:
        # The panel cannot work this out by watching - a car that has moved
        # was driven by something, not necessarily by us - so the result of
        # actually testing it is written down for the panel to read.
        out = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "logs", "seat_calibration.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"mapping": {str(k): v for k, v in mapping.items()}}, fh)

    if a.json:
        print(json.dumps({str(k): v for k, v in mapping.items()}))
        return 0

    bound = [p for p, s in mapping.items() if s is not None]
    seats_hit = {s for s in mapping.values() if s is not None}
    print()
    if len(bound) < a.seats:
        print(f"{a.seats - len(bound)} pad(s) drive nothing. In the game, "
              f"assign a controller to every seat before training - an "
              f"unbound seat spawns a car that answers to no input, and every "
              f"episode on it ends 'stuck'.")
    elif len(seats_hit) < len(bound):
        print("two pads moved the same seat - the game has not given each "
              "seat its own controller.")
    elif sorted(mapping) == sorted(k for k in mapping if mapping[k] == k):
        print("every pad drives its own seat. Nothing to configure.")
    else:
        print("pads are bound, but not in order. Pass this to the trainer, or "
              "re-assign controllers in-game so pad N drives seat N:")
        print("   " + json.dumps({str(k): v for k, v in mapping.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

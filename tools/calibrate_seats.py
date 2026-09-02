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


def press(pad_port: int, button: str = "b", hold_ms: int = 250) -> None:
    """Send a button press to one pad (`press <button> <hold_ms>`)."""
    sock = _pad(pad_port)
    if sock is None:
        return
    try:
        sock.sendall(f"press {button} {hold_ms}\n".encode())
        sock.recv(64)
    except OSError:
        _PADS.pop(pad_port, None)


def respawn_all(seats: int, sock, f, button: str = "b") -> None:
    """Put every car back on the start line before measuring anything.

    Calibration drives each pad at FULL LOCK AND FULL THROTTLE, which leaves
    the cars scattered - and often wedged against a wall or dropped off the
    track. A wedged car cannot move however correctly its pad is wired, so the
    next run reads it as "this pad is not bound to a seat" and the whole
    mapping is condemned on the strength of a car that was simply stuck.

    Observed exactly that: a run measured clean identity, and the very next run
    reported pads 1 and 3 driving nothing, because the first run's full-lock
    test had beached those two cars.

    Give-up is pressed on every pad rather than restarting the map, for the
    same reason the env avoids RequestRestartMap: it is per-seat and cannot
    disturb anything else.
    """
    for i in range(seats):
        press(seat_ports(i, raw=True)["pad"], button)
    # Let the respawn land, then wait for everything to be stationary.
    time.sleep(1.5)
    end = time.time() + 10.0
    while time.time() < end:
        fast = [p for p in read_players(sock, f)
                if abs(float(p.get("speed") or 0.0)) > 0.5]
        if not fast:
            return
        time.sleep(0.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seats", type=int, default=MAX_SEATS)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-save", action="store_true",
                    help="do not record the result for the panel to read")
    ap.add_argument("--no-respawn", action="store_true",
                    help="do not give-up/respawn the cars first. Only for a "
                         "lobby you know is already on the start line - a "
                         "beached car reads as an unbound pad.")
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

    # Everything back on the start line first - see respawn_all.
    if not a.no_respawn:
        print("resetting all cars to the start line before measuring…")
        respawn_all(a.seats, sock, f)

    mapping: dict[int, int | None] = {}
    for pad_index in range(a.seats):
        pad = seat_ports(pad_index, raw=True)["pad"]
        # Neutral first, so the previous pad's input is not still being read.
        for i in range(a.seats):
            drive(seat_ports(i, raw=True)["pad"], 0.0, 0.0)
        # ...and then WAIT FOR EVERYTHING TO ACTUALLY STOP, rather than
        # sleeping a fixed half second. A car released from full throttle
        # coasts for several seconds, and `dist` is cumulative - so the
        # previous pad's car keeps racking up distance while the next pad is
        # being tested, and can win the comparison below. That is exactly how
        # this reported "two pads moved the same seat" on a lobby that was
        # correctly configured: pad 3 moved car 3 by 1.2m while car 2 was
        # still rolling from the pad 2 test.
        if not a.no_respawn and pad_index:
            respawn_all(a.seats, sock, f)
        settle_end = time.time() + 8.0
        while time.time() < settle_end:
            fast = [p for p in read_players(sock, f)
                    if abs(float(p.get("speed") or 0.0)) > 0.5]
            if not fast:
                break
            time.sleep(0.3)
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
        scores: dict[int, float] = {}
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
            if score > 1.0:
                scores[slot] = max(scores.get(slot, 0.0), score)
        # Winner must DOMINATE, not merely lead. An absolute threshold is the
        # wrong test here: on the start line at full lock a car turns more than
        # it travels, so the real answer can be a 1.2m delta - small, but the
        # only thing moving. Requiring 3x the runner-up keeps that answer while
        # refusing to guess when two cars both moved.
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if ranked:
            best, best_score = ranked[0]
            if len(ranked) > 1 and ranked[1][1] * 3.0 > best_score:
                print(f"  ambiguous: car {best} scored {best_score:.1f} but "
                      f"car {ranked[1][0]} scored {ranked[1][1]:.1f} - "
                      f"letting the cars settle longer would help",
                      file=sys.stderr)
                best = None
        mapping[pad_index] = best
        if not a.json:
            where = (f"seat {best}" if best is not None
                     else "NOTHING - this pad is not bound to a seat")
            print(f"pad {pad} -> {where}")

    for i in range(a.seats):
        drive(seat_ports(i, raw=True)["pad"], 0.0, 0.0)
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
        return 2
    elif len(seats_hit) < len(bound):
        print("two pads moved the same seat - the game has not given each "
              "seat its own controller.")
        return 2
    elif sorted(mapping) == sorted(k for k in mapping if mapping[k] == k):
        print("every pad drives its own seat. Nothing to configure.")
    else:
        print("pads are bound, but not in order. Pass this to the trainer, or "
              "re-assign controllers in-game so pad N drives seat N:")
        print("   " + json.dumps({str(k): v for k, v in mapping.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

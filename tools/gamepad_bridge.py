#!/usr/bin/env python3
"""Pipe a REAL controller into the game through a virtual pad it already sees.

The game enumerates input devices once, at launch, and will not pick up a
controller plugged in afterwards - and rebuilding the splitscreen lobby to add
one is a pain. But the four virtual pad servers were started before the game,
so the game already trusts them. This reads your physical pad with evdev and
streams its state to one of those virtual pads over the pad server's text
protocol (`act <steer> <gas> <brake>` + `press <btn>`).

    tools/gamepad_bridge.py                 # auto-detect pad -> virtual pad :8775 (seat 0)
    tools/gamepad_bridge.py --pad 8765      # a different seat's pad
    tools/gamepad_bridge.py --device /dev/input/event20
    tools/gamepad_bridge.py --list          # show candidate devices and exit

Seat<->pad for the current run is in logs/seat_calibration.json (this run:
8775->seat 0, 8785->seat 1, 8795->seat 2, 8765->seat 3). Drive seat 0 and
record with tools/record_line.py (mode: live) or the panel's line recorder.

Runs on system python (needs evdev); no venv.
"""
import argparse
import socket
import sys
import time

try:
    import evdev
    from evdev import ecodes as e
except ImportError:
    sys.exit("needs python-evdev  (sudo pacman -S python-evdev)")

SKIP = ("tmai", "passthrough", "ydotool", "virtual", "spkr", "speaker")


def candidates():
    out = []
    for path in evdev.list_devices():
        try:
            d = evdev.InputDevice(path)
        except OSError:
            continue
        name = d.name.lower()
        if any(s in name for s in SKIP):
            continue
        caps = d.capabilities()
        absl = {c for c, _ in caps.get(e.EV_ABS, [])}
        keys = set(caps.get(e.EV_KEY, []))
        is_pad = (e.ABS_X in absl and
                  (keys & {e.BTN_SOUTH, e.BTN_A, e.BTN_GAMEPAD, e.BTN_START}))
        if is_pad:
            out.append(d)
    return out


def absinfo(dev, code):
    for c, info in dev.capabilities().get(e.EV_ABS, []):
        if c == code:
            return info
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, default=8775,
                    help="virtual pad server port to drive (default 8775 = seat 0)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", help="/dev/input/eventN of your controller "
                    "(default: auto-detect)")
    ap.add_argument("--deadzone", type=float, default=0.08,
                    help="stick centre deadzone, fraction of full throw")
    ap.add_argument("--rate", type=float, default=60.0,
                    help="how often to push state to the pad, Hz")
    ap.add_argument("--invert-steer", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="print candidate devices and exit")
    a = ap.parse_args()

    if a.list:
        for d in candidates():
            print(f"  {d.path:22} {d.name!r}")
        cs = candidates()
        if not cs:
            print("  (no gamepad-like device found - is it plugged in / in XInput mode?)")
        return 0

    if a.device:
        dev = evdev.InputDevice(a.device)
    else:
        cs = candidates()
        if not cs:
            sys.exit("no gamepad found. Plug it in, set it to XInput mode, or "
                     "pass --device /dev/input/eventN  (see --list).")
        dev = cs[0]
    print(f"controller: {dev.name!r} ({dev.path})", flush=True)

    # Which absolute axes this pad actually has. Triggers vary a lot between
    # controllers: split ABS_Z / ABS_RZ (0..255), or ABS_BRAKE / ABS_GAS, or a
    # single combined ABS_Z. Support the common cases; fall back to shoulder
    # buttons for gas/brake if there are no trigger axes at all.
    axes = {c for c, _ in dev.capabilities().get(e.EV_ABS, [])}
    gas_axis = e.ABS_RZ if e.ABS_RZ in axes else (e.ABS_GAS if e.ABS_GAS in axes else None)
    brk_axis = e.ABS_Z if e.ABS_Z in axes else (e.ABS_BRAKE if e.ABS_BRAKE in axes else None)
    xi = absinfo(dev, e.ABS_X)
    gi = absinfo(dev, gas_axis) if gas_axis else None
    bi = absinfo(dev, brk_axis) if brk_axis else None
    print(f"  steer=ABS_X  gas={evdev.ecodes.ABS.get(gas_axis, gas_axis)}  "
          f"brake={evdev.ecodes.ABS.get(brk_axis, brk_axis)}", flush=True)

    steer = gas = brake = 0.0

    def norm_axis(v, info, lo=-1.0):
        if not info or info.max == info.min:
            return 0.0
        t = (v - info.min) / (info.max - info.min)      # 0..1
        return lo + t * (1.0 - lo)

    sock = socket.create_connection((a.host, a.pad), timeout=3)
    sock.settimeout(0.5)
    print(f"driving virtual pad {a.host}:{a.pad}   (Ctrl-C to stop)", flush=True)

    def send(line):
        try:
            sock.sendall((line + "\n").encode())
            sock.recv(64)                               # drain the "ok"
        except (BrokenPipeError, ConnectionResetError, socket.timeout, OSError):
            pass

    period = 1.0 / a.rate
    last_push = 0.0
    last_sent = None
    try:
        while True:
            r = dev.read_one()
            while r is not None:
                if r.type == e.EV_ABS:
                    if r.code == e.ABS_X:
                        s = norm_axis(r.value, xi)
                        if abs(s) < a.deadzone:
                            s = 0.0
                        steer = -s if a.invert_steer else s
                    elif gas_axis and r.code == gas_axis:
                        gas = max(0.0, norm_axis(r.value, gi, lo=0.0))
                    elif brk_axis and r.code == brk_axis:
                        brake = max(0.0, norm_axis(r.value, bi, lo=0.0))
                    elif r.code in (e.ABS_HAT0X, e.ABS_HAT0Y):
                        pass
                elif r.type == e.EV_KEY and r.value == 1:      # button down
                    if r.code in (e.BTN_SOUTH, e.BTN_A):
                        send("press a 120")                    # validate / skip intro
                    elif r.code in (e.BTN_EAST, e.BTN_B):
                        send("press b 600")                    # give up / respawn
                    elif r.code in (e.BTN_START,):
                        send("press start 120")
                    elif r.code in (e.BTN_TR,) and not gas_axis:
                        gas = 1.0
                    elif r.code in (e.BTN_TL,) and not brk_axis:
                        brake = 1.0
                elif r.type == e.EV_KEY and r.value == 0:      # button up
                    if r.code in (e.BTN_TR,) and not gas_axis:
                        gas = 0.0
                    elif r.code in (e.BTN_TL,) and not brk_axis:
                        brake = 0.0
                r = dev.read_one()

            now = time.time()
            if now - last_push >= period:
                last_push = now
                cur = (round(steer, 3), round(gas, 3), round(brake, 3))
                # push on change, plus a keepalive so the pad's 3s deadman
                # timer never releases while you are holding a steady input
                if cur != last_sent or (now % 1.0) < period:
                    send(f"act {cur[0]} {cur[1]} {cur[2]}")
                    last_sent = cur
            time.sleep(period / 2)
    except KeyboardInterrupt:
        pass
    finally:
        send("reset")
        sock.close()
        print("\nstopped, pad centred.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

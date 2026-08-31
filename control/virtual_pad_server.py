#!/usr/bin/env python3
"""Virtual Xbox 360 gamepad for TM2020 under Proton, driven over a local TCP
socket. Wine passes uinput devices through as normal joysticks, so nothing has
to be injected into the game.

Text protocol, one command per line:
    steer <-1.0..1.0>
    gas <0.0..1.0>          R2 / right trigger
    brake <0.0..1.0>        L2 / left trigger (also reverse when stationary)
    act <steer> <gas> <brake>   all three in one syn() - use this in the loop
    press <button> [ms]     e.g. press y, or: press y 400 to hold it
    reset                   centre steering, release both pedals
    state                   JSON of what we are currently holding
    ping

Quick test:
    printf 'steer 0.8\\n' | nc 127.0.0.1 8765
"""
import argparse
import json
import socket
import threading
import time

from evdev import UInput, AbsInfo, ecodes as e

AXIS_MIN, AXIS_MAX = -32768, 32767
TRIG_MIN, TRIG_MAX = 0, 255

BUTTONS = {
    "a": e.BTN_SOUTH, "b": e.BTN_EAST, "x": e.BTN_NORTH, "y": e.BTN_WEST,
    "lb": e.BTN_TL, "rb": e.BTN_TR,
    "select": e.BTN_SELECT, "start": e.BTN_START,
}

CAPABILITIES = {
    e.EV_KEY: list(BUTTONS.values()) + [e.BTN_THUMBL, e.BTN_THUMBR],
    e.EV_ABS: [
        (e.ABS_X, AbsInfo(0, AXIS_MIN, AXIS_MAX, 0, 0, 0)),
        (e.ABS_Y, AbsInfo(0, AXIS_MIN, AXIS_MAX, 0, 0, 0)),
        (e.ABS_RX, AbsInfo(0, AXIS_MIN, AXIS_MAX, 0, 0, 0)),
        (e.ABS_RY, AbsInfo(0, AXIS_MIN, AXIS_MAX, 0, 0, 0)),
        (e.ABS_Z, AbsInfo(0, TRIG_MIN, TRIG_MAX, 0, 0, 0)),   # L2 -> brake
        (e.ABS_RZ, AbsInfo(0, TRIG_MIN, TRIG_MAX, 0, 0, 0)),  # R2 -> gas
        (e.ABS_HAT0X, AbsInfo(0, -1, 1, 0, 0, 0)),
        (e.ABS_HAT0Y, AbsInfo(0, -1, 1, 0, 0, 0)),
    ],
}


class Pad:
    def __init__(self, name="TMAI Virtual Xbox 360 Controller", product=0x028e):
        self.ui = UInput(CAPABILITIES, name=name,
                         vendor=0x045e, product=product, version=0x110)
        # One lock so a multi-command act() can't be interleaved by another
        # client mid-syn and produce a torn input frame.
        self.lock = threading.Lock()
        # What we last sent, for the panel's controller overlay. This is our
        # side of the wire; the plugin reports what the game actually received,
        # and the two disagreeing is the signal that something is wrong.
        self.state = {"steer": 0.0, "gas": 0.0, "brake": 0.0,
                      "buttons": {}, "t": 0.0}

    def act(self, steer, gas, brake):
        steer = max(-1.0, min(1.0, steer))
        gas = max(0.0, min(1.0, gas))
        brake = max(0.0, min(1.0, brake))
        with self.lock:
            self.ui.write(e.EV_ABS, e.ABS_X, int(steer * AXIS_MAX))
            self.ui.write(e.EV_ABS, e.ABS_RZ, int(gas * TRIG_MAX))
            self.ui.write(e.EV_ABS, e.ABS_Z, int(brake * TRIG_MAX))
            self.ui.syn()
            self.state.update(steer=steer, gas=gas, brake=brake, t=time.time())

    def axis(self, code, value, scale):
        with self.lock:
            self.ui.write(e.EV_ABS, code, int(value * scale))
            self.ui.syn()
            key = {e.ABS_X: "steer", e.ABS_RZ: "gas", e.ABS_Z: "brake"}.get(code)
            if key:
                self.state[key] = value
            self.state["t"] = time.time()

    def press(self, name, hold_ms=80):
        code = BUTTONS[name]
        with self.lock:
            self.ui.write(e.EV_KEY, code, 1)
            self.ui.syn()
            # Held until the timer fires, so the overlay lights the button for
            # exactly as long as the game sees it held.
            self.state["buttons"][name] = time.time() + hold_ms / 1000.0
            self.state["t"] = time.time()
        threading.Timer(hold_ms / 1000.0, self._release, args=(code,)).start()

    def _release(self, code):
        with self.lock:
            self.ui.write(e.EV_KEY, code, 0)
            self.ui.syn()

    # Menu navigation tap. Drives BOTH the D-pad hat AND the left stick to
    # full deflection, because different menus read one or the other, then
    # clears both after hold_ms. dir in up|down|left|right.
    _NAV = {
        # dir: (hat_code, hat_val, stick_code, stick_val)
        "up":    (e.ABS_HAT0Y, -1, e.ABS_Y, AXIS_MIN),
        "down":  (e.ABS_HAT0Y,  1, e.ABS_Y, AXIS_MAX),
        "left":  (e.ABS_HAT0X, -1, e.ABS_X, AXIS_MIN),
        "right": (e.ABS_HAT0X,  1, e.ABS_X, AXIS_MAX),
    }

    def nav(self, direction, hold_ms=120):
        hc, hv, sc, sv = self._NAV[direction]
        with self.lock:
            self.ui.write(e.EV_ABS, hc, hv)
            self.ui.write(e.EV_ABS, sc, sv)
            self.ui.syn()
            self.state["t"] = time.time()
        threading.Timer(hold_ms / 1000.0, self._nav_clear, args=(hc, sc)).start()

    def _nav_clear(self, hc, sc):
        with self.lock:
            self.ui.write(e.EV_ABS, hc, 0)
            self.ui.write(e.EV_ABS, sc, 0)
            self.ui.syn()

    def snapshot(self) -> str:
        now = time.time()
        with self.lock:
            held = [b for b, until in self.state["buttons"].items() if until > now]
            return json.dumps({"steer": round(self.state["steer"], 4),
                               "gas": round(self.state["gas"], 4),
                               "brake": round(self.state["brake"], 4),
                               "buttons": held,
                               "age": round(now - self.state["t"], 3)})

    def reset(self):
        self.act(0.0, 0.0, 0.0)

    def close(self):
        self.reset()
        self.ui.close()


def deadman(pad, timeout: float):
    """Neutralise the pad if nothing has driven it for `timeout` seconds.

    A gamepad latches: the last `act` stays applied until something replaces
    it. So a trainer that dies - crashed, SIGKILLed, or stopped from the panel
    before it could tidy up - leaves the throttle exactly where it was, and
    the car keeps driving indefinitely. Observed: four pads still holding
    gas=1.0 with age=163s, cars still going, and nothing running to blame.
    That reads as "training is still going and I cannot stop it", which is the
    worst possible way for it to look.

    Releasing on client disconnect alone is not enough - a wedged process
    holds its socket open - so this is on the pad's own clock.
    """
    while True:
        time.sleep(0.25)
        with pad.lock:
            idle = time.time() - pad.state["t"]
            live = (abs(pad.state["steer"]) > 1e-3
                    or pad.state["gas"] > 1e-3 or pad.state["brake"] > 1e-3)
        if live and idle > timeout:
            print(f"no input for {idle:.1f}s - releasing the pad", flush=True)
            pad.reset()


def handle(conn, pad):
    with conn:
        buf = b""
        while True:
            data = conn.recv(1024)
            if not data:
                # Do NOT release the pad here.
                #
                # A short-lived connection - send one command, close - is a
                # legitimate and common way to drive this: every CLI tool and
                # test script does it. Releasing on disconnect made each of
                # those commands undo itself a millisecond after it landed,
                # so a pad that was working perfectly measured as dead, and
                # hours went into "the game cannot see the controllers" when
                # the game could see them the whole time.
                #
                # The latch this was meant to guard against is already
                # covered by the deadman timer, which does not care how the
                # client left.
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                parts = line.decode(errors="ignore").strip().split()
                if not parts:
                    continue
                try:
                    cmd = parts[0].lower()
                    if cmd == "act":
                        pad.act(float(parts[1]), float(parts[2]), float(parts[3]))
                    elif cmd == "steer":
                        pad.axis(e.ABS_X, max(-1.0, min(1.0, float(parts[1]))), AXIS_MAX)
                    elif cmd == "gas":
                        pad.axis(e.ABS_RZ, max(0.0, min(1.0, float(parts[1]))), TRIG_MAX)
                    elif cmd == "brake":
                        pad.axis(e.ABS_Z, max(0.0, min(1.0, float(parts[1]))), TRIG_MAX)
                    elif cmd == "press":
                        # optional hold time: some bindings (give up) need the
                        # button held rather than tapped
                        hold = float(parts[2]) if len(parts) > 2 else 80.0
                        pad.press(parts[1].lower(), hold_ms=hold)
                    elif cmd == "nav":
                        # d-pad tap for menu navigation: nav up|down|left|right [ms]
                        hold = float(parts[2]) if len(parts) > 2 else 90.0
                        pad.nav(parts[1].lower(), hold_ms=hold)
                    elif cmd == "reset":
                        pad.reset()
                    elif cmd == "state":
                        # Replies with JSON rather than "ok" - the panel's
                        # controller overlay reads this.
                        conn.sendall(pad.snapshot().encode() + b"\n")
                        continue
                    elif cmd == "ping":
                        pass
                    else:
                        conn.sendall(b"err unknown command %s\n" % cmd.encode())
                        continue
                    conn.sendall(b"ok\n")
                except Exception as ex:
                    conn.sendall(f"err {ex}\n".encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on. Loopback by default. Use "
                         "0.0.0.0 to let a learner on another machine drive "
                         "this pad; that also lets anything on your LAN drive "
                         "it, so only on a network you trust.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--idle-release", type=float, default=3.0, metavar="S",
                    help="release the pad if nothing drives it for S seconds. "
                         "A pad latches, so a trainer that dies mid-throttle "
                         "leaves the car driving forever. Generous next to a "
                         "40Hz control loop, which writes every 25ms.")
    ap.add_argument("--instance", type=int, default=0,
                    help="instance index; names the uinput device so several "
                         "games can each be handed their own pad")
    ap.add_argument("--pool", type=int, default=0, metavar="N",
                    help="create N pads in ONE process instead of one: ports "
                         "PORT, PORT+10, PORT+20, ... and devices named "
                         "'... ' (pad 0), '... #1', '... #2', ...  Start the "
                         "whole pool ONCE before any game launches - a game "
                         "will not enumerate a uinput device that appears "
                         "after it started, and restarting the pool while a "
                         "game runs scrambles device order and makes Steam "
                         "re-grab. Size it for the most cars you will ever run "
                         "at once (2 games x 4 seats = 8, so 12-16).")
    ap.add_argument("--distinct-pid", action="store_true",
                    help="give each pad its own USB product id (0x028e + i) so "
                         "SDL_GAMECONTROLLER_IGNORE_DEVICES_EXCEPT can scope a "
                         "game to its pads. OFF by default: the proven path is "
                         "per-user device ownership (udev), and a non-standard "
                         "product id risks winexinput not recognising the pad "
                         "as XInput. Only turn this on if two games must run as "
                         "the SAME linux user, and test XInput still sees them.")
    args = ap.parse_args()

    base_name = "TMAI Virtual Xbox 360 Controller"

    def make_pad(idx: int) -> "Pad":
        name = base_name if idx == 0 else f"{base_name} #{idx}"
        product = 0x028e + idx if args.distinct_pid else 0x028e
        return Pad(name, product=product)

    def serve_pad(pad: "Pad", host: str, port: int):
        pad.reset()
        threading.Thread(target=deadman, args=(pad, args.idle_release),
                         daemon=True).start()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(5)
        print(f"  pad on {host}:{port} "
              f"(auto-release after {args.idle_release:.0f}s idle)", flush=True)
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=handle, args=(conn, pad), daemon=True).start()

    if args.pool and args.pool > 1:
        # One process, N pads. Port stride 10 matches env/ports.py.
        pads = [make_pad(i) for i in range(args.pool)]
        threads = []
        for i, pad in enumerate(pads):
            t = threading.Thread(target=serve_pad,
                                 args=(pad, args.host, args.port + 10 * i),
                                 daemon=True)
            t.start()
            threads.append(t)
        print(f"pad pool of {args.pool} up: ports "
              f"{args.port}..{args.port + 10 * (args.pool - 1)} step 10"
              + (" (distinct product ids)" if args.distinct_pid else ""),
              flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            for pad in pads:
                pad.close()
        return

    # Single-pad path - unchanged.
    name = base_name
    if args.instance:
        name += f" #{args.instance}"
    pad = Pad(name, product=0x028e + args.instance if args.distinct_pid else 0x028e)
    pad.reset()
    threading.Thread(target=deadman, args=(pad, args.idle_release),
                     daemon=True).start()
    print(f"Virtual pad up. Listening on {args.host}:{args.port} "
          f"(auto-release after {args.idle_release:.0f}s idle)", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(5)
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=handle, args=(conn, pad), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        pad.close()


if __name__ == "__main__":
    main()

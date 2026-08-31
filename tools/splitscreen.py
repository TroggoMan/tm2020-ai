#!/usr/bin/env python3
"""Build a 4-seat splitscreen lobby by driving the virtual pads through the
menu - the only way a free/Starter account can play a custom local map.

WHY it has to be pad input: a free account's Permissions::PlayLocalMap() is
false in solo, so `playmap` fails there; splitscreen/hotseat bypasses that
check. The free-account game runs SAC_GetData (signed), not our plugin, so
there is no in-game API to call - the lobby has to be clicked together, and
"clicked" means fake gamepad input.

The click-path is the SAME every launch (only the map pick changes), so it is
a fixed macro. This runs it.

    tools/splitscreen.py --dry-run          print the steps, touch nothing
    tools/splitscreen.py                    run the whole macro
    tools/splitscreen.py --from 3 --to 7    run only steps 3..7 (for tuning)
    tools/splitscreen.py --seats 2          2-player lobby instead of 4

PREREQS (order matters):
  1. 4 pad servers up on 8765/8775/8785/8795 BEFORE the game started
     (a game will not enumerate a uinput device that appeared after it).
     tools/fleet.py --seats 4   (or virtual_pad_server.py --pool 16)
  2. game running, at the MAIN MENU (not in a map). If a map is loaded,
     send `menu` to the broker first.
  3. broker on 8767 carrying that game's telemetry.

TUNING THE MACRO: the STEPS below are a starting guess. Watch the VNC display,
run slices with --from/--to, and adjust. Each step is:
  (pad, kind, arg, repeat, pause_s)   kind = press | nav | wait | sleep
  press  arg = a|b|x|y|lb|rb|start|select
  nav    arg = up|down|left|right
  wait   arg = "ui=1" | "car" | "map" | "menu"   (poll telemetry until true)
  sleep  arg = seconds (float), repeat/pause ignored
Only pad 0 navigates; pads 1..N-1 just press A to JOIN their seat.
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.ports import instance_ports, seat_ports   # noqa: E402


HOST = "127.0.0.1"   # set by --host; the machine the game+broker+pads run on


def broker_addr(instance: int) -> tuple[str, int]:
    return (HOST, instance_ports(instance)["broker"])


def pad_addr(seat: int, instance: int) -> tuple[str, int]:
    return (HOST, seat_ports(seat, game=instance)["pad"])


# ---------------------------------------------------------------------------
# THE MACRO - transcribed from the hand click-path. Same every launch except
# the map-selection block, which is the only part that changes with the map.
#
# "Options" on a PS5 pad == the START button on an Xbox pad (BTN_START).
# The session-timer field needs real keystrokes (a pad cannot type digits),
# so those steps use xdotool against the game's X display.
#
# `right_to_seats` = how many Right taps move the player toggle to `seats`.
# `map_right` = Right taps in "My Maps" to land on the target track.
# ---------------------------------------------------------------------------
def build_steps(seats: int, map_name: str, timer_s: int,
                right_to_seats: int, map_right: int,
                skip_timer: bool = False) -> list[tuple]:
    """VERIFIED keyboard path, walked screen-by-screen on 2026-08-28 against a
    game on Xvfb :99.  KEYS: nav = arrow keys, press a = Return/Enter,
    press start/y/end -> the "End" key (opens the Mode-Settings dialog),
    press esc = Escape.  On :99 (real Xvfb) xdotool drives all of this; the
    game does NOT see the virtual pads under a direct-Proton launch, so
    --nav-pad will NOT work here - use keyboard.

    NOTE: 4-player splitscreen still refuses to LAUNCH ("connect 3 more
    controllers") until the game can see 4 input devices. The menu path
    below is complete; the device problem is separate (launch via Steam).
    """
    timer_block = [] if skip_timer else [
        (0, "key",   "End",       1, 1.0),     # open MODE SETTINGS
        (0, "nav",   "up",        1, 0.5),     # -> TIME LIMIT field
        (0, "press", "a",         1, 0.6),     # enter edit mode
        (0, "key",   "ctrl+a",    1, 0.2),
        (0, "key",   "BackSpace", 8, 0.05),    # clear "300"
        (0, "type",  str(timer_s), 1, 0.3),
        (0, "press", "a",         1, 0.6),     # confirm field
        (0, "nav",   "down",      9, 0.25),    # -> APPLY (past the checkbox)
        (0, "press", "a",         1, 1.0),     # APPLY -> back to setup
    ]
    return [
        (0, "wait",  "menu", 1, 0.0),
        (0, "sleep", 1.2, 0, 0.0),

        (0, "press", "a",     1, 1.5),         # PLAY -> LOCAL tab, HOTSEAT sel.
        (0, "nav",   "right", 1, 0.5),         # -> SPLITSCREEN
        (0, "press", "a",     1, 1.5),         # -> splitscreen setup

        (0, "nav",   "right", right_to_seats, 0.4),  # NUMBER OF PLAYERS 2 -> 4
        *timer_block,

        # add the map
        (0, "nav",   "down",  3, 0.3),         # NUMBER OF PLAYERS -> the [+] box
        (0, "press", "a",     1, 1.5),         # -> TRACK BROWSER (MY LOCAL TRACKS)
        (0, "press", "a",     1, 1.5),         # -> AUTOSAVES / MY MAPS
        (0, "nav",   "right", 1, 0.4),         # -> MY MAPS
        (0, "press", "a",     1, 1.5),         # -> the map grid (first tile sel.)
        (0, "nav",   "right", map_right, 0.4), # -> target track
        (0, "press", "a",     1, 1.0),         # tick it ("ADD 1 TRACK" appears)
        (0, "nav",   "down",  1, 0.4),         # -> ADD 1 TRACK
        (0, "press", "a",     1, 1.5),         # add -> back to setup

        # launch
        (0, "nav",   "down",  1, 0.3),
        (0, "nav",   "right", 1, 0.3),         # -> PLAY
        (0, "press", "a",     1, 2.0),
        (0, "wait",  "car",   1, 0.0),
    ]


# ---------------------------------------------------------------------------
class Link:
    def __init__(self, addr):
        self.s = socket.create_connection(addr, timeout=5)
        self.s.settimeout(1.0)
        self.buf = b""

    def latest(self) -> dict | None:
        end = time.time() + 1.5
        rec = None
        while time.time() < end:
            try:
                d = self.s.recv(65536)
            except socket.timeout:
                break
            if not d:
                break
            self.buf += d
            while b"\n" in self.buf:
                ln, self.buf = self.buf.split(b"\n", 1)
                if not ln.strip():
                    continue
                try:
                    o = json.loads(ln)
                except ValueError:
                    continue
                if "car" in o:
                    rec = o
        return rec

    def send(self, line: str):
        self.s.sendall((line + "\n").encode())


def cond_met(rec: dict | None, cond: str) -> bool:
    if rec is None:
        return False
    if cond == "menu":
        return not rec.get("car") and not rec.get("in_race")
    if cond == "car":
        return bool(rec.get("car"))
    if cond == "map":
        return bool(rec.get("map"))
    if cond.startswith("ui="):
        return str(rec.get("ui")) == cond[3:]
    return False


def pad_cmd(seat: int, line: str, instance: int = 0):
    try:
        with socket.create_connection(pad_addr(seat, instance), timeout=2) as s:
            s.sendall((line + "\n").encode())
            s.recv(64)
    except OSError as e:
        print(f"  ! pad {seat} ({pad_addr(seat, instance)[1]}): {e}")


# Keyboard/mouse injection. ydotool goes through /dev/uinput at the kernel,
# so it reaches an XWayland game on a Wayland desktop where xdotool cannot.
# It types into whatever window has focus - so park the mouse and make sure
# the game window is focused first.
_YDO = shutil.which("ydotool")
_KEY = {  # evdev keycodes for the few keys the macro needs (lowercased lookup)
    "backspace": 14, "delete": 111, "enter": 28, "return": 28, "ctrl": 29,
    "a": 30, "end": 107, "home": 102, "escape": 1, "esc": 1,
    "0": 11, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9,
    "9": 10,
    "up": 103, "down": 108, "left": 105, "right": 106,
}


def type_text(text: str, display: str) -> bool:
    if _YDO:
        subprocess.run([_YDO, "type", "--key-delay", "40", text],
                       capture_output=True, timeout=8)
        return True
    if shutil.which("xdotool"):
        subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "60",
                        text], env={"DISPLAY": display},
                       capture_output=True, timeout=8)
        return True
    print("  ! no ydotool or xdotool - cannot type the timer value")
    return False


def press_key(name: str, display: str) -> bool:
    """name = 'ctrl+a' | 'BackSpace' | 'End' | 'Up' ... (xdotool spelling ok)"""
    if _YDO:
        parts = name.lower().split("+")
        codes = [_KEY[p] for p in parts if p in _KEY]
        seq = [f"{c}:1" for c in codes] + [f"{c}:0" for c in reversed(codes)]
        if seq:
            subprocess.run([_YDO, "key", *seq], capture_output=True, timeout=5)
        return True
    if shutil.which("xdotool"):
        subprocess.run(["xdotool", "key", "--clearmodifiers", name],
                       env={"DISPLAY": display}, capture_output=True, timeout=5)
        return True
    return False


def focus_game(display: str):
    """Best-effort: pull focus onto the game window so injected keys land in
    it. Works only when the display is a plain X server we can reach."""
    if not shutil.which("xdotool"):
        return
    env = {"DISPLAY": display}
    wid = subprocess.run(["xdotool", "search", "--name", "Trackmania"],
                         capture_output=True, text=True, env=env,
                         timeout=3).stdout.split()
    if wid:
        subprocess.run(["xdotool", "windowactivate", "--sync", wid[0]],
                       env=env, timeout=3, capture_output=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seats", type=int, default=4)
    ap.add_argument("--map", default="My Maps/Wide left, mid right - road.Map.Gbx")
    ap.add_argument("--timer", type=int, default=1_000_000,
                    help="session timer seconds (default 300 in-game)")
    ap.add_argument("--right-to-seats", type=int, default=1,
                    help="Right taps to move the player toggle from 2 to --seats")
    ap.add_argument("--map-right", type=int, default=1,
                    help="Right taps in 'My Maps' to reach the target track")
    ap.add_argument("--display", default=":99",
                    help="X display of the game, for typing the timer value")
    ap.add_argument("--nav-pad", action="store_true",
                    help="navigate with the pad D-pad instead of the keyboard "
                         "(needs a pad server with the `nav` command)")
    ap.add_argument("--skip-timer", action="store_true",
                    help="leave the session timer at its 300s default. Use "
                         "this when the game is NOT on a real Xvfb, so the "
                         "digits can't be typed. Costs a lobby rollover every "
                         "5 min (training survives it).")
    ap.add_argument("--host", default="127.0.0.1",
                    help="machine the game/broker/pads run on (broker+pads must "
                         "be bound 0.0.0.0 on that box)")
    ap.add_argument("--instance", type=int, default=0,
                    help="game instance index - picks its broker and seat pads")
    ap.add_argument("--instances", type=int, default=0, metavar="N",
                    help="run the macro against instances 0..N-1 in turn")
    ap.add_argument("--from", dest="lo", type=int, default=0)
    ap.add_argument("--to", dest="hi", type=int, default=10_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--step-pause", type=float, default=0.4,
                    help="baseline pause after every step")
    a = ap.parse_args()

    global HOST
    HOST = a.host

    steps = build_steps(a.seats, a.map, a.timer, a.right_to_seats, a.map_right,
                        a.skip_timer)

    if a.dry_run:
        for i, (pad, kind, arg, rep, pause) in enumerate(steps):
            mark = "  " if a.lo <= i <= a.hi else "· "
            print(f"{mark}{i:2d}  pad{pad}  {kind:6s} {str(arg):10s} "
                  f"x{rep} pause={pause}")
        return 0

    targets = list(range(a.instances)) if a.instances else [a.instance]
    rc = 0
    for inst in targets:
        print(f"\n===== instance {inst} "
              f"(broker {broker_addr(inst)[1]}) =====")
        rc |= run_one(inst, steps, a)
    return rc


def run_one(inst: int, steps: list[tuple], a) -> int:
    try:
        link = Link(broker_addr(inst))
    except OSError as e:
        print(f"  broker {broker_addr(inst)} unreachable: {e} - "
              f"is instance {inst}'s game+plugin up?", file=sys.stderr)
        return 1

    # Park the mouse in the far corner: a cursor sitting over the menu
    # hover-selects items and fights the navigation. Then pull focus onto the
    # game so injected keystrokes land in it.
    if _YDO:
        subprocess.run([_YDO, "mousemove", "-a", "10000", "10000"],
                       capture_output=True, timeout=3)
    elif shutil.which("xdotool"):
        subprocess.run(["xdotool", "mousemove", "10000", "10000"],
                       env={"DISPLAY": a.display}, capture_output=True, timeout=3)
    focus_game(a.display)

    kmap = {"up": "Up", "down": "Down", "left": "Left", "right": "Right"}
    for i, (pad, kind, arg, rep, pause) in enumerate(steps):
        if not (a.lo <= i <= a.hi):
            continue
        print(f"[{i:2d}] pad{pad} {kind} {arg} x{rep}")
        if kind == "wait":
            deadline = time.time() + 30
            while time.time() < deadline:
                if cond_met(link.latest(), str(arg)):
                    print(f"     condition '{arg}' met")
                    break
                time.sleep(0.5)
            else:
                print(f"     !! timed out waiting for '{arg}' - continuing anyway")
        elif kind == "sleep":
            time.sleep(float(arg))
        elif kind == "type":
            focus_game(a.display)
            type_text(str(arg), a.display)
        elif kind == "key":
            press_key(str(arg), a.display)
        elif kind == "nav":
            for _ in range(max(1, rep)):
                if a.nav_pad:
                    pad_cmd(pad, f"nav {arg}", inst)
                else:
                    press_key(kmap[str(arg)].lower(), a.display)
                time.sleep(pause or a.step_pause)
        else:  # press - always the pad
            for _ in range(max(1, rep)):
                pad_cmd(pad, f"press {arg}", inst)
                time.sleep(pause or a.step_pause)
        time.sleep(a.step_pause)

    print("     done - check the display.")
    return 0


if __name__ == "__main__":
    time.sleep(5)
    sys.exit(main())

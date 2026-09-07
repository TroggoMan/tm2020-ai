#!/usr/bin/env python3
"""Serve plugin-format telemetry read straight out of the game's RAM.

WHY THIS EXISTS
    TMAITelemetry is unsigned -> Openplanet Developer mode -> a PAID account
    (FLEET_V2.md). That is the whole reason the free fleet cannot run our
    plugin. `/proc/<pid>/mem` has no such constraint, so this adapter gives a
    free School-mode instance real vehicle state with no plugin at all.

    Same contract as telemetry/sac_getdata_adapter.py: it re-serves the exact
    newline-JSON our plugin speaks, so telemetry/broker.py and env/tm_env.py
    need no transport change.

        python3 telemetry/ram_adapter.py --serve-port 8776 --map-uid <uid>
        python3 telemetry/broker.py --upstream-port 8776 --port 8777

WHAT IS REAL AND WHAT IS A ZERO
    Real, checked against the plugin frame-by-frame (see env/ram_state.py for
    the per-field correlations): pos, vel, left/up/dir, speed, side_speed,
    in_steer, in_gas, in_brake, gear, rpm, damper, per-wheel ground contact,
    race_time, finished.

    Honest zeros, exactly as FLEET_V2 argues for the SAC_GetData adapter: slip,
    icing, dirt, wear, wetness, turbo/reactor/cruise/airbrake. Every one of
    those is genuinely 0/inactive on a plain road map.

    `cp` and `lap` always read 0 - and so does the real plugin, on every map,
    because the game's RaceWaypointTimes reads 0 in Time Attack. env/tm_env.py
    never reads them: it counts gate crossings from `pos` against the surveyed
    map landmarks. See "Checkpoints: count them yourself" in README.md.
"""
import argparse, json, os, re, socket, subprocess, sys, threading, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "env"))
import ram_state as rs                                    # noqa: E402
from sac_getdata_adapter import Server                     # noqa: E402


class Pad:
    """The instance's uinput gamepad, over the pad server's line protocol."""

    def __init__(self, port):
        self.port = port

    def _send(self, line):
        try:
            with socket.create_connection(("127.0.0.1", self.port), 2) as s:
                s.sendall(line.encode() + b"\n")
                s.recv(64)
            return True
        except OSError:
            return False

    def act(self, steer, gas, brake):
        return self._send(f"act {steer} {gas} {brake}")

    def press(self, button, hold_ms=200):
        return self._send(f"press {button} {hold_ms}")


class RamSource:
    def __init__(self, pid, pad=None, map_uid="", hz=100.0, display=None):
        self.pid, self.pad, self.map_uid = pid, pad, map_uid
        self.period, self.display = 1.0 / hz, display
        self.vs = None
        self.addr = None

    # -- locating ----------------------------------------------------------
    def locate(self, timeout=240.0):
        """Find the vehicle struct. Needs the car MOVING, so if we own a pad we
        hold the gas ourselves rather than requiring a human to drive."""
        stop = threading.Event()
        if self.pad:
            def hold():
                i = 0
                while not stop.is_set():
                    if i > 300:                 # ~15 s: give up and go again,
                        i = 0                   # so a wall is not a dead end
                        self.pad.press("b", 200); time.sleep(1.0); continue
                    self.pad.act(0.0, 1.0, 0.0); i += 1; time.sleep(0.05)
            self.pad.press("b", 200); time.sleep(1.6)
            threading.Thread(target=hold, daemon=True).start()
            time.sleep(1.5)
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                hits = rs.locate_structural(self.pid)
                if len(hits) == 1:
                    self.addr = hits[0]
                    self.vs = rs.VehicleState(self.pid, self.addr)
                    print(f"[ram] vehicle struct at 0x{self.addr:x}", flush=True)
                    return self.addr
                print(f"[ram] locate returned {len(hits)} candidates, retrying",
                      flush=True)
        finally:
            stop.set(); time.sleep(0.2)
            if self.pad:
                self.pad.act(0.0, 0.0, 0.0)
        raise RuntimeError("could not locate the vehicle struct")

    # -- run control -------------------------------------------------------
    def restart(self):
        """Put the car back on the start line.

        Give-up is a pad button and works mid-race. The FINISH screen is the
        exception: no gamepad button dismisses it (measured - b/y/a/start/select
        all do nothing), but Enter over XTEST does, because the game lives on a
        plain Xvfb. That keystroke is the only reason this works without the
        plugin's `restart` command.
        """
        try:
            finished = self.vs.read()["finished"] if self.vs else False
        except OSError:
            finished = False
        if finished and self.display:
            subprocess.run(["xdotool", "key", "Return"],
                           env={**os.environ, "DISPLAY": self.display},
                           capture_output=True)
            time.sleep(1.0)
        if self.pad:
            self.pad.press("b", 200)
        return {"ok": True, "cmd": "restart", "src": "ram_adapter"}

    # -- the frames --------------------------------------------------------
    def record(self, r):
        return {
            "car": True,
            "map": self.map_uid,
            "pos": list(r["pos"]),
            "vel": list(r["vel"]),
            "dir": list(r["dir"]),
            "up": list(r["up"]),
            "left": list(r["left"]),
            "speed": r["speed"],
            "side_speed": r["side_speed"],
            "gear": r["gear"],
            "rpm": r["rpm"],
            "in_steer": r["in_steer"],
            "in_gas": r["in_gas"],
            "in_brake": 1.0 if r["in_brake"] else 0.0,
            "race_time": r["race_time"],
            "finished": r["finished"],
            "ground": r["ground"],
            "damper": list(r["damper"]),
            # Per-wheel ground contact is real; the MATERIAL is not readable
            # yet, so report asphalt for a wheel that is down. Correct on the
            # plain road maps this adapter is for, wrong on anything with grass
            # or dirt in the racing line.
            "mat": [16 if c else 80 for c in r["contact"]],
            # --- honest zeros: genuinely inactive on a plain road map --------
            "slip": [0.0, 0.0, 0.0, 0.0],
            "icing": [0.0, 0.0, 0.0, 0.0],
            "dirt": [0.0, 0.0, 0.0, 0.0],
            "wear": [0.0, 0.0, 0.0, 0.0],
            "adherence": 1.0,
            "wetness": 0.0,
            "turbo": False, "turbo_time": 0.0, "turbo_lvl": 0,
            "reactor_type": 0, "reactor_lvl": 0, "reactor_timer": 0.0,
            "cruise": 0, "sim_coef": 1.0, "air_brake": 0.0,
            # --- NOT available from RAM: see the module docstring ------------
            "cp": 0, "lap": 0,
            "src": "ram",
        }

    def records(self):
        if self.vs is None:
            self.locate()
        misses = 0
        while True:
            try:
                r = self.vs.read()
                misses = 0
            except (OSError, ValueError):
                misses += 1
                if misses > 50:                 # the game restarted, or the
                    print("[ram] lost the struct, re-locating", flush=True)
                    self.vs = None; self.locate(); misses = 0
                time.sleep(0.05)
                continue
            yield self.record(r)
            time.sleep(self.period)


class RamServer(Server):
    """`Server`, but with the one command the free fleet actually needs."""

    def _handle_client(self, conn):
        with self.lock:
            self.clients.append(conn)
        buf = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    cmd = raw.strip().decode(errors="replace")
                    if not cmd:
                        continue
                    head = cmd.split()[0]
                    if head == "restart":
                        reply = self.source.restart()
                    else:
                        # landmarks/dumpmap are survey features of the plugin.
                        # SURVEY runs on the privileged instance anyway.
                        reply = {"ok": False, "cmd": head,
                                 "err": "not supported via the RAM adapter"}
                    try:
                        conn.sendall((json.dumps(reply) + "\n").encode())
                    except OSError:
                        return
        except OSError:
            return
        finally:
            with self.lock:
                if conn in self.clients:
                    self.clients.remove(conn)
            try:
                conn.close()
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, default=0,
                    help="game pid; default: auto-detect the biggest Trackmania")
    ap.add_argument("--user", default=None,
                    help="pick the game belonging to this Linux user. With a "
                         "fleet there is more than one game on the box and "
                         "auto-detect is a coin toss, so name the instance.")
    ap.add_argument("--pad-port", type=int, default=8765,
                    help="pad server for this instance - used to drive the car "
                         "while locating, and to give up on restart")
    ap.add_argument("--display", default=":99",
                    help="X display this instance is on; Enter is sent here to "
                         "dismiss the finish screen")
    ap.add_argument("--map-uid", default="",
                    help="map uid to stamp on every frame - RAM does not carry "
                         "it and downstream needs it")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--serve-port", type=int, default=8776)
    ap.add_argument("--hz", type=float, default=100.0)
    a = ap.parse_args()

    pid = a.pid or rs.find_game_pid(user=a.user)
    if not pid:
        sys.exit("no running Trackmania process found"
                 + (f" for user {a.user}" if a.user else ""))
    print(f"[ram] game pid {pid}", flush=True)
    src = RamSource(pid, Pad(a.pad_port) if a.pad_port else None,
                    a.map_uid, a.hz, a.display)
    RamServer(src, a.bind, a.serve_port).serve()


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# KNOWN GAPS - read before pointing this at a real campaign track
#
# cp / lap always 0, and that is not a gap: the plugin emits 0 too, always,
#   because RaceWaypointTimes reads 0 in Time Attack. tm_env.py counts gates
#   from position instead. Do not go hunting for it in RAM - 672 samples over
#   several laps with checkpoints genuinely crossed never moved it.
#
# mat is inferred from ground contact, not read. Surface one-hot will say
#   "asphalt" over grass.
#
# ground_dist is absent: the test maps are walled, so the car never leaves the
#   ground and there was no signal to correlate against. `contact` covers the
#   airborne case.
#
# playmap is not available - loading a map is a plugin command. A free instance
#   still needs its map chosen by hand or by menu navigation. This is the same
#   FLEET_V2 gap the SAC_GetData route has.
# ---------------------------------------------------------------------------

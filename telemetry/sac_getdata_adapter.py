#!/usr/bin/env python3
"""Adapter: SAC_GetData ("TrackmaniaRL Connect", siteid 421) -> our plugin's wire format.

Why this exists
---------------
Our own telemetry plugin is unsigned, so it only loads in Openplanet
*Developer* mode, which needs a *paid* account. A free/Starter account gets
*School* mode, which loads only signed plugins. SAC_GetData IS signed and
confirmed loading in School mode, and it serves a fixed 33-float binary stream
on 127.0.0.1:9000 plus a JSONL readiness channel on 9001.

This process reads that binary stream and re-serves it as **exactly the
newline-delimited JSON our plugin speaks**, on the same port the broker
expects upstream (8766 by default). So `telemetry/broker.py` and
`env/tm_env.py` need no changes at all - to everything downstream this looks
like our plugin, just with fewer fields populated.

What survives, what does not
----------------------------
Direct from the stream: pos, vel, dir, up, speed, rpm, gear, per-wheel slip,
per-wheel ground material, adherence. Derived here: `left` = cross(up, dir),
`side_speed` = dot(vel, left), `ground` ~= (flying_duration == 0).

NOT in the stream, emitted at their "nothing" defaults so `_observe` reads
them as inactive rather than as a booster-underfoot: icing/dirt/wetness = 0,
turbo/reactor/cruise/air_brake absent, sim_coef = 1.0. On a plain road/asphalt
test map every one of those is genuinely zero for the whole run, so the
observation is complete there. Maps with real ice/water/boost need the
reconstruction layer (see FLEET_V2.md) - not this file's job.

Commands (the broker forwards anything a client writes upstream):
  landmarks  -> {"ok": false}   there are none; explore stage cannot run here
  anything else -> {"ok": false}
Give-up / reset is a virtual-pad button press and never comes through here.

Usage
-----
    python3 telemetry/sac_getdata_adapter.py                 # 9000/9001 -> serve 8766
    python3 telemetry/sac_getdata_adapter.py --serve-port 8776
    python3 telemetry/sac_getdata_adapter.py --mock          # no game needed

Then point the broker at it:  telemetry/broker.py --upstream-port 8766
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import struct
import sys
import threading
import time

# ---------------------------------------------------------------------------
# The 33-float layout, in order, straight from TrackmaniaRL_Connect.as. Names
# are ours; the comment is what the plugin appends at that slot.
# ---------------------------------------------------------------------------
FRAME = "<33f"
FRAME_BYTES = struct.calcsize(FRAME)          # 132
FIELDS = [
    "cp_count",        # NumberOfCheckpointsPassed (0 until driving)
    "lap",             # CurrentLapNumber
    "finished",        # bool
    "race_time",       # ms since start, 0 before the gun
    "px", "py", "pz",  # api.Position
    "vx", "vy", "vz",  # api.Velocity
    "dx", "dy", "dz",  # vis.Dir
    "ux", "uy", "uz",  # vis.Up
    "speed",           # api.Speed  (m/s, signed)
    "rpm",             # api.EngineRpm
    "gear",            # api.EngineCurGear
    "slip_fl", "slip_fr", "slip_rl", "slip_rr",     # vis.**SlipCoef
    "mat_fl", "mat_fr", "mat_rl", "mat_rr",         # vis.**GroundContactMaterial (enum int)
    "skid_count",      # api.WheelsSkiddingCount
    "flying",          # api.FlyingDuration (ms airborne; 0 = on ground)
    "adherence",       # api.AdherenceCoef
    "in_steer",        # api.InputSteer   [-1, 1]
    "in_gas",          # api.InputGasPedal [0, 1]
    "in_braking",      # bool
]
assert len(FIELDS) == 33


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 1e-9 else v


def frame_to_record(vals: dict) -> dict:
    """One unpacked 33-float frame -> the JSON record our plugin emits.

    The `"car"` key is the marker `TelemReader`/`Broker` use to tell a
    telemetry line from a command reply - it must be present.
    """
    pos = [vals["px"], vals["py"], vals["pz"]]
    vel = [vals["vx"], vals["vy"], vals["vz"]]
    dir_ = _norm([vals["dx"], vals["dy"], vals["dz"]])
    up = _norm([vals["ux"], vals["uy"], vals["uz"]])
    # Our plugin's `left` is cross(up, dir): a right-handed frame with dir
    # forward and up up. `_observe` uses it for the lookahead car-frame and
    # for side speed, so it has to match the plugin's handedness.
    left = _norm(_cross(up, dir_))
    side_speed = sum(a * b for a, b in zip(vel, left))
    mats = [int(round(vals["mat_fl"])), int(round(vals["mat_fr"])),
            int(round(vals["mat_rl"])), int(round(vals["mat_rr"]))]

    return {
        "car": True,
        # uid is filled in by the caller from the :9001 session channel; keep
        # the key present so downstream never KeyErrors on a heartbeat.
        "map": vals.get("_map", ""),
        "pos": pos,
        "vel": vel,
        "dir": dir_,
        "up": up,
        "left": left,
        "speed": vals["speed"],
        "slip": [vals["slip_fl"], vals["slip_fr"], vals["slip_rl"], vals["slip_rr"]],
        "mat": mats,
        "adherence": vals["adherence"],
        # FlyingDuration is milliseconds off the ground; 0 is the only value
        # that reliably means "wheels down". A short hop reads as airborne,
        # which is the safe direction to be wrong in.
        "ground": vals["flying"] == 0,
        "gear": int(round(vals["gear"])),
        "rpm": vals["rpm"],
        "side_speed": side_speed,
        # Inputs the game actually applied, for the fidelity check.
        "in_steer": vals["in_steer"],
        "in_gas": vals["in_gas"],
        "in_brake": 1.0 if vals["in_braking"] >= 0.5 else 0.0,
        # Race progress, for episode / reset logic.
        "cp": int(round(vals["cp_count"])),
        "lap": int(round(vals["lap"])),
        "finished": vals["finished"] >= 0.5,
        "race_time": vals["race_time"],
        "skid_count": int(round(vals["skid_count"])),
        # --- not in the stream: emitted as "inactive", correct on plain maps,
        #     reconstructed elsewhere for maps that actually have these -------
        "icing": [0.0, 0.0, 0.0, 0.0],
        "dirt": [0.0, 0.0, 0.0, 0.0],
        "wetness": 0.0,
        "turbo": False,
        "turbo_time": 0.0,
        "turbo_lvl": 0,
        "reactor_type": 0,
        "reactor_lvl": 0,
        "reactor_timer": 0.0,
        "cruise": 0,
        "sim_coef": 1.0,
        "air_brake": 0.0,
        "src": "sac_getdata",
    }


# ---------------------------------------------------------------------------
# Session channel (:9001) - one-shot query for the loaded map uid.
# ---------------------------------------------------------------------------
def query_map_uid(host: str, port: int, timeout: float = 2.0) -> str:
    req = json.dumps({"protocol_version": "2",
                      "command": "verify_loaded_map"}) + "\n"
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall(req.encode())
            s.settimeout(timeout)
            data = s.recv(4096).decode(errors="replace").strip()
        for line in data.splitlines():
            obj = json.loads(line)
            if obj.get("status") == "ok" and obj.get("map_uid"):
                return obj["map_uid"]
    except (OSError, ValueError):
        pass
    return ""


# ---------------------------------------------------------------------------
# Upstream reader: binary 33-float stream on :9000
# ---------------------------------------------------------------------------
class SacSource:
    def __init__(self, host="127.0.0.1", data_port=9000, session_port=9001):
        self.host = host
        self.data_port = data_port
        self.session_port = session_port
        self.map_uid = ""
        self._last_uid_check = 0.0

    def _refresh_uid(self):
        now = time.time()
        if now - self._last_uid_check < 3.0:
            return
        self._last_uid_check = now
        uid = query_map_uid(self.host, self.session_port)
        if uid:
            self.map_uid = uid

    def records(self):
        """Yield JSON records forever, reconnecting to :9000 as needed."""
        buf = b""
        sock = None
        while True:
            if sock is None:
                try:
                    sock = socket.create_connection((self.host, self.data_port),
                                                    timeout=5.0)
                    sock.settimeout(1.0)
                    buf = b""
                    print(f"[adapter] connected to SAC_GetData "
                          f"{self.host}:{self.data_port}", flush=True)
                except OSError:
                    time.sleep(2.0)
                    continue
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                data = b""
            if not data:
                print("[adapter] SAC_GetData stream lost, reconnecting", flush=True)
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
                continue

            buf += data
            while len(buf) >= FRAME_BYTES:
                chunk, buf = buf[:FRAME_BYTES], buf[FRAME_BYTES:]
                vals = dict(zip(FIELDS, struct.unpack(FRAME, chunk)))
                self._refresh_uid()
                vals["_map"] = self.map_uid
                yield frame_to_record(vals)


class MockSource:
    """A car doing lazy circles on plain asphalt. Enough to prove the wire
    format and the broker/env path without a game or an account."""
    def __init__(self, hz=100.0):
        self.dt = 1.0 / hz

    def records(self):
        t = 0.0
        while True:
            r = 30.0
            ang = t * 0.4
            pos = [r * math.cos(ang), 0.0, r * math.sin(ang)]
            speed = r * 0.4
            vel = [-speed * math.sin(ang), 0.0, speed * math.cos(ang)]
            dir_ = _norm(vel)
            up = [0.0, 1.0, 0.0]
            left = _norm(_cross(up, dir_))
            vals = {
                "cp_count": int(t // 10) % 3, "lap": 0, "finished": 0.0,
                "race_time": t * 1000.0,
                "px": pos[0], "py": pos[1], "pz": pos[2],
                "vx": vel[0], "vy": vel[1], "vz": vel[2],
                "dx": dir_[0], "dy": dir_[1], "dz": dir_[2],
                "ux": up[0], "uy": up[1], "uz": up[2],
                "speed": speed, "rpm": 6000.0 + 1000.0 * math.sin(t),
                "gear": 3,
                "slip_fl": 0.02, "slip_fr": 0.02, "slip_rl": 0.05, "slip_rr": 0.05,
                "mat_fl": 16, "mat_fr": 16, "mat_rl": 16, "mat_rr": 16,   # Asphalt
                "skid_count": 0, "flying": 0, "adherence": 1.0,
                "in_steer": 0.3, "in_gas": 1.0, "in_braking": 0.0,
                "_map": "MOCKSACGETDATA0000000000000",
            }
            yield frame_to_record(vals)
            t += self.dt
            time.sleep(self.dt)


# ---------------------------------------------------------------------------
# Downstream server: mimic the plugin's one-client newline-JSON socket.
# ---------------------------------------------------------------------------
class Server:
    def __init__(self, source, bind="127.0.0.1", port=8766):
        self.source = source
        self.addr = (bind, port)
        self.clients: list[socket.socket] = []
        self.lock = threading.Lock()
        self.frames = 0

    def _fanout(self, line: bytes):
        with self.lock:
            dead = []
            for c in self.clients:
                try:
                    c.sendall(line)
                except OSError:
                    dead.append(c)
            for c in dead:
                self.clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    def _handle_client(self, conn: socket.socket):
        with self.lock:
            self.clients.append(conn)
        buf = b""
        try:
            conn.settimeout(None)
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
                    # Explore stage asks for `landmarks`; there are none here.
                    # Everything else (playmap, dumpmap, perms) is a plugin
                    # feature SAC_GetData does not have.
                    reply = {"ok": False, "cmd": cmd.split()[0],
                             "err": "not supported via SAC_GetData adapter"}
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

    def _pump(self):
        for rec in self.source.records():
            self.frames += 1
            self._fanout((json.dumps(rec, separators=(",", ":")) + "\n").encode())

    def serve(self):
        threading.Thread(target=self._pump, daemon=True).start()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(self.addr)
        srv.listen(8)
        print(f"[adapter] serving plugin-format telemetry on "
              f"{self.addr[0]}:{self.addr[1]}", flush=True)
        last = time.time()
        while True:
            srv.settimeout(5.0)
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                now = time.time()
                print(f"[adapter] {self.frames} frames, "
                      f"{len(self.clients)} client(s)", flush=True)
                last = now
                continue
            threading.Thread(target=self._handle_client, args=(conn,),
                             daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sac-host", default="127.0.0.1")
    ap.add_argument("--sac-data-port", type=int, default=9000)
    ap.add_argument("--sac-session-port", type=int, default=9001)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--serve-port", type=int, default=8766,
                    help="port the broker's --upstream-port should point at")
    ap.add_argument("--mock", action="store_true",
                    help="synthesize frames; no game or account needed")
    args = ap.parse_args()

    source = (MockSource() if args.mock
              else SacSource(args.sac_host, args.sac_data_port,
                             args.sac_session_port))
    Server(source, args.bind, args.serve_port).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())

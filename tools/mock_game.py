#!/usr/bin/env python3
"""A fake game: a pad server and a telemetry broker on one instance's ports.

Exists so the fleet plumbing can be tested without N copies of Trackmania and
N Ubisoft accounts. It speaks both protocols and simulates a car with enough
physics to be driven - throttle accelerates, steering turns, the race clock
runs, and a give-up press resets it - so a full training loop can be run
against it and the multi-instance wiring verified before any real game is
involved.

    python3 tools/mock_game.py --instance 1

It is NOT a training target. The physics are a toy and a policy trained
against it has learned nothing about Trackmania. The only question it answers
is whether N environments, N pads and N brokers are correctly wired to N
separate simulations, which is exactly the thing that is otherwise impossible
to test without the accounts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.ports import instance_ports   # noqa: E402


class Car:
    """A car on a straight road running along +Z from the origin."""

    def __init__(self, instance: int):
        self.instance = instance
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.heading = 0.0          # radians, 0 = +Z
        self.speed = 0.0
        self.t0 = time.time()
        self.finished = False
        self.gear = 1

    def control(self, steer, gas, brake):
        with self.lock:
            self.steer, self.gas, self.brake = steer, gas, brake

    steer = gas = brake = 0.0

    def tick(self, dt: float):
        with self.lock:
            self.speed += (self.gas * 12.0 - self.brake * 20.0
                           - 0.4 - 0.02 * self.speed) * dt
            self.speed = max(0.0, min(self.speed, 110.0))
            self.heading += self.steer * dt * 1.2 * min(self.speed / 20.0, 1.0)
            self.x += math.sin(self.heading) * self.speed * dt
            self.z += math.cos(self.heading) * self.speed * dt
            self.gear = max(1, min(5, int(self.speed / 22) + 1))
            if self.z > 900:
                self.finished = True

    def record(self) -> dict:
        with self.lock:
            c, s = math.cos(self.heading), math.sin(self.heading)
            return {
                "car": True, "map": f"MOCK{self.instance}",
                "race_time": int((time.time() - self.t0) * 1000),
                "ui": 1, "finished": self.finished,
                "pos": [self.x, self.y, self.z],
                "vel": [s * self.speed, 0.0, c * self.speed],
                "dir": [s, 0.0, c], "up": [0.0, 1.0, 0.0],
                "left": [c, 0.0, -s],
                "speed": self.speed, "gear": self.gear,
                "rpm": 2000 + (self.speed % 22) * 300,
                "ground": True, "adherence": 1.0,
                "slip": [0.0] * 4, "mat": [16] * 4,
                "icing": [0.0] * 4, "dirt": [0.0] * 4, "wetness": 0.0,
                "turbo": False, "turbo_time": 0.0, "turbo_lvl": 0,
                "reactor_lvl": 0, "reactor_type": 0, "reactor_timer": 0.0,
                "cruise": 0, "sim_coef": 1.0, "side_speed": 0.0,
                "air_brake": 0.0,
                "in_steer": self.steer, "in_gas": self.gas,
                "in_brake": self.brake,
            }


def serve_pad(port: int, car: Car):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(5)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=_pad_client, args=(conn, car),
                         daemon=True).start()


def _pad_client(conn, car: Car):
    buf = b""
    with conn:
        while True:
            try:
                data = conn.recv(1024)
            except OSError:
                return
            if not data:
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                p = line.decode(errors="ignore").split()
                if not p:
                    continue
                try:
                    if p[0] == "act":
                        car.control(float(p[1]), float(p[2]), float(p[3]))
                    elif p[0] == "reset":
                        car.control(0.0, 0.0, 0.0)
                    elif p[0] == "press":
                        # b is give up: put the car back on the start line.
                        if p[1].lower() == "b":
                            car.reset()
                    elif p[0] == "state":
                        conn.sendall(json.dumps(
                            {"steer": car.steer, "gas": car.gas,
                             "brake": car.brake, "buttons": [],
                             "age": 0.0}).encode() + b"\n")
                        continue
                    conn.sendall(b"ok\n")
                except (IndexError, ValueError) as ex:
                    conn.sendall(f"err {ex}\n".encode())


def serve_broker(port: int, car: Car, hz: float):
    """Stands in for the broker: streams telemetry, answers plugin commands."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(5)
    clients: list[socket.socket] = []
    lock = threading.Lock()

    def accept():
        while True:
            conn, _ = srv.accept()
            with lock:
                clients.append(conn)
            threading.Thread(target=commands, args=(conn,), daemon=True).start()

    def commands(conn):
        buf = b""
        while True:
            try:
                data = conn.recv(4096)
            except OSError:
                return
            if not data:
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = line.decode(errors="ignore").strip()
                reply = None
                if cmd == "landmarks":
                    reply = {"reply": "landmarks", "ok": True, "items": [
                        {"kind": "spawn", "pos": [0, 0, 0]},
                        {"kind": "checkpoint", "pos": [0, 0, 300]},
                        {"kind": "checkpoint", "pos": [0, 0, 600]},
                        {"kind": "finish", "pos": [0, 0, 900]},
                    ]}
                elif cmd.startswith("dumpmap"):
                    reply = {"reply": "dumpmap", "ok": True, "uid": f"MOCK",
                             "base_height": 8, "blocks": 0, "items": 0,
                             "effects": [], "boxes": [], "names": []}
                elif cmd == "restart":
                    car.reset()
                    reply = {"reply": "restart", "ok": True}
                elif cmd == "ping":
                    reply = {"reply": "ping", "ok": True}
                if reply is not None:
                    try:
                        conn.sendall(json.dumps(reply).encode() + b"\n")
                    except OSError:
                        return

    threading.Thread(target=accept, daemon=True).start()

    dt = 1.0 / hz
    while True:
        time.sleep(dt)
        car.tick(dt)
        payload = json.dumps(car.record()).encode() + b"\n"
        with lock:
            dead = []
            for c in clients:
                try:
                    c.sendall(payload)
                except OSError:
                    dead.append(c)
            for c in dead:
                clients.remove(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--hz", type=float, default=100.0)
    args = ap.parse_args()

    ports = instance_ports(args.instance)
    car = Car(args.instance)
    threading.Thread(target=serve_pad, args=(ports["pad"], car),
                     daemon=True).start()
    print(f"mock game {args.instance}: pad {ports['pad']}, "
          f"broker {ports['broker']}", flush=True)
    try:
        serve_broker(ports["broker"], car, args.hz)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Closed-loop test: send inputs on the pad socket, watch them come back in
the telemetry stream, and measure the actuation round-trip.

This is the check that the two halves are actually the same loop - that what we
write on 8765 reaches the car and is observable on 8766 - and it puts a number
on the latency the control loop has to budget for.

    python3 tools/closed_loop_test.py
"""
import json
import socket
import statistics
import sys
import threading
import time

PAD = ("127.0.0.1", 8765)
TELEM = ("127.0.0.1", 8766)

# (label, steer, gas, brake, hold_s, field to watch, expected value)
STEPS = [
    ("centre",      0.0, 0.0, 0.0, 0.8, "in_steer", 0.0),
    ("full gas",    0.0, 1.0, 0.0, 1.2, "in_gas",   1.0),
    ("steer left", -1.0, 1.0, 0.0, 1.2, "in_steer", -1.0),
    ("steer right", 1.0, 1.0, 0.0, 1.2, "in_steer", 1.0),
    ("half right",  0.5, 0.5, 0.0, 1.2, "in_steer", 0.5),
    ("brake",       0.0, 0.0, 1.0, 1.2, "in_brake", True),
    ("release",     0.0, 0.0, 0.0, 0.8, "in_gas",   0.0),
]

TOL = 0.06


class Telemetry(threading.Thread):
    """Reads the stream continuously so we always have the newest sample."""

    daemon = True

    def __init__(self, sock):
        super().__init__()
        self.sock = sock
        self.latest = None
        self.samples = []
        self.recording = False
        self.lock = threading.Lock()
        self.stop = False

    def run(self):
        buf = b""
        while not self.stop:
            try:
                data = self.sock.recv(65536)
            except OSError:
                return
            if not data:
                return
            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                with self.lock:
                    self.latest = rec
                    if self.recording:
                        self.samples.append((now, rec))

    def start_recording(self):
        with self.lock:
            self.samples = []
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False
            return list(self.samples)


def matches(rec, field, want):
    got = rec.get(field)
    if isinstance(want, bool):
        return bool(got) == want
    if got is None:
        return False
    return abs(float(got) - want) <= TOL


def main():
    try:
        tsock = socket.create_connection(TELEM, timeout=5)
    except OSError as ex:
        print(f"No telemetry on {TELEM}: {ex}  (is the plugin loaded?)")
        return 1
    try:
        psock = socket.create_connection(PAD, timeout=5)
    except OSError as ex:
        print(f"No pad server on {PAD}: {ex}")
        return 1

    telem = Telemetry(tsock)
    telem.start()
    time.sleep(0.5)

    with telem.lock:
        first = telem.latest
    if first is None:
        print("Telemetry connected but no samples arrived.")
        return 1
    if not first.get("car"):
        print("Warning: car=false - not in a drivable state. Inputs may not register.")
    print(f"spawn={first.get('spawn')} ui={first.get('ui')} "
          f"map={first.get('map')}\n")

    latencies = []
    print(f"{'step':<12} {'field':<10} {'result':<10} {'latency':>9}")
    print("-" * 45)

    for label, steer, gas, brake, hold, field, want in STEPS:
        telem.start_recording()
        t0 = time.perf_counter()
        psock.sendall(f"act {steer} {gas} {brake}\n".encode())
        psock.recv(64)

        deadline = t0 + hold
        hit = None
        while time.perf_counter() < deadline:
            with telem.lock:
                rec = telem.latest
            if rec and matches(rec, field, want):
                hit = time.perf_counter() - t0
                break
            time.sleep(0.002)
        telem.stop_recording()

        if hit is None:
            print(f"{label:<12} {field:<10} {'NOT SEEN':<10} {'-':>9}")
        else:
            latencies.append(hit * 1000)
            print(f"{label:<12} {field:<10} {'ok':<10} {hit*1000:>7.1f}ms")
        time.sleep(max(0.0, deadline - time.perf_counter()))

    psock.sendall(b"reset\n")
    psock.recv(64)
    telem.stop = True

    print("-" * 45)
    if latencies:
        print(f"round-trip: min {min(latencies):.1f}ms  "
              f"median {statistics.median(latencies):.1f}ms  "
              f"max {max(latencies):.1f}ms   n={len(latencies)}/{len(STEPS)}")
    else:
        print("No inputs were observed in telemetry at all.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

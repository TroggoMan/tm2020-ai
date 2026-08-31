#!/usr/bin/env python3
"""Bring up and supervise the per-instance plumbing for N game instances.

Each game instance needs three things of its own, and the port convention from
FLEET.md ties them together by index:

    instance 0    pad 8765   plugin 8766   broker 8767
    instance 1    pad 8775   plugin 8776   broker 8777
    instance 2    pad 8785   plugin 8786   broker 8787

The stride is ten, not one: the three base ports are adjacent, so stepping
each by the instance index walks them straight into each other - instance 1's
pad would land on instance 0's plugin. Instance 0 keeps exactly the ports
everything already uses.

The plugin's listen port is the one thing this script cannot set - it lives in
the game's own Openplanet config, per Wine prefix. Everything else is started
and watched here.

    python3 tools/fleet.py --instances 2          # start and supervise
    python3 tools/fleet.py --instances 2 --check  # just report what is up

WHAT THIS DOES NOT DO, deliberately: it does not create Ubisoft accounts and it
does not log in. One account can only be signed in once at a time, so N
concurrent games need N accounts - created BY HAND, once each, and signed in
once interactively per Wine prefix, after which the prefix is snapshotted and
reused. Automating that login is the single thing most likely to get the
accounts banned, and a retry loop against Ubisoft's auth endpoint is
indistinguishable from credential stuffing from their side. See FLEET.md.

The pad server needs the SYSTEM python, not the venv: it imports evdev, which
is installed system-wide, and it needs permission to create uinput devices.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT)
from env.ports import instance_ports   # noqa: E402  (stdlib only - no numpy)


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def system_python() -> str:
    """The interpreter that can see evdev and create uinput devices.

    Explicitly not the venv: the venv has torch and no evdev, and a pad server
    that starts under it fails with an ImportError several seconds after
    everything else has come up, which reads as "the game isn't responding".
    """
    for cand in ("/usr/bin/python3", shutil.which("python3")):
        if not cand:
            continue
        try:
            subprocess.run([cand, "-c", "import evdev"], check=True,
                           capture_output=True, timeout=10)
            return cand
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                OSError):
            continue
    return ""


class Proc:
    """One supervised child, restarted if it dies."""

    def __init__(self, name: str, argv: list[str], log: str):
        self.name = name
        self.argv = argv
        self.log = log
        self.p: subprocess.Popen | None = None
        self.restarts = 0
        self.started = 0.0

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.log), exist_ok=True)
        f = open(self.log, "a", buffering=1)
        f.write(f"\n--- {self.name} started {time.strftime('%F %T')} ---\n")
        self.p = subprocess.Popen(self.argv, cwd=ROOT, stdout=f,
                                  stderr=subprocess.STDOUT)
        self.started = time.time()

    def alive(self) -> bool:
        return self.p is not None and self.p.poll() is None

    def stop(self) -> None:
        if not self.alive():
            return
        self.p.terminate()
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.p.kill()


def check(instances: int) -> int:
    """Report what is actually listening, without starting anything."""
    print(f"{'inst':>4}  {'pad':>10}  {'broker':>10}  {'plugin':>10}")
    missing = 0
    for i in range(instances):
        p = instance_ports(i)
        pad, brk, plg = p["pad"], p["broker"], p["plugin"]
        row = []
        for port in (pad, brk, plg):
            up = port_open(port)
            missing += 0 if up else 1
            row.append(f"{port}{' up' if up else ' DOWN'}")
        print(f"{i:>4}  {row[0]:>10}  {row[1]:>10}  {row[2]:>10}")
    if missing:
        print(f"\n{missing} port(s) not listening.")
        print("  pad/broker: this script starts those (drop --check).")
        print("  plugin: set each game's Openplanet TMAITelemetry port to the "
              "number above, in that game's own Wine prefix, and reload it.")
    return 0 if not missing else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=2)
    ap.add_argument("--check", action="store_true",
                    help="report which ports are listening and exit")
    ap.add_argument("--no-pad", action="store_true",
                    help="do not start pad servers (they are already running)")
    ap.add_argument("--no-broker", action="store_true")
    ap.add_argument("--seats", type=int, default=1,
                    help="splitscreen: N pads into ONE game, sharing one "
                         "broker (pads 8765/8775/8785/8795)")
    args = ap.parse_args()

    n = max(1, args.instances)
    if args.check:
        return check(n)

    py = system_python()
    if not py and not args.no_pad:
        print("no system python with evdev found. Install it with:\n"
              "  sudo pacman -S python-evdev\n"
              "The pad server cannot run inside the venv - it needs evdev and "
              "uinput permission.", file=sys.stderr)
        return 1

    procs: list[Proc] = []

    if args.seats > 1:
        # Splitscreen: N pads into ONE game. One broker, not N - every seat
        # reads its own entry out of the same telemetry stream, so a second
        # broker would just be a second connection to the same plugin.
        from env.ports import seat_ports
        for i in range(args.seats):
            p = seat_ports(i)
            if not args.no_pad:
                procs.append(Proc(
                    f"pad{i}",
                    [py, "control/virtual_pad_server.py",
                     "--port", str(p["pad"]), "--instance", str(i)],
                    os.path.join(ROOT, "logs", f"pad{i}.log")))
        if not args.no_broker:
            p = seat_ports(0)
            procs.append(Proc(
                "broker0",
                [sys.executable, "telemetry/broker.py",
                 "--port", str(p["broker"]),
                 "--upstream-port", str(p["plugin"])],
                os.path.join(ROOT, "logs", "broker0.log")))
        return _supervise(procs, args.seats, seats=True)

    for i in range(n):
        ports = instance_ports(i)
        if not args.no_pad:
            procs.append(Proc(
                f"pad{i}",
                [py, "control/virtual_pad_server.py",
                 "--port", str(ports["pad"]), "--instance", str(i)],
                os.path.join(ROOT, "logs", f"pad{i}.log")))
        if not args.no_broker:
            procs.append(Proc(
                f"broker{i}",
                [sys.executable, "telemetry/broker.py",
                 "--port", str(ports["broker"]),
                 "--upstream-port", str(ports["plugin"])],
                os.path.join(ROOT, "logs", f"broker{i}.log")))

    return _supervise(procs, n)


def _supervise(procs: list[Proc], n: int, seats: bool = False) -> int:
    """Start everything, then keep it alive until interrupted."""
    for p in procs:
        p.start()
        print(f"started {p.name}: {' '.join(p.argv[1:])}", flush=True)

    stopping = False

    def shutdown(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    label = "seat" if seats else "instance"
    print(f"\nsupervising {len(procs)} processes for {n} {label}(s). "
          f"Ctrl-C to stop.\n", flush=True)
    time.sleep(1.5)
    if seats:
        from env.ports import seat_ports
        print(f"{'seat':>4}  {'pad':>10}  {'broker':>10}")
        for i in range(n):
            sp = seat_ports(i)
            print(f"{i:>4}  {sp['pad']}{' up' if port_open(sp['pad']) else ' DOWN'}"
                  f"  {sp['broker']}{' up' if port_open(sp['broker']) else ' DOWN'}")
        print(f"\nStart training with:  --seats {n}", flush=True)
        print("All seats share ONE game in splitscreen, so one account drives "
              "all of them. Bind each seat to its own pad in the game's "
              "controller settings.\n", flush=True)
    else:
        check(n)
        print(f"\nStart training with:  --instances {n}", flush=True)
        print("Each instance needs its OWN game, signed in to its OWN Ubisoft "
              "account - the same account cannot be logged in twice. Create "
              "those by hand; do not script the login.\n", flush=True)

    last_report = 0.0
    while not stopping:
        time.sleep(1.0)
        for p in procs:
            if not p.alive():
                # Restart, but not in a tight loop: something that dies
                # instantly every time is a configuration problem, and hammering
                # it just buries the reason in the log.
                if time.time() - p.started < 5.0:
                    time.sleep(5.0)
                p.restarts += 1
                print(f"{p.name} died (restart #{p.restarts}); see {p.log}",
                      flush=True)
                p.start()
        now = time.time()
        if now - last_report > 60 and not seats:
            last_report = now
            down = [str(i) for i in range(n)
                    if not port_open(instance_ports(i)["plugin"])]
            if down:
                print(f"instances with no plugin connected: {', '.join(down)}",
                      flush=True)

    print("\nstopping", flush=True)
    for p in procs:
        p.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

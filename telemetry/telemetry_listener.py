#!/usr/bin/env python3
"""Client for the TMAITelemetry plugin's JSON-lines stream.

The plugin is the server (127.0.0.1:8766) and this is the client, so this
process can be restarted freely without the game noticing.

Default mode prints a live one-line status so you can eyeball that real data
is flowing. --raw dumps every line verbatim, --record writes them to a file
for offline inspection, --stats reports the actual arrival rate (worth knowing
before assuming a fixed control period).
"""
import argparse
import json
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 8766

UI_SEQUENCE = {
    0: "None", 1: "Playing", 2: "Intro", 3: "Outro", 4: "Podium",
    5: "CustomMTClip", 6: "EndRound", 7: "PlayersPresentation",
    8: "UIInteraction", 9: "RollingBgIntro", 10: "CustomMTClipUI", 11: "Finish",
}


def status_line(d):
    if not d.get("car"):
        return f"no car   ui={UI_SEQUENCE.get(d.get('ui'), d.get('ui'))}"
    pos = d.get("pos", [0, 0, 0])
    kmh = d.get("speed", 0.0) * 3.6
    return (
        f"{kmh:7.1f} km/h  g{d.get('gear', 0)} {d.get('rpm', 0):6.0f}rpm  "
        f"({pos[0]:8.1f},{pos[1]:7.1f},{pos[2]:8.1f})  "
        f"s={d.get('in_steer', 0):+.2f} g={d.get('in_gas', 0):.2f} "
        f"b={'Y' if d.get('in_brake') else 'n'}  "
        f"{'gnd' if d.get('ground') else 'AIR'} skid{d.get('skidding', 0)}  "
        f"t={d.get('race_time', 0)}ms cp={d.get('cp', 0)}  "
        f"{UI_SEQUENCE.get(d.get('ui'), d.get('ui'))}"
    )


def connect(host, port, retry):
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            print(f"Connected to plugin at {host}:{port}", flush=True)
            return sock
        except OSError as ex:
            if not retry:
                print(f"Could not connect to {host}:{port}: {ex}", flush=True)
                return None
            print(f"\rWaiting for plugin at {host}:{port}... ({ex.strerror})",
                  end="", flush=True)
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--raw", action="store_true", help="print every line verbatim")
    ap.add_argument("--record", metavar="FILE", help="append raw lines to FILE")
    ap.add_argument("--stats", action="store_true", help="report arrival rate only")
    ap.add_argument("--once", action="store_true", help="print one line and exit")
    ap.add_argument("--no-retry", action="store_true", help="fail instead of waiting")
    args = ap.parse_args()

    sink = open(args.record, "a") if args.record else None
    try:
        while True:
            sock = connect(args.host, args.port, retry=not args.no_retry)
            if sock is None:
                return 1
            n, t0, last_report = 0, time.time(), time.time()
            buf = b""
            with sock:
                while True:
                    try:
                        data = sock.recv(65536)
                    except socket.timeout:
                        continue
                    if not data:
                        print("\nPlugin closed the connection.", flush=True)
                        break
                    buf += data
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        if not raw.strip():
                            continue
                        n += 1
                        text = raw.decode(errors="replace")
                        if sink:
                            sink.write(text + "\n")
                        if args.raw or args.once:
                            print(text, flush=True)
                            if args.once:
                                return 0
                            continue
                        if args.stats:
                            continue
                        try:
                            print("\r" + status_line(json.loads(text)).ljust(150),
                                  end="", flush=True)
                        except json.JSONDecodeError:
                            print(f"\n[unparseable] {text}", flush=True)
                    now = time.time()
                    if now - last_report >= 1.0:
                        if args.stats:
                            print(f"{n} lines, {n / max(now - t0, 1e-6):.1f}/s", flush=True)
                        last_report = now
            if sink:
                sink.flush()
            if args.no_retry:
                return 0
    except KeyboardInterrupt:
        print("\nbye", flush=True)
        return 0
    finally:
        if sink:
            sink.close()


if __name__ == "__main__":
    sys.exit(main())

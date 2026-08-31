#!/usr/bin/env python3
"""Send a command to the TMAITelemetry plugin over the telemetry socket and
print its reply.

The plugin serves one client at a time, so stop any running telemetry_listener
before using this.

    ./tmai_cmd.py ping
    ./tmai_cmd.py restart                       # episode reset
    ./tmai_cmd.py goto <mapuid>                 # switch to an already-loaded map
    ./tmai_cmd.py playmap https://trackmania.exchange/mapgbx/12345
    ./tmai_cmd.py tmx 12345                     # same, by TMX map id
    ./tmai_cmd.py menu
    ./tmai_cmd.py rate 60
"""
import json
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 8766
TMX = "https://trackmania.exchange"


def send(cmd, timeout=5.0):
    sock = socket.create_connection((HOST, PORT), timeout=timeout)
    sock.sendall((cmd + "\n").encode())
    # Telemetry lines and the reply share the socket; the reply is the first
    # line that isn't a telemetry sample.
    buf = b""
    deadline = time.time() + timeout
    with sock:
        while time.time() < deadline:
            try:
                data = sock.recv(65536)
            except socket.timeout:
                break
            if not data:
                break
            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "t" in rec and ("car" in rec):
                    continue  # a telemetry sample, keep waiting
                return rec
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    args = sys.argv[1:]
    if args[0] == "tmx":
        if len(args) < 2:
            print("tmx needs a map id")
            return 1
        cmd = f"playmap {TMX}/mapgbx/{args[1]}"
    else:
        cmd = " ".join(args)

    try:
        reply = send(cmd)
    except OSError as ex:
        print(f"Could not reach the plugin on {HOST}:{PORT}: {ex}")
        print("Is it loaded, and is anything else holding the connection?")
        return 1

    if reply is None:
        print("No reply (the plugin may not support this command).")
        return 1
    print(json.dumps(reply))
    return 0 if reply.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())

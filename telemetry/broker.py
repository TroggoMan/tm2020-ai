#!/usr/bin/env python3
"""Fan-out broker for the single plugin connection.

The plugin serves exactly one client, which means the trainer, the web GUI and
any ad-hoc listener can't coexist. The broker holds that one connection and
re-serves it locally: every client gets the full telemetry stream, and any
client can send commands upstream.

    python3 telemetry/broker.py            # upstream 8766, downstream 8767

Everything else in the project should connect to the broker (8767) rather than
the plugin directly.
"""
import argparse
import json
import socket
import sys
import threading
import time

UPSTREAM = ("127.0.0.1", 8766)
DOWNSTREAM = ("127.0.0.1", 8767)


class Broker:
    def __init__(self, upstream, downstream):
        self.upstream_addr = upstream
        self.downstream_addr = downstream
        self.clients: list[socket.socket] = []
        self.lock = threading.Lock()
        self.up: socket.socket | None = None
        self.latest: dict | None = None
        self.lines = 0
        self.connected_since: float | None = None

    # -- upstream ---------------------------------------------------------

    def pump_upstream(self):
        buf = b""
        while True:
            if self.up is None:
                try:
                    self.up = socket.create_connection(self.upstream_addr, timeout=5)
                    self.up.settimeout(1.0)
                    self.connected_since = time.time()
                    print(f"upstream connected {self.upstream_addr}", flush=True)
                except OSError:
                    time.sleep(2.0)
                    continue
            try:
                data = self.up.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                data = b""
            if not data:
                print("upstream lost, reconnecting", flush=True)
                try:
                    self.up.close()
                except OSError:
                    pass
                self.up = None
                self.connected_since = None
                buf = b""
                time.sleep(1.0)
                continue

            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                self.lines += 1
                try:
                    rec = json.loads(raw)
                    if "car" in rec:
                        self.latest = rec
                except json.JSONDecodeError:
                    pass
                self.fanout(raw + b"\n")

    def fanout(self, payload: bytes):
        with self.lock:
            dead = []
            for c in self.clients:
                try:
                    c.sendall(payload)
                except OSError:
                    dead.append(c)
            for c in dead:
                self.clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    def send_upstream(self, line: bytes) -> bool:
        if self.up is None:
            return False
        try:
            self.up.sendall(line)
            return True
        except OSError:
            return False

    # -- downstream -------------------------------------------------------

    def handle_client(self, conn: socket.socket):
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
                # Anything a client writes is a command for the plugin.
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        self.send_upstream(line + b"\n")
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

    def serve(self):
        threading.Thread(target=self.pump_upstream, daemon=True).start()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(self.downstream_addr)
        srv.listen(8)
        print(f"broker listening on {self.downstream_addr}", flush=True)
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self.handle_client, args=(conn,),
                             daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-port", type=int, default=UPSTREAM[1])
    ap.add_argument("--port", type=int, default=DOWNSTREAM[1])
    ap.add_argument("--bind", default="127.0.0.1",
                    help="address to listen on. Loopback by default. Use "
                         "0.0.0.0 to let a learner on ANOTHER MACHINE read "
                         "this game's telemetry - that is what spreads the "
                         "games across boxes while keeping one learner. It "
                         "also exposes the stream to your LAN, so only do it "
                         "on a network you trust.")
    args = ap.parse_args()
    # Upstream is always local: the plugin runs inside the game on this box.
    b = Broker(("127.0.0.1", args.upstream_port), (args.bind, args.port))
    try:
        b.serve()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())

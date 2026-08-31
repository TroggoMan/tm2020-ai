#!/usr/bin/env python3
"""Send commands to virtual_pad_server.py. (nc isn't installed on this box.)

    ./padctl.py act 0.5 1.0 0.0     # steer half right, full gas
    ./padctl.py reset
    ./padctl.py demo                # scripted wiggle: gas, steer L, steer R, stop
"""
import socket
import sys
import time

HOST, PORT = "127.0.0.1", 8765


def send(sock, line):
    sock.sendall((line + "\n").encode())
    return sock.recv(64).decode().strip()


def demo(sock):
    steps = [
        ("reset", 0.5),
        ("act 0.0 1.0 0.0", 1.5),   # straight, full gas
        ("act -0.8 1.0 0.0", 1.0),  # hard left
        ("act 0.8 1.0 0.0", 1.0),   # hard right
        ("act 0.0 0.0 1.0", 1.5),   # full brake
        ("reset", 0.0),
    ]
    for cmd, hold in steps:
        print(f"{cmd:24s} -> {send(sock, cmd)}")
        time.sleep(hold)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    sock = socket.create_connection((HOST, PORT), timeout=3)
    try:
        if sys.argv[1] == "demo":
            demo(sock)
        else:
            print(send(sock, " ".join(sys.argv[1:])))
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

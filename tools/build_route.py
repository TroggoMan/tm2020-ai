#!/usr/bin/env python3
"""Build the route model for a map and cache it for the panel.

    tools/build_route.py <map-uid>

Needs the venv (numpy). Landmarks come from the running plugin via the broker
(:8767); the occupancy grid from maps/<uid>.json (fetched on the first training
run, or delete it to force a re-fetch). Writes maps/<uid>.roadtrace.json, which
the stdlib control panel reads directly for its route editor.
"""
import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.lidar import OccupancyGrid
from env.mapdata import Gates
from env.routemodel import merged_line

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def landmarks(port=8767):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.settimeout(5)
    s.sendall(b"landmarks\n")
    buf = b""
    while b"\n" not in buf:
        d = s.recv(65536)
        if not d:
            break
        buf += d
    for ln in buf.split(b"\n"):
        if not ln.strip():
            continue
        j = json.loads(ln)
        if j.get("cmd") == "landmarks" and isinstance(j.get("items"), list):
            return j["items"]
    return []


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    uid = sys.argv[1]
    gpath = os.path.join(ROOT, "maps", f"{uid}.json")
    dump = None
    if os.path.isfile(gpath):
        with open(gpath) as f:
            dump = json.load(f)
        if not dump.get("boxes"):
            grid = OccupancyGrid.load(gpath)
            dump = {"boxes": getattr(grid, "boxes", None),
                    "names": getattr(grid, "names", None),
                    "base_height": getattr(grid, "base_height", 8),
                    "block_size": list(getattr(grid, "block", (32, 8, 32)))}
    else:
        print(f"no occupancy cache at {gpath} - run a training pass first")

    items = landmarks()
    if not items:
        sys.exit("no landmarks from the broker - is a map loaded and the plugin on?")
    g = Gates(items)
    print(f"landmarks: {g.describe()}")

    rm = merged_line(ROOT, uid, gates=g, dump=dump, verbose=True)
    if rm is None:
        sys.exit("could not build a route model")
    print(f"\nsource   : {rm['source']}")
    print(f"order    : {rm['order']}")
    print(f"length   : {rm['line'].length:.0f} m  ({len(rm['line'].points)} pts)")
    print(f"cached -> maps/{uid}.roadtrace.json  (panel reads this)")


if __name__ == "__main__":
    main()

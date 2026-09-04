#!/usr/bin/env python3
"""Cache the geometry of the CURRENTLY LOADED map, on demand.

The occupancy grid and the road trace were only ever built as a side effect of
`TrackmaniaEnv._load_map`, i.e. on the first reset of a training run. So a map
you just loaded - or just saved a new copy of in the editor, which changes its
uid - has nothing on disk until a trainer has driven it once, and if that
first reset lands while the game is still loading the track it caches nothing
and (before the retry guard) never tried again.

This does the same two dumps as a standalone step:

    tools/dump_map.py                 # occupancy + roadtrace for the live map
    tools/dump_map.py --force         # re-dump even if the cache already exists
    tools/dump_map.py --port 8766     # talk straight to the plugin, not the broker

Writes maps/<uid>.json (occupancy) and maps/<uid>.roadtrace.json.
Runs on either python; needs numpy (use the venv).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.lidar import OccupancyGrid, pick_convention          # noqa: E402
from env import roadtrace                                     # noqa: E402
from env.mapdata import GATE_CLUSTER_M                         # noqa: E402


def ask(cmd: str, wait: float, port: int) -> dict | None:
    """Send one command, return the first non-telemetry JSON line."""
    s = socket.create_connection(("127.0.0.1", port), timeout=wait + 5)
    s.sendall((cmd + "\n").encode())
    s.settimeout(3.0)
    buf = b""
    deadline = time.time() + wait
    try:
        while time.time() < deadline:
            try:
                data = s.recv(1 << 20)
            except socket.timeout:
                continue
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
                if "t" in rec and "car" in rec:
                    continue                             # a telemetry sample
                return rec
    finally:
        s.close()
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767,
                    help="8767 = telemetry broker (default), 8766 = plugin direct")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if maps/<uid>.json already exists")
    a = ap.parse_args()

    lm = ask("landmarks", 8.0, a.port)
    if not lm or not lm.get("ok"):
        sys.exit(f"no landmarks reply on :{a.port} - is a map loaded and the "
                 f"plugin up? ({(lm or {}).get('err', 'no reply')})")
    uid = lm.get("map")
    if not uid:
        sys.exit("landmarks reply carried no map uid")

    items = lm.get("items") or []
    spawn = next((it["pos"] for it in items if it.get("kind") == "spawn"), None)
    fins = [it["pos"] for it in items if it.get("kind") == "finish"]
    # A wide / linked finish is several 'Goal' landmarks; use their centroid so
    # the line ends in the middle of it, not off at the first gate.
    finish = roadtrace._cluster_xz(fins, GATE_CLUSTER_M)[0].tolist() if fins \
        else None
    raw_cps = [it["pos"] for it in sorted(
        (it for it in items if it.get("kind") == "checkpoint"),
        key=lambda it: it.get("order", 0))]
    # Linked / multi-block checkpoints arrive as several landmarks at the same
    # `order`; the game only needs the car through one of them. Collapse each
    # group to its centroid so the trace targets one gate, not a slalom.
    cps = [c.tolist() for c in roadtrace._cluster_xz(raw_cps, GATE_CLUSTER_M)] \
        if raw_cps else []
    print(f"map {uid}: spawn={'yes' if spawn else 'no'}  "
          f"checkpoints={len(cps)} (from {len(raw_cps)} landmarks)  "
          f"finish={'yes' if finish else 'no'}")

    occ_path = os.path.join(ROOT, "maps", f"{uid}.json")
    grid = None
    if os.path.exists(occ_path) and not a.force:
        grid = OccupancyGrid.load(occ_path)
        print(f"  occupancy: cache already present ({len(grid or [])} cells) "
              f"- use --force to rebuild")

    if grid is None or not len(grid):
        dump = ask("dumpmap occupancy", 30.0, a.port)
        if not dump or not dump.get("ok"):
            sys.exit(f"dumpmap occupancy failed: "
                     f"{(dump or {}).get('err', 'no reply')}")
        blocks = int(dump.get("blocks", 0) or 0)
        if blocks == 0:
            sys.exit("dumpmap returned 0 blocks - the map is still loading. "
                     "Wait for it to finish and run this again.")
        on_track = [p for p in ([spawn] if spawn else []) + cps
                    + ([finish] if finish else []) if p]
        conv = pick_convention(dump, on_track) if on_track else 0
        grid = OccupancyGrid.from_dump(dump, convention=conv)
        if grid is None or not len(grid):
            sys.exit("occupancy dump produced no cells (from_dump returned "
                     "nothing) - the block list may be an unknown shape")
        grid.save(occ_path)
        print(f"  occupancy: {blocks} blocks -> {len(grid)} cells "
              f"-> {occ_path}")

    # -- road trace ------------------------------------------------------------
    boxes = getattr(grid, "boxes", None)
    names = getattr(grid, "names", None)
    if not boxes:
        print("  roadtrace: skipped (cache is an old cells-only occupancy file "
              "with no per-block boxes; --force to re-pull one with boxes)")
        return 0
    if not finish or not spawn:
        print("  roadtrace: skipped (need both a spawn and a finish landmark)")
        return 0

    model = roadtrace.build_road_trace(
        boxes, names,
        getattr(grid, "base_height", 8),
        list(getattr(grid, "block", (32, 8, 32))),
        spawn, cps, finish, verbose=True)
    if model is None:
        print("  roadtrace: build failed (no start/finish road block found, or "
              "the block ribbon is disconnected). Add/adjust track pieces, or "
              "patch the line in the panel route editor.")
        return 1
    roadtrace.save(ROOT, uid, model)
    rt_path = roadtrace.cache_path(ROOT, uid)
    print(f"  roadtrace: {len(model['line'].points)} pts, "
          f"{model['line'].length:.0f} m, {model.get('coverage', 0) * 100:.0f}% "
          f"block cover -> {rt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

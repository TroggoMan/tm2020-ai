#!/usr/bin/env python3
"""What block am I on, and what is under the wheels, right now.

    tools/whereami.py                     one reading
    tools/whereami.py --watch             keep printing as you drive

Joins two things the project otherwise keeps apart: the MATERIAL the game
reports per wheel (physics - grip, slide behaviour) and the BLOCK NAME from the
occupancy dump (geometry - what the tracer and lidar see). They disagree in
ways that matter: a Platform piece reports Asphalt exactly like a road, and a
grass-surfaced road block is named nothing like the bare Grass terrain beside
it. Seeing both at once is how you build the name -> surface mapping from
evidence instead of guessing at substrings.
"""
import argparse, json, os, socket, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from env.surfaces import material_name
from env.lidar import is_drivable

AIRBORNE = "XXX_Null"


def latest(sock_buf, s, want_secs=1.0):
    buf = sock_buf
    best = None
    t0 = time.time()
    while time.time() - t0 < want_secs:
        try:
            d = s.recv(65536)
        except socket.timeout:
            break
        if not d:
            break
        buf += d
        while b"\n" in buf:
            ln, buf = buf.split(b"\n", 1)
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            p = r if r.get("pos") else next(
                (q for q in (r.get("players") or []) if q.get("pos")), None)
            if p and p.get("pos"):
                best = (p, r.get("map"))
    return best, buf


def blocks_at(dump, pos):
    names = dump["names"]
    bx, by, bz = dump["block_size"]
    bh = dump["base_height"]
    arr = np.asarray(dump["boxes"]).reshape(-1, 8)
    cx, cz = int(pos[0] // bx), int(pos[2] // bz)
    cy = pos[1] / by + bh
    out = []
    for x, y, z, dr, sx, sy, sz, ni in arr.tolist():
        if x <= cx < x + sx and z <= cz < z + sz:
            # how far the car is above this block's own coord cell
            out.append((abs((y + sy - 1) - cy), names[ni], (x, y, z), (sx, sy, sz), dr))
    out.sort()
    return out, (cx, cy, cz)


def report(p, uid, dump):
    pos = p["pos"]
    mats = p.get("mat") or []
    names = [material_name(v) for v in mats]
    grounded = sorted({n for n in names if n != AIRBORNE})
    spd = (p.get("speed") or 0) * 3.6
    print(f"  pos ({pos[0]:7.1f},{pos[1]:6.1f},{pos[2]:7.1f})   {spd:5.0f} km/h")
    print(f"  MATERIAL under wheels : {' + '.join(grounded) if grounded else 'airborne'}"
          f"   (per wheel {names})")
    if dump is None:
        print("  BLOCK: no occupancy dump for this map")
        return
    hits, cell = blocks_at(dump, pos)
    print(f"  cell x={cell[0]} z={cell[2]} (y~{cell[1]:.2f})")
    if not hits:
        print("  BLOCK: nothing in the dump at this cell")
        return
    for d, nm, coord, size, dr in hits[:4]:
        print(f"  BLOCK {nm:34s} coord={coord} size={size} dir={dr} "
              f"dY={d:.2f}  drivable={is_drivable(nm)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=float, default=2.0)
    a = ap.parse_args()
    s = socket.create_connection((a.host, a.port), timeout=10)
    s.settimeout(5)
    buf = b""
    got, buf = latest(buf, s, 2.0)
    if not got:
        print("no car position in the telemetry")
        return 1
    p, uid = got
    # The map uid must come from the LANDMARKS reply, not from a telemetry
    # record - the heartbeat frames routinely carry no `map` field, and
    # guessing the map by "which survey happens to have a block at this cell"
    # picks the wrong one: every Stadium map has blocks at cell (47,24).
    if not uid:
        s.sendall(b"landmarks\n")
        t0 = time.time()
        while time.time() - t0 < 5 and not uid:
            try:
                d = s.recv(65536)
            except socket.timeout:
                break
            if not d:
                break
            buf += d
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1)
                if not ln.strip():
                    continue
                try:
                    j = json.loads(ln)
                except ValueError:
                    continue
                if j.get("cmd") == "landmarks" or j.get("reply") == "landmarks":
                    uid = j.get("map")
                    break
    dump = None
    if uid:
        path = os.path.join("maps", f"{uid}.json")
        if os.path.isfile(path):
            try:
                cand = json.load(open(path))
                if cand.get("boxes"):
                    dump = cand
                    print(f"(map {uid} - survey {path})")
            except ValueError:
                pass
        if dump is None:
            print(f"(map {uid} - NOT surveyed; run tools/build_route.py {uid})")
    else:
        print("(could not resolve the map uid - no block lookup)")
    try:
        while True:
            report(p, uid, dump)
            if not a.watch:
                break
            print()
            time.sleep(a.every)
            got, buf = latest(buf, s, 1.5)
            if got:
                p, uid = got
    except KeyboardInterrupt:
        pass
    finally:
        s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

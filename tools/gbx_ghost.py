"""Read a TM2020 .Ghost.Gbx directly - no game, no replay, no recorder.

WHY THIS EXISTS

Every other way of getting a ghost into the bot goes through the game: install
the ghost, play it, and let tools/record_line.py watch the telemetry stream.
That works, but it needs a running game, a live plugin and a healthy disk, and
it samples the ghost at whatever rate the broker happens to deliver. When the
Steam drive died there was no path at all - which is a bad property for the one
input we care most about, a world record lap.

Parsing the file has none of those dependencies and is strictly better data:
the ghost stores the sim's own per-sample state, including the exact inputs the
car received, at the game's own rate, with exact timestamps.

FORMAT (verified against BigBang1112/gbx-net, which is the reference
implementation the community uses)

    GBX header    magic, version, compression flags, class 0x03092000
    body          LZO1X - liblzo2 via ctypes, so nothing to install
      chunk 0x0911F000  CPlugEntRecordData
        version, uncompressedSize, compressedSize, zlib payload
          start(ms), end(ms)
          EntRecordDesc[]     classId, 3 ints, MwBuffer, int
          NoticeRecordDesc[]  (v>=2)
          EntList             one element per recorded entity

Version 11 stores each entity's samples with COLUMNAR DELTA ENCODING: all the
timestamps first, then, for each byte index of the sample struct, one byte per
sample, delta-encoded across samples. So byte 47 of every sample is stored
contiguously, and each is a delta from the previous sample's byte 47. That is
why the payload compresses so well, and why scanning the raw bytes for float
triplets finds nothing - no sample is contiguous on disk.

The vehicle sample layout (CSceneVehicleVis.EntRecordDelta) is a fixed set of
byte offsets; see _decode_vehicle.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import struct
import zlib
from itertools import accumulate

GHOST_CLASS = 0x03092000
ENT_RECORD_DATA = 0x0911F000


# --- container ------------------------------------------------------------

def _lzo_decompress(comp: bytes, usize: int) -> bytes:
    """LZO1X via the system liblzo2.

    ctypes rather than python-lzo on purpose: liblzo2 is already installed on
    any desktop, and the alternative is a compiled pip dependency in a venv
    that already has to hold torch.
    """
    for name in ("liblzo2.so.2", "liblzo2.so", "/usr/lib/liblzo2.so.2"):
        try:
            lzo = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("liblzo2 not found - install lzo")
    fn = lzo.lzo1x_decompress_safe
    fn.argtypes = [ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p,
                   ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
    out = ctypes.create_string_buffer(usize)
    n = ctypes.c_ulong(usize)
    rc = fn(comp, len(comp), out, ctypes.byref(n), None)
    if rc != 0:
        raise ValueError(f"lzo1x_decompress_safe failed (rc={rc})")
    return out.raw[:n.value]


def read_body(path: str) -> bytes:
    """GBX header -> decompressed body."""
    data = open(path, "rb").read()
    if data[:3] != b"GBX":
        raise ValueError("not a GBX file")
    o = 3
    ver = struct.unpack_from("<H", data, o)[0]; o += 2
    fmt, refc, bodyc = data[o], data[o + 1], data[o + 2]; o += 3
    if fmt != ord("B"):
        raise ValueError("text-format GBX is not supported")
    if ver >= 4:
        o += 1
    cls = 0
    if ver >= 3:
        cls = struct.unpack_from("<I", data, o)[0]; o += 4
    if cls != GHOST_CLASS:
        raise ValueError(f"not a ghost: class 0x{cls:08X}")
    if ver >= 6:
        uds = struct.unpack_from("<I", data, o)[0]; o += 4 + uds
    o += 4                                   # numNodes
    next_ = struct.unpack_from("<I", data, o)[0]; o += 4
    if next_:
        raise ValueError("external reference table is not supported")
    if bodyc != ord("C"):
        return data[o:]
    usize, csize = struct.unpack_from("<II", data, o); o += 8
    return _lzo_decompress(data[o:o + csize], usize)


def map_uid(body: bytes) -> str | None:
    """The map this ghost was driven on, out of the ghost body.

    Found by the GBX string framing - a u32 length followed by that many bytes -
    rather than by scanning for anything that looks like a uid. Skin names and
    GUIDs in the same file produce 26-28 character runs too, so a shape-only
    match picks the wrong one; requiring a length prefix of exactly 27 leaves
    one candidate per file. Verified: two different players' ghosts of the same
    map both yield _PqN43rTpwod0yEWsQipDltp4qi, and a third map yields its own.

    Needed because a ghost-derived line is named after the ghost, not the map,
    so without this the handover cannot tell which map a line belongs to.
    """
    tag = struct.pack("<I", 27)
    start = 0
    while True:
        i = body.find(tag, start)
        if i < 0:
            return None
        start = i + 4
        cand = body[i + 4:i + 31]
        if len(cand) == 27 and re.fullmatch(rb"[A-Za-z0-9_-]{27}", cand):
            return cand.decode()


def find_record_data(body: bytes) -> bytes:
    """Locate chunk 0x0911F000 and inflate its payload.

    Scans for the chunk id rather than walking every node: the ghost body holds
    node classes this tool has no reason to model, and a wrong guess about one
    of them would desynchronise the whole read. The candidate is only accepted
    if its declared sizes are self-consistent AND the payload actually inflates,
    which is a far stronger check than a chunk id alone.
    """
    tag = struct.pack("<I", ENT_RECORD_DATA)
    best = None
    start = 0
    while True:
        i = body.find(tag, start)
        if i < 0:
            break
        start = i + 4
        p = i + 4
        if p + 12 > len(body):
            continue
        version, usize, csize = struct.unpack_from("<iii", body, p)
        if not (5 <= version <= 32) or not (0 < csize <= len(body) - p - 12):
            continue
        if not (0 < usize <= 512 << 20):
            continue
        blob = body[p + 12:p + 12 + csize]
        try:
            raw = zlib.decompress(blob)
        except zlib.error:
            continue
        if len(raw) != usize:
            continue
        if best is None or len(raw) > len(best[1]):
            best = (version, raw)
    if best is None:
        raise ValueError("no CPlugEntRecordData found")
    return best


# --- record data ----------------------------------------------------------

class _R:
    def __init__(self, b: bytes):
        self.b = b
        self.o = 0

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.b, self.o)[0]; self.o += 4; return v

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.b, self.o)[0]; self.o += 4; return v

    def u8(self) -> int:
        v = self.b[self.o]; self.o += 1; return v

    def data(self) -> bytes:
        n = self.i32()
        v = self.b[self.o:self.o + n]; self.o += n; return v


def parse_record(raw: bytes, version: int) -> dict:
    r = _R(raw)
    start = end = 0
    if version >= 1:
        start, end = r.i32(), r.i32()

    descs = []
    for _ in range(r.i32()):
        d = {"class_id": r.u32(), "u01": r.i32(), "u02": r.i32(),
             "u03": r.i32()}
        d["data"] = r.data()
        d["u04"] = r.i32()
        descs.append(d)

    if version >= 2:
        for _ in range(r.i32()):
            r.i32(); r.i32()
            if version >= 4:
                r.u32()

    # The continuation flag is read ONCE before the loop and then again inside
    # the body, after each element's samples - not at the top of every
    # iteration. Re-reading it per iteration eats one byte per element and
    # desynchronises everything after the first.
    elems = []
    has_next = r.u8()
    while has_next:
        e = {"type": r.i32(), "u01": r.i32(), "u02": r.i32(), "u03": r.i32()}
        e["u04"] = r.i32() if version >= 6 else e["u01"]
        if version >= 11:
            e["times"], e["samples"] = _read_encoded_deltas(r)
        else:
            e["times"], e["samples"] = _read_plain_deltas(r)
        has_next = r.u8()
        # Samples2: a per-entity event list, read to stay in sync, not used.
        while r.u8():
            r.i32(); r.i32(); r.data()
        elems.append(e)
    return {"start": start, "end": end, "descs": descs, "elems": elems}


def _read_plain_deltas(r: _R):
    """The pre-11 layout: a flat list of (time, buffer) pairs.

    Older ghosts store each sample as its own length-prefixed buffer with an
    ABSOLUTE timestamp - there is no columnar transpose and no delta encoding,
    so nothing accumulates. Buffers can differ in length between samples, so
    they are padded to the widest one; the vehicle decode only ever reads
    fixed offsets below 104 and a short trailing sample would otherwise be
    unreadable rather than merely incomplete.
    """
    times, bufs = [], []
    while r.u8():
        times.append(r.i32())
        bufs.append(r.data())
    if not bufs:
        return [], []
    width = max(len(b) for b in bufs)
    return times, [b.ljust(width, b"\0") for b in bufs]


def _read_encoded_deltas(r: _R):
    """Columnar delta decoding (version >= 11).

    Layout: numSamples, sampleSize, numSamples timestamps, then sampleSize
    slices of numSamples bytes each. Slice i holds byte i of every sample,
    delta-encoded across samples, so it cumulative-sums mod 256 back into a
    column. numpy does the whole column at once; doing it per byte in Python
    costs seconds on a long ghost.
    """
    n = r.i32()
    if n == 0:
        return [], []
    size = r.i32()
    if n < 0 or size < 0:
        raise ValueError(f"bad delta dimensions {n}x{size}")

    dt = struct.unpack_from(f"<{n}i", r.b, r.o)
    r.o += 4 * n
    times = list(accumulate(dt))

    rows = [bytearray(size) for _ in range(n)]
    for i in range(size):
        col = accumulate(r.b[r.o:r.o + n], lambda a, b: (a + b) & 0xFF)
        r.o += n
        for b, v in enumerate(col):
            rows[b][i] = v
    return times, [bytes(x) for x in rows]


# --- vehicle samples ------------------------------------------------------

def _read_transform(buf: np.ndarray, off: int):
    """The packed transform at byte 47 of a vehicle sample.

    Position is plain floats; rotation is an angle plus an axis in spherical
    form; speed is log-encoded; velocity is speed plus a spherical direction.
    """
    x, y, z = struct.unpack_from("<3f", buf, off)
    angle = struct.unpack_from("<H", buf, off + 12)[0] * math.pi / 65535
    a_head = struct.unpack_from("<h", buf, off + 14)[0] * math.pi / 32767
    a_pitch = struct.unpack_from("<h", buf, off + 16)[0] / 32767 * math.pi / 2
    speed = math.exp(struct.unpack_from("<h", buf, off + 18)[0] / 1000.0)
    v_head = struct.unpack_from("<b", buf, off + 20)[0] / 127 * math.pi
    v_pitch = struct.unpack_from("<b", buf, off + 21)[0] / 127 * math.pi / 2

    s = math.sin(angle)
    axis = (s * math.cos(a_pitch) * math.cos(a_head),
            s * math.cos(a_pitch) * math.sin(a_head),
            s * math.sin(a_pitch))
    quat = (axis[0], axis[1], axis[2], math.cos(angle))
    vel = (speed * math.cos(v_pitch) * math.cos(v_head),
           speed * math.cos(v_pitch) * math.sin(v_head),
           speed * math.sin(v_pitch))
    return (x, y, z), quat, speed, vel


def _basis(q):
    """Rotation matrix columns -> (right, up, forward) in the game's axes."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _decode_vehicle(times: np.ndarray, samples: np.ndarray) -> list[dict]:
    """CSceneVehicleVis.EntRecordDelta -> the same rows record_line --demo writes."""
    out = []
    for k in range(len(samples)):
        b = samples[k]
        pos, quat, speed, vel = _read_transform(b, 47)
        m = _basis(quat)
        right = (m[0][0], m[1][0], m[2][0])
        up = (m[0][1], m[1][1], m[2][1])
        fwd = (m[0][2], m[1][2], m[2][2])

        steer = ((b[14] / 255.0) - 0.5) * 2.0
        brake = b[18] / 255.0
        gas = min(1.0, (b[15] / 255.0) + brake)
        gear = (b[91] - 1) / 4.0
        # Wheel contact materials: 0xFF on every wheel means airborne.
        mats = (b[24], b[26], b[28], b[30])
        ground = any(mm != 0xFF for mm in mats)

        out.append({
            "pos": [round(v, 5) for v in pos],
            "vel": [round(v, 5) for v in vel],
            "dir": [round(v, 5) for v in fwd],
            "up": [round(v, 5) for v in up],
            "left": [round(-v, 5) for v in right],
            "speed": round(speed, 5),
            "gear": int(round(gear * 4)),
            "rpm": int(b[5]),
            "slip": [b[32] / 255.0, b[33] / 255.0, 0.0, 0.0],
            "adherence": None,
            "ground": bool(ground),
            "steer": round(steer, 5),
            "gas": round(gas, 5),
            "brake": round(brake, 5),
            "race_time": int(times[k]),
            "materials": list(mats),
            "ice": [b[81] / 255.0, b[82] / 255.0, b[83] / 255.0, b[84] / 255.0],
        })
    return out


def load(path: str) -> dict:
    body = read_body(path)
    uid = map_uid(body)
    version, raw = find_record_data(body)
    rec = parse_record(raw, version)

    # Pick the vehicle stream by PHYSICS, not by size.
    #
    # A ghost holds more than one vehicle stream, and they are not equivalent:
    # in the Magga WR, one starts 0.36 s late and contains a 244 m teleport
    # (a respawn, or an entity being re-seeded), while the other covers
    # 0.000-28.850 s continuously. Picking "the one with the most samples"
    # chose the broken one. The test that separates them is that a correctly
    # decoded stream's own speed field agrees with the finite difference of
    # its own positions - independent quantities out of different bytes of the
    # sample - so a stream with a discontinuity fails on the samples around it.
    best = None
    for e in rec["elems"]:
        times, samples = e["times"], e["samples"]
        if not samples or len(samples[0]) < 104:
            continue
        rows = _decode_vehicle(times, samples)
        if len(rows) < 20:
            continue
        ok = tot = 0
        for a, b in zip(rows, rows[1:]):
            dt = (b["race_time"] - a["race_time"]) / 1000.0
            if dt <= 0:
                tot = 0
                break
            fd = math.dist(a["pos"], b["pos"]) / dt
            tot += 1
            if abs(a["speed"] - fd) / max(fd, 1.0) < 0.25:
                ok += 1
        if not tot or ok / tot < 0.98:
            continue
        t0 = rows[0]["race_time"] / 1000.0
        span = (rows[-1]["race_time"] - rows[0]["race_time"]) / 1000.0
        key = (-t0, span, len(rows))
        if best is None or key > best[0]:
            best = (key, rows)
    if best is not None:
        best = best[1]
    if best is None:
        raise ValueError("no vehicle stream found in this ghost")
    return {"version": version, "start": rec["start"], "end": rec["end"],
            "descs": rec["descs"], "samples": best, "map_uid": uid}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ghost")
    ap.add_argument("--line", help="write a reference line here")
    ap.add_argument("--demo", help="write demo/ghost-input samples here")
    ap.add_argument("--min-step", type=float, default=0.5,
                    help="metres between reference-line points")
    a = ap.parse_args()

    g = load(a.ghost)
    s = g["samples"]
    dist = sum(math.dist(p["pos"], q["pos"]) for p, q in zip(s, s[1:]))
    print(f"{os.path.basename(a.ghost)}")
    print(f"  record version {g['version']}   {len(s)} samples")
    print(f"  {g['start']} -> {g['end']} ms   ({g['end']/1000:.3f} s)")
    print(f"  {dist:.1f} m   top {max(r['speed'] for r in s)*3.6:.0f} km/h")
    rate = len(s) / (g["end"] / 1000.0) if g["end"] else 0
    print(f"  {rate:.1f} samples/s")
    print(f"  map {g['map_uid'] or '(unknown)'}")

    if a.demo:
        os.makedirs(os.path.dirname(a.demo) or ".", exist_ok=True)
        with open(a.demo, "w") as f:
            json.dump(s, f)
        print(f"  wrote {a.demo}  ({len(s)} samples)")

    if a.line:
        keep = [s[0]["pos"]]
        for r in s[1:]:
            if math.dist(r["pos"], keep[-1]) > a.min_step:
                keep.append(r["pos"])
        os.makedirs(os.path.dirname(a.line) or ".", exist_ok=True)
        with open(a.line, "w") as f:
            json.dump({"spacing_resampled": True, "map": g["map_uid"],
                       "points": keep}, f)
        print(f"  wrote {a.line}  ({len(keep)} points)")


if __name__ == "__main__":
    main()

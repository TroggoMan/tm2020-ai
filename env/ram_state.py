"""Read TM2020 vehicle state straight from game RAM.

Why: TMAITelemetry is an UNSIGNED Openplanet plugin -> Developer mode -> paid
account (FLEET_V2.md). /proc/<pid>/mem has none of those constraints, so a free
School-mode fleet instance can still yield real vehicle state.

Every offset below was placed by CORRELATING memory against live telemetry
across a varied drive (gas/brake/steer sweeps, on and off the road), then
re-confirmed on an independent second run. Pearson r is quoted per field. The
struct's ADDRESS changes every launch - always locate() first.

`locate_structural()` needs no telemetry at all: it recognises the struct by its
own shape (a unit quaternion, dampers in range, 0/1 contact flags, a sane gear
and rpm) and confirms it by checking that position advances by velocity. That is
the path that matters for a free account, where no plugin frame exists to seed
from.

See TICK.md for the method and the traps that produced wrong answers first.
"""
import os, re, math, struct, time

import numpy as np

# ---------------------------------------------------------------- offsets ---
# All relative to the vehicle struct base, which is where POS sits.
QUAT     = -0x10  # f32[4] rotation, (w,x,y,z), CONJUGATE of the body->world
                  #         rotation - see orientation(). |q| == 1 exactly.
POS      =  0x00  # f32[3] position                          r=1.0000
VEL      =  0x0c  # f32[3] velocity                          r=0.9995
FINISHED = -0x230 # u8     1 once the run has crossed the finish. Found by
                  #        diffing a 256 KB window across the finish line on two
                  #        separate runs and keeping only what flipped both
                  #        times; 540/540 frames agree with the plugin.
START_MS = -0x2c  # u32    clock value at the start of THIS run. Rewritten on
                  #        every respawn.
TIME_MS  =  0x18  # u32    absolute simulation clock, ms. Byte-identical copy at
                  #        0x3c; never resets on its own.
                  # race_time = TIME_MS - START_MS, which reproduces the
                  # plugin's race_time to within one 100 Hz tick across
                  # respawns (16/16 samples inside 40 ms, median 5 ms - and the
                  # plugin frame is itself that stale).
IN_STEER =  0x40  # f32    steering input, -1..1             r=0.9999
IN_GAS   =  0x44  # f32    gas pedal, 0..1                   r=1.0000
IN_BRAKE =  0x48  # u32    brake pressed, 0/1                r=1.0000
RPM      =  0x50  # f32    engine rpm                        r=0.9998
GEAR     =  0x54  # u32    current gear                      r=1.0000
CONTACT  =  0x60  # u32[4] per-wheel ground contact, 0/1     exact, see below
DAMPER   =  0x80  # f32[4] suspension travel                 r>=0.991
VEL2     =  0x68  # NOTE: this was documented as a second velocity copy. It is
                  # not - 0x60..0x6c is the contact-flag array. Do not use.
STRUCT_LEN = 0x90

# Wheel arrays in RAM run FL, FR, RR, RL; the plugin (and the policy) use
# FL, FR, RL, RR. Proven twice: the two front wheels are the only ones with a
# steer angle, and a frame with exactly one wheel off the ground mapped the
# rear pair unambiguously.
RAM_TO_PLUGIN_WHEEL = (0, 1, 3, 2)


def find_game_pid(uid=None, user=None):
    """`pgrep -x Trackmania.exe` finds nothing: under Proton comm is
    "MainThread", and TICK's gameProcessId is Wine's PID, not a Linux one.

    With a fleet there is more than one game on the box, so pass `uid` or
    `user` to say which instance you mean - each one runs as its own Linux
    user (tools/steam-instance). Unfiltered, this returns whichever the
    /proc walk happens to hit first, which with a fleet is a coin toss.
    """
    if user is not None and uid is None:
        import pwd
        uid = pwd.getpwnam(user).pw_uid
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cl = open(f"/proc/{d}/cmdline", "rb").read().decode("utf8", "replace")
        except OSError:
            continue
        if "Trackmania" in cl:
            if uid is not None:
                try:
                    if os.stat(f"/proc/{d}").st_uid != uid:
                        continue
                except OSError:
                    continue
            try:
                rss = int(re.search(r"VmRSS:\s+(\d+)",
                                    open(f"/proc/{d}/status").read()).group(1))
            except Exception:
                rss = 0
            if rss > 500_000:
                return int(d)
    return None


def _regions(pid, cap=256 * 1024 * 1024):
    out = []
    for line in open(f"/proc/{pid}/maps"):
        m = re.match(r"([0-9a-f]+)-([0-9a-f]+) (\S{4})", line)
        if not m:
            continue
        a, b, perm = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
        if perm[0] == "r" and perm[1] == "w" and (b - a) <= cap:
            out.append((a, b))
    return out


# ------------------------------------------------------------ locating it ---
def _shape_ok(f, u, i):
    """Does the 0xc0 bytes around float index `i` look like the vehicle struct?

    Vectorised over `i`. `f`/`u` are the same buffer as float32 and uint32.
    Position-independent on purpose: a scan of the whole address space takes
    ~30-50 s, so anything that depends on where the car is right now is a race.
    """
    q = f[i-4]**2 + f[i-3]**2 + f[i-2]**2 + f[i-1]**2
    ok = np.abs(q - 1.0) < 1e-5                        # unit quaternion
    ok &= np.abs(f[i-4]) <= 1.0
    ok &= (np.abs(f[i]) < 3e3) & (np.abs(f[i+2]) < 3e3)
    ok &= (f[i+1] > -5e2) & (f[i+1] < 3e3)
    ok &= (np.abs(f[i]) + np.abs(f[i+2])) > 1e-3       # not an all-zero block
    ok &= (f[i+3]**2 + f[i+4]**2 + f[i+5]**2) < 4e4    # sane velocity
    ok &= (f[i+20] >= 0) & (f[i+20] < 2.5e4)           # rpm
    ok &= (u[i+21] > 0) & (u[i+21] < 8)                # gear
    for k in range(24, 28):                            # ground-contact flags
        ok &= u[i+k] < 2
    d = f[i+32] + f[i+33] + f[i+34] + f[i+35]          # suspension, 0..2 m
    for k in range(32, 36):
        ok &= (f[i+k] >= 0) & (f[i+k] <= 2.0)
    ok &= d > 1e-6
    ok &= (f[i+17] >= -1.001) & (f[i+17] <= 1.001)     # in_steer
    ok &= (f[i+18] >= 0) & (f[i+18] <= 1.001)          # in_gas
    ok &= u[i+19] < 2                                  # in_brake
    return ok


def locate_structural(pid, confirm=6, dt=0.07, timeout=25.0):
    """Find the vehicle struct WITHOUT any telemetry.

    Screens on the struct's own shape - a unit quaternion, a moving velocity, a
    running engine, suspension in range, 0/1 flags - then confirms survivors by
    physics: over `dt`, position must advance by velocity, and it must actually
    move. The car MUST be driving while this runs; a parked car has no velocity
    to confirm against and the screen alone keeps six-figure noise.
    """
    cands = []
    with open(f"/proc/{pid}/mem", "rb", 0) as mem:
        for a, b in _regions(pid):
            try:
                mem.seek(a); buf = mem.read(b - a)
            except OSError:
                continue
            n = len(buf) // 4
            if n < 68:
                continue
            f = np.frombuffer(buf, dtype="<f4", count=n)
            u = np.frombuffer(buf, dtype="<u4", count=n)
            i = np.arange(4, n - 36)          # i indexes POS.x; struct is i-4 .. i+35
            with np.errstate(invalid="ignore", over="ignore"):
                ok = _shape_ok(f, u, i)
            for j in np.nonzero(ok)[0]:
                cands.append(a + int(i[j]) * 4)

    # Physics confirmation, scored rather than sequential. A round where a
    # candidate claims no speed proves nothing about it (the car spends much of
    # a scan stopped against a wall), so it neither passes nor fails. A single
    # contradiction is fatal; three clean confirmations are enough.
    passes = {c: 0 for c in cands}
    fails = {c: 0 for c in cands}
    dead = set()
    deadline = time.time() + timeout
    with open(f"/proc/{pid}/mem", "rb", 0) as mem:
        while time.time() < deadline:
            live = [c for c in cands if c not in dead]
            if live and all(passes[c] >= confirm for c in live):
                break
            prev = {}
            for c in live:
                try:
                    mem.seek(c); prev[c] = struct.unpack("<6f", mem.read(24))
                except OSError:
                    dead.add(c)
            t0 = time.time(); time.sleep(dt); el = time.time() - t0
            for c, p in prev.items():
                try:
                    mem.seek(c); now = struct.unpack("<6f", mem.read(24))
                except OSError:
                    dead.add(c); continue
                sp = math.sqrt(sum(v * v for v in p[3:]))
                if sp < 5.0:
                    continue                      # says nothing either way
                tol = 0.05 + 0.25 * sp * el
                if all(abs((now[k] - p[k]) - p[3 + k] * el) < tol for k in range(3)):
                    passes[c] += 1
                else:
                    fails[c] += 1
    return [c for c in cands if c not in dead and passes[c] >= confirm
            and fails[c] <= passes[c] // 3]


def locate(pid, telemetry_fn, ptol=None, vtol=None, confirm=3, settle=0.4):
    """Find the struct using one plugin telemetry frame as a seed.

    Faster and surer than locate_structural when the plugin IS available. The
    car must be moving: parked, the scene's pose objects are indistinguishable
    from the physics body (that mistake cost a whole session once).
    """
    cands = []
    with open(f"/proc/{pid}/mem", "rb", 0) as mem:
        for a, b in _regions(pid):
            # Re-seed PER REGION. A full scan takes ~50 s; a car doing 45 m/s
            # is two kilometres away by the end of it, so one seed frame taken
            # up front matches nothing in the regions read last. This was a
            # silent zero-hit failure, not an error.
            t = telemetry_fn()
            px, py, pz = t["pos"]
            speed = math.sqrt(sum(v * v for v in t["vel"]))
            tol = ptol if ptol is not None else 0.30 + 0.05 * speed
            # vtol has to scale too. A fixed 0.60 m/s looks generous at a
            # standstill and is hopeless at 46 m/s in a corner: the plugin
            # frame is a frame or two old, and hard cornering changes velocity
            # by more than that in the gap. Left fixed, this rejected the
            # CORRECT struct on every attempt while the car was quick, and
            # only ever succeeded once the car slowed down.
            vt = vtol if vtol is not None else 0.60 + 0.03 * speed
            try:
                mem.seek(a); buf = mem.read(b - a)
            except OSError:
                continue
            n = len(buf) // 4
            if n < 40:
                continue
            arr = np.frombuffer(buf, dtype="<f4", count=n)
            uarr = np.frombuffer(buf, dtype="<u4", count=n)
            with np.errstate(invalid="ignore"):
                idx = np.nonzero(np.abs(arr[:-2] - px) < tol)[0]
                if not idx.size:
                    continue
                ok = idx[(np.abs(arr[idx + 1] - py) < tol) &
                         (np.abs(arr[idx + 2] - pz) < tol)]
            ok = ok[(ok >= 4) & (ok + 36 < n)]
            if not ok.size:
                continue
            with np.errstate(invalid="ignore", over="ignore"):
                ok = ok[_shape_ok(arr, uarr, ok)]
            for i in ok:
                i = int(i)
                if all(abs(float(arr[i + 3 + k]) - t["vel"][k]) < vt for k in range(3)):
                    cands.append(a + i * 4)

    with open(f"/proc/{pid}/mem", "rb", 0) as mem:
        for _ in range(confirm):
            if not cands:
                break
            time.sleep(settle)
            t = telemetry_fn()
            p, v = t["pos"], t["vel"]
            sp = math.sqrt(sum(c * c for c in v))
            tol = ptol if ptol is not None else 0.30 + 0.05 * sp
            vt = vtol if vtol is not None else 0.60 + 0.03 * sp
            keep = []
            for a in cands:
                try:
                    mem.seek(a); raw = mem.read(0x18)
                except OSError:
                    continue
                pp = struct.unpack_from("<3f", raw, 0)
                vv = struct.unpack_from("<3f", raw, VEL)
                if all(abs(pp[k] - p[k]) < tol for k in range(3)) and \
                   all(abs(vv[k] - v[k]) < vt for k in range(3)):
                    keep.append(a)
            cands = keep
    return cands


# --------------------------------------------------------------- reading ----
def orientation(q):
    """(left, up, dir) unit vectors from the stored quaternion.

    The stored quaternion is the conjugate of the body->world rotation: used
    directly it yields a basis whose off-diagonal terms have the wrong sign
    (left.2 and dir.0 come out anti-correlated with the plugin's). Conjugating
    first drops the median error against telemetry to 3e-6, which is the
    plugin's own 5-decimal print rounding.
    """
    w, x, y, z = q[0], -q[1], -q[2], -q[3]
    # ROWS of the rotation matrix, not columns. Using the columns gives a basis
    # that is component-wise close on a straight line - two of the three terms
    # are identical - and silently wrong the moment the car yaws: `left` picks
    # up the forward direction, so side_speed comes out as the full speed.
    left = (1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w))
    up   = (2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w))
    dir_ = (2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y))
    return left, up, dir_


class VehicleState:
    def __init__(self, pid, addr):
        self.pid, self.addr = pid, addr
        self.mem = open(f"/proc/{pid}/mem", "rb", 0)

    def read(self):
        """One coherent snapshot: a single read covers quat..damper."""
        self.mem.seek(self.addr + FINISHED)
        raw = self.mem.read(STRUCT_LEN - FINISHED)
        o = -FINISHED                               # raw index of the struct base
        f = lambda k, n=3: struct.unpack_from("<%df" % n, raw, o + k)
        u = lambda k: struct.unpack_from("<I", raw, o + k)[0]

        vel = f(VEL)
        left, up, dir_ = orientation(f(QUAT, 4))
        w = RAM_TO_PLUGIN_WHEEL
        damp, cont = f(DAMPER, 4), [u(CONTACT + 4 * i) for i in range(4)]
        return {
            "pos": f(POS),
            "vel": vel,
            "left": left, "up": up, "dir": dir_,
            # speed/side_speed/front_speed are NOT stored - they are projections
            # of velocity, and reproduce the plugin's values to ~1e-3.
            "speed": math.sqrt(sum(c * c for c in vel)),
            "side_speed": sum(vel[k] * left[k] for k in range(3)),
            "front_speed": sum(vel[k] * dir_[k] for k in range(3)),
            "finished": bool(struct.unpack_from("<B", raw, o + FINISHED)[0]),
            "time_ms": u(TIME_MS),
            # A map reload resets the clock but leaves the old start stamp
            # behind, so this is briefly negative (and wraps, unsigned) until
            # the next respawn writes it. Clamp rather than emit garbage.
            "race_time": max(0, u(TIME_MS) - u(START_MS)),
            "in_steer": f(IN_STEER, 1)[0],
            "in_gas": f(IN_GAS, 1)[0],
            "in_brake": bool(u(IN_BRAKE)),
            "rpm": f(RPM, 1)[0],
            "gear": u(GEAR),
            "damper": tuple(damp[w.index(i)] for i in range(4)),
            "contact": tuple(bool(cont[w.index(i)]) for i in range(4)),
            "ground": any(cont),
        }

    def close(self):
        self.mem.close()


# --- NOT available from this struct -----------------------------------------
# ground_dist: could not be placed, and this map cannot place it - the car never
#   leaves the ground (the track is walled), so ground_dist spanned 0..0.073 and
#   every high correlate was just something tracking ride height. Retry on a map
#   with a jump. `contact[4]` covers the airborne case in the meantime.
# slip[4]: saturates to exactly 0.0 or 1.0, so it carries about one bit. Several
#   0/1 arrays in the struct track it at r=1.0 and disagree on ~6% of frames,
#   which is within telemetry staleness - not enough to call it placed.
# icing, wear, brake_coef, dirt, adherence: constant across every run so far
#   (stock car, dry road). A field that never moves cannot be correlated. Per
#   FLEET_V2 they are genuinely inactive here, which is where this matters least.
# lidar / fx_ahead: computed by OUR plugin from surveyed map geometry. Not
#   vehicle state, and can NEVER come from RAM.
#
# A second, richer object exists at a stable offset from a pose anchor: it holds
# speed, side_speed and per-wheel blocks of 44 bytes (damper +0, steer angle +12
# - nonzero only on the two front wheels, which is how the wheel order above was
# proven). It adds nothing this struct plus projection does not already give,
# so it is a lead, not a dependency.

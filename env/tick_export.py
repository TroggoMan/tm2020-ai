"""Write a run's inputs as a TICK TAS script, for playback and brute-forcing.

TICK's script format, confirmed against its own /source endpoint:

    0 seed 12345
    0 accel 1
    1000 steer 60
    5000 accel 0

Times are MILLISECONDS and tick = ms / 10, i.e. 100 Hz - the rate TM2020's
physics actually runs at, which is NOT our control rate. Steering is an integer
-127..127 and maps exactly onto our normalised -1..1 (`steer 60` was measured
producing in_steer 0.472 = 60/127).

WHY SPARSE. A script is a list of CHANGES, not a sample per tick: TICK holds
the last value until something replaces it. A 17-second lap at our 40 Hz
control rate is ~680 samples, but the car spends most of a lap with the
throttle pinned and the wheel in one place, so only the transitions matter.
Emitting every sample would produce a script that is both enormous and lies
about how the car was driven - it would imply 680 deliberate inputs where there
were perhaps 80.

WHAT IS RECORDED is what we SENT to the pad, after the action decode and any
slew limit - the same numbers a keyboard player's keystrokes would produce, not
the network's raw floats and not what the game reported applying. That is the
thing TICK replays.
"""
from __future__ import annotations

import os
import time


def to_script(samples, seed: int | None = None, quantise: int = 127) -> str:
    """Turn [(t_ms, steer -1..1, gas 0/1, brake 0/1)] into a TICK script.

    Only changes are emitted. Steering is quantised to TICK's integer scale
    FIRST and compared after, so a stream of floats that all round to the same
    integer produces one line rather than hundreds.
    """
    out = []
    if seed is not None:
        out.append(f"0 seed {int(seed)}")
    last_s = last_g = last_b = None
    for t_ms, steer, gas, brake in samples:
        t = max(0, int(round(float(t_ms))))
        s = int(round(max(-1.0, min(1.0, float(steer))) * quantise))
        g = 1 if float(gas) > 0.5 else 0
        b = 1 if float(brake) > 0.5 else 0
        # accel/brake before steer at the same instant: a TAS reads more
        # naturally as "power state, then where the wheel is".
        if g != last_g:
            out.append(f"{t} accel {g}")
            last_g = g
        if b != last_b:
            out.append(f"{t} brake {b}")
            last_b = b
        if s != last_s:
            out.append(f"{t} steer {s}")
            last_s = s
    return "\n".join(out) + ("\n" if out else "")


def write_best(root: str, map_uid: str, samples, race_time_ms,
               episode: int | None = None, seed: int | None = None) -> str | None:
    """Save a script for a new best lap. Returns the path, or None."""
    if not samples or not map_uid:
        return None
    d = os.path.join(root, "runs", map_uid, "tick")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return None
    secs = (race_time_ms or 0) / 1000.0
    name = f"best_{secs:07.3f}s"
    if episode is not None:
        name += f"_ep{episode:05d}"
    name += f"_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    path = os.path.join(d, name)
    body = to_script(samples, seed=seed)
    header = (f"# TM2020-AI export - {map_uid}\n"
              f"# lap {secs:.3f}s, {len(samples)} control samples -> "
              f"{body.count(chr(10))} input changes\n"
              f"# times are ms; tick = ms/10 (100Hz); steer is -127..127\n")
    try:
        with open(path, "w") as f:
            f.write(header + body)
    except OSError:
        return None
    return path

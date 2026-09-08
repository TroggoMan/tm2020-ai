"""Drive the warm-up from a recorded ghost's inputs.

WHY THIS RATHER THAN STUFFING THE BUFFER DIRECTLY

The obvious way to learn from a ghost is to convert its recording into
(obs, action, reward, next_obs) rows and push them into the replay buffer. It
does not survive contact with the detail: `tools/record_line.py --demo` saves
RAW TELEMETRY (pos, vel, dir, speed, gear, slip, steer/gas/brake), not the
150-dim observation, and rebuilding that offline means rebuilding the reward
offline too. The env's reward is stateful - progress along the line, the par
charge, checkpoint gates, surface terms, streaks - and an approximation of it
is not a harmless approximation: the critic learns Q from exactly those
numbers, so a demo scored differently from everything around it teaches the
critic that the ghost's states are worth something they are not.

So instead the car DRIVES the ghost's inputs. The env then produces the
observation and the reward through its own pipeline, exactly as for any other
transition, and what lands in the buffer is real. It costs wall-clock that a
direct injection would not - but it costs no correctness.

WHAT IT IS AND IS NOT

This is open loop. It replays a fixed input sequence by index, so it only
reproduces the ghost's lap while the car is where the ghost was. TM2020's
physics is deterministic, which is what makes that work at all from a common
spawn - and also why any drift is permanent rather than self-correcting. Once
the car is off the ghost's line the inputs are no longer the right ones, so
this is a WARM-UP source, not a driver: it fills the buffer with a good lap's
worth of transitions and then hands over.

Non-repeating on purpose: when the sequence runs out it holds neutral rather
than looping, because looping would splice the end of a lap onto the start of
one and record a transition that never happened.
"""
from __future__ import annotations

import json

import numpy as np


def _resample(rows: list, hz: float | None) -> list:
    """Put the samples on the control clock, holding each input until the next.

    THE CURSOR IS TIME, NOT INDEX. GhostDriver advances one sample per control
    step, so a file whose rows are not spaced at exactly 1/control_hz replays at
    the wrong speed. That is not a subtle degradation: a ghost read out of a
    .Gbx is 20 Hz against a 40 Hz control rate, so replaying it row-per-step
    runs the lap at DOUBLE speed - every input released half a lap early, the
    car nowhere near the line, and a buffer full of transitions that look like
    a world record's inputs and are nothing of the sort.

    Recorder demos have the same problem from the other direction: they append
    a row every 0.5 m of movement, so their rows are spaced by DISTANCE, and
    a slow corner is sampled densely in space but sparsely in time.

    Zero-order hold is the right interpolation here rather than something
    smoother, because these are held control positions - the pedal was down
    for that whole interval, and averaging across a release would invent an
    input the driver never made.

    Files with no usable race_time are returned untouched: guessing a clock is
    worse than leaving the caller's assumption visible.
    """
    if not hz or hz <= 0 or len(rows) < 2:
        return rows
    times = []
    for r in rows:
        t = r.get("race_time") if isinstance(r, dict) else None
        times.append(None if t is None else float(t))
    if any(t is None for t in times):
        return rows
    t0 = times[0]
    span = (times[-1] - t0) / 1000.0
    if span <= 0:
        return rows
    step = 1000.0 / hz
    out, j = [], 0
    n = int(span * hz)
    for k in range(n + 1):
        target = t0 + k * step
        while j + 1 < len(times) and times[j + 1] <= target:
            j += 1
        out.append(rows[j])
    return out


def load(path: str, hz: float | None = None) -> np.ndarray:
    """Read a --demo file into an (N, 3) array of [steer, gas, brake] actions.

    The file is telemetry, so the inputs are the ones the ghost's car RECEIVED.
    They are converted into the action space the policy uses - which is what
    the env decodes back into pedals - rather than into pedal values directly,
    so a transition recorded here means the same thing as one the policy
    produced.
    """
    with open(path) as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = rows.get("samples") or rows.get("rows") or []
    rows = _resample(rows, hz)
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        steer = float(r.get("steer", 0.0) or 0.0)
        gas = r.get("gas", 0.0)
        brake = r.get("brake", 0.0)
        gas = float(gas) if not isinstance(gas, bool) else (1.0 if gas else 0.0)
        brake = (float(brake) if not isinstance(brake, bool)
                 else (1.0 if brake else 0.0))
        # The pedals are thresholded by the env, so saturate rather than sit on
        # the threshold: +1/-1 decode to on/off under any threshold below 1.
        out.append((max(-1.0, min(1.0, steer)),
                    1.0 if gas > 0.5 else -1.0,
                    1.0 if brake > 0.5 else -1.0))
    return np.asarray(out, dtype=np.float32)


class GhostDriver:
    """Serve one ghost's actions per env, advancing independently.

    Each env keeps its own cursor: seats reset at different moments, so a
    shared index would hand seat 2 the input for a lap position seat 0 is at.
    """

    def __init__(self, actions: np.ndarray, n_envs: int = 1,
                 hz: float = 0.0):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.idx = [0] * max(1, n_envs)
        # The rate the actions were resampled onto, so a race clock in
        # milliseconds can be turned into an index. 0 means "not time-indexed"
        # and the cursor just counts steps, as it used to.
        self.hz = float(hz)

    def __len__(self) -> int:
        return len(self.actions)

    def reset_env(self, i: int) -> None:
        if 0 <= i < len(self.idx):
            self.idx[i] = 0

    def sync(self, i: int, race_time_ms) -> None:
        """Put seat i's cursor where the RACE CLOCK says it should be.

        The cursor used to be a step counter, which is only equivalent to time
        if every step advances the clock by exactly 1/hz. It does not: the
        countdown, the respawn settle and any dropped frame all consume steps
        while the race clock stands still. Each one shifts the whole remaining
        lap earlier by that much, permanently, because nothing ever pulled the
        cursor back - so the car turned before it reached the corner, at any
        control rate.

        Indexing on the clock makes those free. During the countdown race_time
        is 0 or absent, so the cursor sits on the first input and the record
        starts when the car is actually released.
        """
        if self.hz <= 0 or not (0 <= i < len(self.idx)):
            return
        # A MISSING clock means the race has not started - the countdown, the
        # spawn settle, a respawn. That is the case this exists for, so it must
        # pin the cursor at the start, NOT skip the sync. Skipping let the
        # cursor fall back to counting steps through the countdown, which spent
        # about 1.5s of the lap before the car had moved and made every corner
        # arrive that much early, consistently, at any control rate.
        k = 0 if race_time_ms is None else int(
            round(float(race_time_ms) / 1000.0 * self.hz))
        self.idx[i] = max(0, min(k, len(self.actions)))

    def batch(self, n_envs: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (actions, live) - `live[i]` is False where the ghost is spent.

        The caller needs to know WHICH seats the ghost is still driving, not
        just what it would like them to do. A spent ghost used to hand back
        neutral, which parks the car until the stuck timer fires and fills the
        buffer with a seat doing nothing for the rest of a long episode. With
        the flag, the caller can drive those seats with the scripted pursuit
        driver instead, so the ghost seeds a real world-record lap and pursuit
        takes over from where it ran out.
        """
        while len(self.idx) < n_envs:
            self.idx.append(0)
        out = np.zeros((n_envs, 3), dtype=np.float32)
        # Neutral when the sequence is spent: steer 0, and pedals at -1 so the
        # env decodes them OFF rather than leaving the throttle latched.
        out[:, 1] = -1.0
        out[:, 2] = -1.0
        live = np.zeros(n_envs, dtype=bool)
        for i in range(n_envs):
            k = self.idx[i]
            if k < len(self.actions):
                out[i] = self.actions[k]
                self.idx[i] = k + 1
                live[i] = True
        return out, live

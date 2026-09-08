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


def load(path: str) -> np.ndarray:
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

    def __init__(self, actions: np.ndarray, n_envs: int = 1):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.idx = [0] * max(1, n_envs)

    def __len__(self) -> int:
        return len(self.actions)

    def reset_env(self, i: int) -> None:
        if 0 <= i < len(self.idx):
            self.idx[i] = 0

    def batch(self, n_envs: int) -> np.ndarray:
        while len(self.idx) < n_envs:
            self.idx.append(0)
        out = np.zeros((n_envs, 3), dtype=np.float32)
        # Neutral when the sequence is spent: steer 0, and pedals at -1 so the
        # env decodes them OFF rather than leaving the throttle latched.
        out[:, 1] = -1.0
        out[:, 2] = -1.0
        for i in range(n_envs):
            k = self.idx[i]
            if k < len(self.actions):
                out[i] = self.actions[k]
                self.idx[i] = k + 1
        return out

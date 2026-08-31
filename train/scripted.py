"""A hand-written driver, used to seed the replay buffer.

The complaint that produced this file: *"the ai hasn't even tried the most
basic option of just driving in a complete straight line and just
accelerating"*. That is exactly right, and it is not a bug in the policy - it
is what SAC's warm-up does.

For the first `learning_starts` steps SB3 ignores the network entirely and
samples uniformly from the action space. With three independent uniform axes,
"full throttle, straight, for two seconds" has probability ~0 of ever
appearing: every step re-rolls the steering, so the car snakes at a random
average throttle and stops. The buffer then fills with thousands of
transitions that all say "flailing goes nowhere", and the critic's first
opinion of the world is formed from them.

So the warm-up drives instead of flailing. This is not a teacher the policy is
trained to copy - there is no imitation loss anywhere - it just means the first
few thousand transitions in the buffer contain the trivially good behaviour as
well as the bad, so the critic has something to compare against from step one.
SAC is off-policy; where the data came from does not matter to it.

Two levels, both deliberately simple:

  * `straight` - full throttle, no steering. Type A. It is the thing you would
    try first, and on a track that starts with a straight it gets real
    progress into the buffer immediately.
  * `pursuit` - full throttle plus pure-pursuit steering toward a lookahead
    point, braking only when the corner ahead is sharp and the car is fast.
    Still crude, still nothing like a racing line, but it gets round.

The observation layout is read from OBS_GROUPS rather than hard-coded offsets,
so adding an input group cannot silently make this steer off the wrong slice.
"""
from __future__ import annotations

import numpy as np

from env.hints import Hint, Tapper
from env.tm_env import LOOKAHEAD, OBS_GROUPS

# Group -> (start, length) in the observation vector.
OFFSETS: dict[str, tuple[int, int]] = {}
_at = 0
for _name, _n in OBS_GROUPS:
    OFFSETS[_name] = (_at, _n)
    _at += _n


def _group(obs: np.ndarray, name: str) -> np.ndarray:
    start, n = OFFSETS[name]
    return obs[start:start + n]


def lookahead_points(obs: np.ndarray) -> np.ndarray:
    """The lookahead points as (n, 3) in the car's frame, in metres.

    `_observe` packs them as forward/left/up triples scaled by 1/50, which is
    undone here so the geometry below is in real units.
    """
    return _group(obs, "ahead").reshape(len(LOOKAHEAD), 3) * 50.0


# One tapper per environment, keyed by index. Hints are pulse trains, so the
# driver has to remember where in the pulse it is; a stateless function cannot
# tap. Keyed by env index because a VecEnv's rows are independent cars.
_TAPPERS: dict[int, Tapper] = {}


def reset_tappers() -> None:
    _TAPPERS.clear()


def _tap(obs: np.ndarray, hints: list, env_index: int, step_ms: float) -> bool:
    """Should this car be tapping the brake right now?

    Only speed-window hints are usable here. A hint pinned to a section of
    track (`s_from`/`s_to`) cannot be honoured by the scripted driver at all:
    arc length along the reference line is not in the observation, and in a
    SubprocVecEnv the env that knows it is in another process entirely. Those
    hints still work - as the reward term, inside the env, which is where the
    number lives. This is the honest split, not an oversight.
    """
    speed_ms = float(_group(obs, "speed")[0]) * 100.0
    for h in hints:
        if h.s_from or h.s_to:
            continue
        if not h.covers(speed_ms, 0.0):
            continue
        t = _TAPPERS.setdefault(env_index, Tapper())
        return t.should_press(h, step_ms)
    return False


def drive(obs: np.ndarray, mode: str = "pursuit",
          steer_gain: float = 2.0, brake_above: float = 0.55,
          hints: list | None = None, env_index: int = 0,
          step_ms: float = 0.0) -> np.ndarray:
    """One action for one observation, in the network's own [-1, 1] space.

    Returns the same three numbers the policy would: steer, and the pre-decode
    values for gas and brake. Positive means "on" for the pedals, which is what
    the environment's binary threshold reads.
    """
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    tapping = bool(hints) and _tap(obs, hints, env_index, step_ms)
    if mode == "straight":
        # Even the trivial driver honours hints: on a straight road the only
        # way to get into a speed slide at all is a tap, and "straight" is the
        # mode the car is in when it reaches 400km/h.
        return np.array([0.0, 1.0, 1.0 if tapping else -1.0], dtype=np.float32)

    pts = lookahead_points(obs)
    # Aim at the second lookahead point (10m by default). Near enough that the
    # car can actually reach it, far enough that it is not chasing noise.
    aim = pts[min(1, len(pts) - 1)]
    forward, left = aim[0], aim[1]
    # Pure pursuit: steer in proportion to the bearing of the aim point. The
    # sign is +left because car_frame's second axis is the car's left.
    bearing = np.arctan2(left, max(forward, 1.0))
    steer = float(np.clip(-steer_gain * bearing, -1.0, 1.0))

    # Brake only when the corner is genuinely sharp AND the car is quick
    # enough for it to matter. A scripted driver that brakes for everything
    # never gets far enough down the track to put anything useful in the
    # buffer.
    speed = float(_group(obs, "speed")[0]) * 100.0
    far = pts[min(3, len(pts) - 1)]
    curve = abs(np.arctan2(far[1], max(far[0], 1.0)))
    hard = curve > brake_above and speed > 30.0
    brake = hard or tapping
    return np.array([steer, -1.0 if hard else 1.0, 1.0 if brake else -1.0],
                    dtype=np.float32)


def drive_batch(obs, mode: str = "pursuit", hints: list | None = None,
                step_ms: float = 0.0) -> np.ndarray:
    """Vectorised over a VecEnv's stacked observations."""
    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.stack([drive(row, mode, hints=hints, env_index=i,
                           step_ms=step_ms)
                     for i, row in enumerate(arr)]).astype(np.float32)

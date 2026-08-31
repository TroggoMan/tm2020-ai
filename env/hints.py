"""Hints: "tap the brake here, for about this long".

There is a gap between "the reward says go fast" and "the policy discovers a
speed slide". A speed slide needs the car to be above ~400km/h *and* to have
been knocked sideways into a narrow band of sideways speed, and the only way
into that band from a standing start is a deliberate brake tap. Uniform
exploration will not find it: a tap is a specific short pulse at a specific
speed, and the odds of SAC rolling one by accident and then rolling the follow
-up steering that keeps it are effectively zero.

So a hint says where to try, and how long the pulse should be. It is a nudge,
not a script:

  * the scripted warm-up driver *performs* hints, so the buffer contains
    genuine tapped transitions from the first minute, and
  * the reward pays a little for tapping inside the window, so the policy has
    a reason to keep the behaviour once it is off the warm-up.

Neither one forces anything at training time. The policy is always free to
stop tapping if lap time says tapping is slower - which is the whole point of
making this a hint rather than a hard-coded input sequence.

A hint is scoped three ways, and they combine (all of the ones you set must
hold):

    speed_from_kmh / speed_to_kmh    how fast the car is
    cp_from / cp_to                  which checkpoint-to-checkpoint SECTION
    s_from / s_to                    metres of arc length along the line

    {"name": "sd-entry", "speed_from_kmh": 400, "speed_to_kmh": 600,
     "hold_ms": 120, "gap_ms": 700, "w": 0.05}

    {"name": "sector-2-entry", "cp_from": 1, "cp_to": 2,
     "hold_ms": 90, "gap_ms": 500, "gas": 1.0, "w": 0.05}

**Sections are checkpoints, not times.** `cp_from: 1, cp_to: 2` means "after
checkpoint 1 has been taken and before checkpoint 2 is" - that is the natural
unit for optimising a track piece by piece, it survives the reference line
being re-recorded, and it is the same unit the split times are reported in.
`s_from`/`s_to` are the finer-grained version for when you want part of a
section; they are metres along the line, not seconds.

A hint can also **force inputs** rather than only tapping. `gas`, `brake` and
`steer` each take a value in [-1, 1] (or null for "no opinion"), and the
warm-up driver uses them in place of whatever it would have done inside that
section. That is how you get a known-input sequence into the buffer for one
piece of track without hand-writing a whole driver:

    {"name": "hold-flat-through-1", "cp_from": 0, "cp_to": 1,
     "gas": 1.0, "brake": -1.0, "steer": 0.0}

Forcing applies to the **warm-up only**. It is never applied on top of the
policy's own action later in training - that would put an action in the replay
buffer that the policy did not choose, and the critic would learn from a
transition that never happened as described.
"""
from __future__ import annotations

from dataclasses import dataclass

MS_TO_KMH = 3.6


@dataclass
class Hint:
    name: str = "hint"
    control: str = "brake"          # which pedal the tap is on
    speed_from_kmh: float = 0.0
    speed_to_kmh: float = 0.0       # 0 = no upper bound
    s_from: float = 0.0
    s_to: float = 0.0               # 0 = no upper bound
    cp_from: int = -1               # -1 = from the start line
    cp_to: int = -1                 # -1 = to the finish
    # Named places, recorded by driving to them. Resolved to s_from/s_to
    # against the loaded line, so they survive the line being re-recorded.
    mark_from: str = ""
    mark_to: str = ""
    hold_ms: float = 120.0          # how long one tap lasts
    gap_ms: float = 700.0           # quiet time before the next tap
    w: float = 0.0                  # reward per step while tapping in window
    # Forced inputs for the warm-up driver, in the network's own [-1, 1]
    # space. None means "no opinion, drive normally on this axis".
    gas: float | None = None
    brake: float | None = None
    steer: float | None = None

    @property
    def scoped(self) -> bool:
        """False for a hint that names no window at all.

        Such a hint would otherwise apply to the entire track, which is never
        what someone meant to write - it is what a typo produces.
        """
        return bool(self.speed_from_kmh or self.speed_to_kmh
                    or self.s_from or self.s_to
                    or self.cp_from >= 0 or self.cp_to >= 0)

    def covers(self, speed_ms: float, s: float = 0.0, cp: int = 0) -> bool:
        kmh = float(speed_ms) * MS_TO_KMH
        if self.speed_from_kmh and kmh < self.speed_from_kmh:
            return False
        if self.speed_to_kmh and kmh > self.speed_to_kmh:
            return False
        if self.s_from and s < self.s_from:
            return False
        if self.s_to and s > self.s_to:
            return False
        # cp is how many checkpoints have been TAKEN, so the section between
        # checkpoint 1 and checkpoint 2 is the stretch where cp == 1.
        if self.cp_from >= 0 and cp < self.cp_from:
            return False
        if self.cp_to >= 0 and cp >= self.cp_to:
            return False
        return self.scoped

    def forced(self) -> dict:
        """The axes this hint has an opinion about, as a plain dict."""
        return {k: v for k, v in (("steer", self.steer), ("gas", self.gas),
                                  ("brake", self.brake)) if v is not None}


def parse(items) -> list[Hint]:
    """Config dicts -> Hints, ignoring keys we do not know.

    Unknown keys are dropped rather than raising: this is loaded from a file
    the panel writes and that you edit by hand mid-run, and a typo should cost
    you one hint, not the training run.
    """
    out: list[Hint] = []
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        fields = {k: raw[k] for k in Hint.__dataclass_fields__ if k in raw}
        try:
            out.append(Hint(**fields))
        except TypeError:
            continue
    return out


class Tapper:
    """Turns a hint into an on/off pulse train, in wall-clock milliseconds.

    Used by the scripted warm-up driver. Kept separate from the reward so the
    two cannot drift apart: the reward asks "is the pedal down inside a tap
    window", and this answers "should it be", from the same hold/gap numbers.
    """

    def __init__(self):
        self.started_ms: float | None = None
        self.ended_ms: float = -1e9

    def reset(self) -> None:
        self.started_ms = None
        self.ended_ms = -1e9

    def should_press(self, hint: Hint, now_ms: float) -> bool:
        if self.started_ms is not None:
            if now_ms - self.started_ms < hint.hold_ms:
                return True
            self.ended_ms = now_ms
            self.started_ms = None
            return False
        if now_ms - self.ended_ms < hint.gap_ms:
            return False
        self.started_ms = now_ms
        return True

    def in_tap(self, hint: Hint, now_ms: float) -> bool:
        """True while a tap is meant to be in progress, without advancing it."""
        return (self.started_ms is not None
                and now_ms - self.started_ms < hint.hold_ms)


def resolve_marks(hints: list[Hint], marks: dict, line) -> list[str]:
    """Turn `mark_from`/`mark_to` into `s_from`/`s_to` on the loaded line.

    Marks are stored as world positions, not arc lengths, because arc length
    belongs to whichever line is loaded and lines get re-recorded. Resolving
    late means a mark you drove to last week still points at the same corner
    after the explore stage builds a new line through it.

    Returns the names it could not find, so a typo surfaces as a warning
    rather than as a hint that silently covers the whole track.
    """
    missing = []
    for h in hints:
        for name, attr in ((h.mark_from, "s_from"), (h.mark_to, "s_to")):
            if not name:
                continue
            pos = marks.get(name)
            if pos is None:
                missing.append(name)
                continue
            try:
                setattr(h, attr, float(line.project(pos)[1]))
            except Exception:
                missing.append(name)
    return missing


def active(hints: list[Hint], speed_ms: float, s: float = 0.0,
           cp: int = 0) -> Hint | None:
    """The first hint covering this state. First, not best: hints are an
    ordered list you wrote, so an earlier entry is a deliberate override."""
    for h in hints:
        if h.covers(speed_ms, s, cp):
            return h
    return None

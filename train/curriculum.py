#!/usr/bin/env python3
"""Train the track one sector at a time, then the whole lap, then repeat.

WHY, from this map's own splits over 356 finished laps:

    sector          best   median   headroom
    start->CP1     6.90s    8.93s    +2.03s
    CP1->CP2       3.85s    6.13s    +2.28s
    CP2->CP3       6.26s   11.72s    +5.46s
    CP3->CP4       5.08s    7.96s    +2.88s
    CP4->CP5       7.91s   10.34s    +2.43s
    CP5->finish    3.84s    9.25s    +5.41s

    best actual lap 40.59s, theoretical best (sum of best sectors) 33.84s

The car has ALREADY driven a 33.84s lap - just never all in one go. Whole-lap
training spreads its attention over 55 seconds of driving, so a sector it gets
right one lap in ten is drowned by the nine it does not. Working one sector at
a time concentrates the episodes where the headroom is.

HOW A SECTOR EPISODE WORKS. The episode ends the moment the target sector's
exit gate is crossed, and pays the finish bonus for it - so a partial lap is
worth finishing rather than abandoning. The car still starts from the start
line: TM2020's respawn returns you to the LAST checkpoint passed, so after
completing CP2->CP3 a respawn puts you at CP3, not back at CP2, and the same
sector cannot be looped that way. Driving up to the sector is the price of
drilling it, and it is still cheaper than a full lap for every sector but the
last.

ADVANCING. Either 25 improvements OR 100 episodes without one, whichever comes
first. The patience half is not optional: improvements get exponentially harder
- start->CP1 is already at 6.90s against an 8.93s median - so "25 improvements"
alone would stall the curriculum on sector one indefinitely.

FORGETTING is the standing risk: drilling one sector degrades the others, which
is what the whole-lap phase at the end of each cycle is for. If full-lap times
come back worse each cycle, shorten the per-sector phase rather than lengthen
the full-lap one.
"""
from __future__ import annotations

import json
import os
import time

from stable_baselines3.common.callbacks import BaseCallback

#: label() needs to tell "use the current phase" apart from "the full-lap
#: phase", and the full-lap phase IS None - so None cannot double as the
#: default. It printed the full lap as "start->CP1" until this existed.
_UNSET = object()


class SectorCurriculum(BaseCallback):
    """Phase through sectors, then the full lap, and repeat.

    :param n_gates: number of scoring gates including the finish, so a map with
        5 checkpoints plus a finish is 6 and yields 6 sectors.
    :param improvements: new sector bests before advancing.
    :param patience: episodes with no improvement before advancing anyway.
    :param full_lap: run a whole-track phase at the end of each cycle.
    """

    STATE_FILE = "curriculum.json"

    def __init__(self, n_gates: int | None = None, improvements: int = 25,
                 patience: int = 100, max_episodes: int = 100,
                 full_lap: bool = True, verbose: int = 0):
        super().__init__(verbose)
        # None = learn it from the first info that carries cp_total. The gate
        # count is not known until the MAP loads, which is after the envs are
        # built, and it is not an env attribute - it is computed per step as
        # len(self.gates). Asking for it at construction got 0 or a guess.
        self.n_gates = int(n_gates) if n_gates else 0
        self.target_improvements = int(improvements)
        self.patience = int(patience)
        # Phase i<n_gates drills sector i (entry gate i, exit gate i+1).
        # Phase n_gates, when enabled, is the whole lap.
        self.full_lap = bool(full_lap)
        self.phases = self._build_phases()
        self.phase_i = 0
        self.cycle = 1
        self.improvements = 0
        self.since_improvement = 0
        # Hard cap on a phase, whatever else happens. Without it a sector that
        # keeps yielding small gains never advances: every improvement resets
        # the patience counter, so "100 without improvement" was never reached
        # in 175 episodes of drilling sector 0. A curriculum is meant to cycle,
        # not to perfect one sector.
        self.max_episodes = int(max_episodes)
        self.episodes = 0
        self.best: dict[int | None, float] = {}
        self.state_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", self.STATE_FILE)
        self._applied = None

    def _build_phases(self):
        return (list(range(self.n_gates)) +
                ([None] if self.full_lap else [])) or [None]

    def _learn_gates(self, info) -> bool:
        """Pick up the gate count the first time an info carries it."""
        if self.n_gates:
            return True
        n = info.get("cp_total")
        if not n:
            return False
        # cp_total counts CHECKPOINTS; the finish is signalled separately by
        # `finished`, not as another gate. So a 5-checkpoint map has 6 sectors
        # - the last being CP5 -> finish - and using cp_total directly loses
        # that sector and mislabels the one before it.
        self.n_gates = int(n) + 1
        self.phases = self._build_phases()
        self.phase_i = 0
        print(f"curriculum: {self.n_gates} gates -> "
              f"{len(self.phases)} phases: "
              + ", ".join(self.label(s) for s in self.phases), flush=True)
        self._apply()
        self._publish()
        return True

    # -- phase bookkeeping -------------------------------------------------

    @property
    def sector(self) -> int | None:
        return self.phases[self.phase_i]

    def label(self, s=_UNSET) -> str:
        s = self.sector if s is _UNSET else s
        if s is None:
            return "full lap"
        entry = "start" if s == 0 else f"CP{s}"
        exit_ = "finish" if s == self.n_gates - 1 else f"CP{s + 1}"
        return f"{entry}->{exit_}"

    def _publish(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "cycle": self.cycle,
                    "phase": self.phase_i,
                    "phases": len(self.phases),
                    "sector": self.sector,
                    "label": self.label(),
                    "improvements": self.improvements,
                    "target_improvements": self.target_improvements,
                    "since_improvement": self.since_improvement,
                    "patience": self.patience,
                    "episodes": self.episodes,
                    "max_episodes": self.max_episodes,
                    "best": {str(k): v for k, v in self.best.items()},
                    "updated": time.time(),
                }, f)
            os.replace(tmp, self.state_path)
        except (OSError, TypeError):
            pass

    def _apply(self) -> None:
        """Tell every env which sector it is driving.

        Set through the VecEnv rather than passed at construction, so a phase
        change costs nothing - no restart, no lost replay buffer.
        """
        s = self.sector
        exit_gate = None if s is None else s + 1
        if exit_gate == self._applied:
            return
        try:
            # Clear the derived stop distance FIRST: it is cached per sector,
            # and leaving the previous phase's value in place would stop the
            # new sector at the old sector's line - silently drilling the
            # wrong stretch of track.
            self.training_env.set_attr("sector_exit_s", None)
            self.training_env.set_attr("sector_exit", exit_gate)
            self._applied = exit_gate
        except Exception as ex:                            # noqa: BLE001
            print(f"  curriculum: could not set sector on the envs ({ex})",
                  flush=True)

    def _advance(self, why: str) -> None:
        done = self.label()
        best = self.best.get(self.sector)
        self.phase_i += 1
        if self.phase_i >= len(self.phases):
            self.phase_i = 0
            self.cycle += 1
            print(f"\ncurriculum: cycle {self.cycle - 1} complete, starting "
                  f"cycle {self.cycle}\n", flush=True)
        self.improvements = 0
        self.since_improvement = 0
        self.episodes = 0
        print(f"curriculum: {done} done ({why}"
              + (f", best {best:.2f}s" if best else "")
              + f") -> now {self.label()}", flush=True)
        self._apply()
        self._publish()

    # -- callback ----------------------------------------------------------

    def _on_training_start(self) -> None:
        print(f"curriculum: advance on {self.target_improvements} "
              f"improvements or {self.patience} episodes without one",
              flush=True)
        if self.n_gates:
            print(f"curriculum: starting with {self.label()}", flush=True)
            self._apply()
        else:
            print("curriculum: waiting for the map to report its gate count",
                  flush=True)
        self._publish()

    def _on_step(self) -> bool:
        # Re-apply defensively: a worker that respawned mid-phase would
        # otherwise keep the previous phase's exit gate.
        self._apply()
        # Do NOT tally sector progress during the warm-up. Those episodes are
        # the scripted driver, not the policy, and its run-to-run variance
        # would rack up "improvements" and advance sectors before the policy
        # has taken a single gradient step. The gate above still applies, so
        # the warm-up drives the current sector.
        if self.num_timesteps < getattr(self.model, "learning_starts", 0):
            return True
        # `dones` alongside `infos`: the episode cap must count EVERY episode
        # in the phase, not only the ones that completed the sector. Counting
        # completions alone meant a sector the car could not finish never
        # reached the cap and never advanced - precisely the stall the cap
        # exists to prevent.
        dones = self.locals.get("dones")
        dones = [] if dones is None else list(dones)
        infos = self.locals.get("infos") or []
        for i, info in enumerate(infos):
            if not info:
                continue
            if not self._learn_gates(info):
                continue
            if i < len(dones) and dones[i]:
                self.episodes += 1
            t = info.get("sector_time")
            if t is None:
                # A failed attempt still counts against the cap, and against
                # patience: no improvement happened.
                if i < len(dones) and dones[i]:
                    self.since_improvement += 1
                    if self.episodes >= self.max_episodes:
                        self._advance(f"{self.max_episodes} episodes")
                    elif self.since_improvement >= self.patience:
                        self._advance(f"{self.patience} without improvement")
                continue
            s = self.sector
            prev = self.best.get(s)
            if prev is None or t < prev:
                self.best[s] = float(t)
                self.improvements += 1
                self.since_improvement = 0
                print(f"  curriculum [{self.label()}] improvement "
                      f"{self.improvements}/{self.target_improvements}: "
                      f"{t:.2f}s"
                      + (f" (was {prev:.2f}s)" if prev else ""), flush=True)
                self._publish()
            else:
                self.since_improvement += 1
            if self.improvements >= self.target_improvements:
                self._advance(f"{self.target_improvements} improvements")
            elif self.since_improvement >= self.patience:
                self._advance(f"{self.patience} without improvement")
            elif self.episodes >= self.max_episodes:
                self._advance(f"{self.max_episodes} episodes")
        return True

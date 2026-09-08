"""Advance reward.par_speed up a ladder as the car earns each rung.

WHY A LADDER

par_speed sets where "break-even" sits: progress and the par charge sum to
w_progress * dt * (speed - par), so at par a step is worth nothing, above it
pays and below it costs. That makes it the dial that decides how much lap time
matters - but the RIGHT value moves as the car gets faster.

Set it at the eventual target and every lap reads deeply negative, all the
positive signal comes from the finish and checkpoint bonuses, and the useful
dynamic range of the progress term is wasted. Set it at today's pace and it is
maximally informative now but goes slack as soon as the car improves. So it
wants raising, in steps, as the car earns each one - which is a thing a human
has to remember to do, at a moment they have to notice.

THE TRIGGER IS THE RUNG ITSELF

Each rung's own implied lap time is the test: a par of P km/h over a line of
L metres means a lap of L / (P / 3.6) seconds. When the car's median finish
over the last `window` finishes reaches that, it has earned the rung - by
definition, because it is now averaging the pace that par was asking for - and
the next one is written into the config, which the env hot-reloads.

Median, not mean: one 40-second recovery lap should not hold the ladder back,
and one lucky 16-second lap should not advance it.

COSTS, so they are not a surprise

Advancing rewrites the tuning config, which bumps its generation. Every
transition already in the replay buffer was scored under the old par and
nothing relabels them, so expect the return to step down and the regression
guard to notice. That is the documented cost of any reward change and it is
why the ladder moves rarely - only when a rung is genuinely earned.
"""
from __future__ import annotations

import collections
import json
import os

from stable_baselines3.common.callbacks import BaseCallback


class ParLadder(BaseCallback):
    """Watch finish times; step reward.par_speed up when a rung is earned.

    :param rungs: par speeds in km/h, ascending. The first one at or above the
        config's current par_speed is where the ladder starts.
    :param window: how many recent FINISHES the median is taken over.
    :param cfg_path: the tuning file to rewrite. Resolved lazily from the env
        so it follows the map.
    """

    def __init__(self, rungs, window: int = 50, min_finishes: int = 0):
        super().__init__()
        self.rungs = [float(r) for r in rungs if float(r) > 0]
        self.window = max(2, int(window))
        # Do not advance off a handful of laps even if they are all quick.
        self.min_finishes = max(self.window // 2, int(min_finishes))
        self.times = collections.deque(maxlen=self.window)
        self.line_m = None
        self.cfg_path = None
        self.advances = 0

    # -- helpers ----------------------------------------------------------

    def _resolve(self) -> bool:
        """Find the line length and the config file, once the envs exist."""
        if self.line_m is not None and self.cfg_path:
            return True
        try:
            self.line_m = float(self.training_env.get_attr("line")[0].length)
            self.cfg_path = self.training_env.get_attr("cfg")[0].path
        except Exception:                                      # noqa: BLE001
            return False
        # Rungs come from the MAP's config, which is not knowable when the
        # callback is built - the map uid is only settled once the envs exist,
        # so a ladder set per-map was read from the default config and came
        # back empty every time. Now they are picked up here, on the same
        # resolve that finds the line length.
        if not self.rungs and self.cfg_path:
            try:
                with open(self.cfg_path) as f:
                    found = json.load(f).get("reward", {}).get("par_ladder") or []
                self.rungs = [float(r) for r in found if float(r) > 0]
                if self.rungs:
                    print(f"par ladder: {', '.join(str(int(r)) for r in self.rungs)}"
                          f" km/h, from {os.path.basename(self.cfg_path)}, "
                          f"advancing on the median of the last {self.window} "
                          f"finishes", flush=True)
            except (OSError, ValueError, TypeError):
                pass
        return bool(self.line_m and self.cfg_path)

    def _lap_for(self, par_kmh: float) -> float:
        """The lap time this par speed is asking for, in seconds."""
        return self.line_m / (par_kmh / 3.6)

    def _current_par_kmh(self) -> float | None:
        try:
            with open(self.cfg_path) as f:
                return float(json.load(f)["reward"]["par_speed"]) * 3.6
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_par(self, par_kmh: float) -> bool:
        try:
            with open(self.cfg_path) as f:
                data = json.load(f)
            data.setdefault("reward", {})["par_speed"] = round(par_kmh / 3.6, 4)
            tmp = self.cfg_path + ".ladder.tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.cfg_path)
            return True
        except OSError:
            return False

    # -- callback ---------------------------------------------------------

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            return True
        got = False
        for info, done in zip(infos, dones):
            if done and info and info.get("finished") and info.get("race_time"):
                self.times.append(float(info["race_time"]) / 1000.0)
                got = True
        if not got or len(self.times) < self.min_finishes:
            return True
        if not self._resolve():
            return True

        par = self._current_par_kmh()
        if par is None:
            return True
        # The next rung strictly above where we are now.
        nxt = next((r for r in self.rungs if r > par + 0.5), None)
        if nxt is None:
            return True

        srt = sorted(self.times)
        median = srt[len(srt) // 2]
        # par 0 means "no time pressure yet" - the usual way to rebuild a
        # policy on an empty buffer, because a large par makes a failed episode
        # a flat -6000 against a good one's +300 and the critic cannot fit that
        # range from nothing. There is no lap time to beat at par 0, so the
        # gate is simply "is it finishing consistently": once `window` finishes
        # exist, the first rung is earned. Without this, _lap_for(0) divides by
        # zero and takes the run down.
        if par <= 0.0:
            target = float("inf")
        else:
            target = self._lap_for(par)
            if median > target:
                return True

        if self._write_par(nxt):
            self.advances += 1
            print(f"  PAR LADDER: median of the last {len(self.times)} finishes "
                  f"is {median:.2f}s, which is the {par:.0f} km/h pace this rung "
                  f"was asking for ({target:.2f}s). Advancing par_speed "
                  f"{par:.0f} -> {nxt:.0f} km/h (next target "
                  f"{self._lap_for(nxt):.2f}s).\n"
                  f"    The buffer still holds transitions scored at the old "
                  f"par and nothing relabels them - expect a step down in "
                  f"return while they age out. Not a regression.", flush=True)
            self.times.clear()
        return True

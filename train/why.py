"""Explains what the trainer is doing and why.

The episode log tells you *what* happened - `off_line t=11.44s`. It never told
you why the policy is changing, which is the part you actually need when a run
has been going for six hours and is getting worse.

Two things make that answerable:

  * The environment now reports `info["reward_parts"]`, so we can say which
    term the episode's return actually came from. "It lost 40 points" is not
    useful; "38 of those 40 came from leaving the line, not from being slow" is.
  * SAC's own optimiser state - entropy coefficient, actor loss, critic loss -
    says whether it is still exploring, whether the value estimates are stable,
    and whether it has stopped improving.

The gloss is a small rule table over those numbers. It is deliberately blunt
and deliberately hedged: these are heuristics about a stochastic optimiser, not
measurements, and they are labelled as such rather than dressed up.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

# SAC records these every train() call. Names are SB3's, not ours.
METRICS = ("train/ent_coef", "train/actor_loss", "train/critic_loss",
           "train/ent_coef_loss")

WINDOW = 20      # episodes of history the trend rules look back over
MIN_HISTORY = 6  # below this, say nothing rather than guess from noise


class WhyLog(BaseCallback):
    def __init__(self, path: str, window: int = WINDOW):
        super().__init__()
        self.path = path
        self.window = window
        self.ep = 0
        self.returns: deque[float] = deque(maxlen=window)
        self.reasons: deque[str] = deque(maxlen=window)
        self.hist: dict[str, deque[float]] = {
            k: deque(maxlen=window) for k in METRICS}
        self.best: float | None = None

    def _on_training_start(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except OSError:
            pass

    # -- rules ------------------------------------------------------------

    @staticmethod
    def _trend(series) -> float:
        """Crude slope: mean of the recent half minus mean of the older half.
        Good enough to say "rising" or "falling" and cheap enough to run every
        episode."""
        vals = list(series)
        if len(vals) < 4:
            return 0.0
        half = len(vals) // 2
        return statistics.fmean(vals[half:]) - statistics.fmean(vals[:half])

    #: Terms that are all the SAME charge wearing different labels, and must
    #: not be reported as if they competed with each other.
    #:
    #: `step_cost` is the time an episode used; `unused_time` is the time it
    #: did not, charged on a failure so that crashing early cannot be a way of
    #: avoiding the clock. For any failed episode the two sum to exactly
    #: `time_cost * max_steps` - a constant, identical whether it died on step
    #: 100 or step 1500. Ranking them separately made the WHY log announce
    #: that "78% of this episode was the unused_time term, not lap time",
    #: which is precisely backwards: unused_time IS lap time.
    TIME_TERMS = ("step_cost", "unused_time")

    def _explain(self, parts: dict, reason: str, ret: float) -> list[str]:
        out: list[str] = []

        # Where the return actually came from. Sorted by magnitude so the
        # dominant term leads, whichever sign it has.
        if parts:
            merged = dict(parts)
            time_total = sum(merged.pop(k, 0.0) for k in self.TIME_TERMS)
            if time_total:
                merged["time"] = time_total

            ranked = sorted(merged.items(), key=lambda kv: -abs(kv[1]))
            top, val = ranked[0]
            total_abs = sum(abs(v) for v in merged.values()) or 1.0
            share = abs(val) / total_abs
            if share > 0.5 and top not in ("progress", "time"):
                out.append(
                    f"{share:.0%} of this episode's reward movement was the "
                    f"'{top}' term ({val:+.1f}) - that is what it is optimising "
                    f"against right now, not lap time")
            elif share > 0.5 and top == "time":
                out.append(
                    f"the time charge ({val:+.1f}) dominated - on a failed "
                    f"episode that is a fixed cost for the whole episode "
                    f"length, the same however early it ended, so it is not "
                    f"something the policy can avoid by crashing. What it can "
                    f"change is the progress it earned against it "
                    f"({parts.get('progress', 0.0):+.1f})")
            neg = [(k, v) for k, v in ranked if v < 0 and k != "time"]
            if neg and abs(neg[0][1]) > abs(parts.get("progress", 0.0)):
                out.append(
                    f"losses from '{neg[0][0]}' ({neg[0][1]:+.1f}) outweigh all "
                    f"the progress it earned ({parts.get('progress', 0.0):+.1f})")

        if len(self.returns) < MIN_HISTORY:
            return out

        ent = self.hist["train/ent_coef"]
        if ent:
            cur = ent[-1]
            if cur > 0.5:
                out.append(
                    f"entropy coefficient is {cur:.2f} - still exploring hard on "
                    f"purpose, so a lot of these actions are deliberately random")
            if self._trend(ent) > 0.02:
                out.append(
                    "entropy is rising, not falling: the critic has not found a "
                    "clearly better action yet, so SAC is widening its search")

        crit = self.hist["train/critic_loss"]
        if len(crit) >= MIN_HISTORY:
            med = statistics.median(crit)
            if med > 0 and crit[-1] > 3 * med:
                out.append(
                    f"critic loss spiked to {crit[-1]:.1f} against a median of "
                    f"{med:.1f} - the reward it just got surprised it, which "
                    f"usually means a config change or a new part of the track")

        actor = self.hist["train/actor_loss"]
        ret_trend = self._trend(self.returns)
        if len(actor) >= MIN_HISTORY:
            if self._trend(actor) < 0 and abs(ret_trend) < 1.0:
                out.append(
                    "actor loss is falling while episode return is flat - it is "
                    "getting more confident about something that is not scoring "
                    "any better. That is what a local optimum looks like")

        if ret_trend > 2.0:
            out.append(f"episode return trending up ({ret_trend:+.1f} over the "
                       f"last {len(self.returns)}) - this is working")
        elif ret_trend < -2.0:
            out.append(f"episode return trending down ({ret_trend:+.1f} over the "
                       f"last {len(self.returns)}) - it is getting worse")

        # A single ending repeating is the most actionable thing there is - but
        # only when it is a FAILURE. Finishing every lap is the goal, and
        # calling that "stuck on one failure mode, tune that term" told you to
        # go and break the thing that was working.
        if len(self.reasons) >= MIN_HISTORY and reason:
            same = sum(1 for r in self.reasons if r == reason)
            if same >= len(self.reasons) * 0.8:
                if reason.upper() == "FINISH":
                    out.append(
                        f"{same} of the last {len(self.reasons)} episodes "
                        f"FINISHED - the route is solved, so what is left is "
                        f"lap time. Watch the split times rather than the "
                        f"return from here.")
                else:
                    out.append(
                        f"{same} of the last {len(self.reasons)} episodes ended "
                        f"'{reason}' - it is stuck on one failure mode, and "
                        f"tuning that term is likely to do more than more "
                        f"training will")
        return out

    # -- callback ---------------------------------------------------------

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []),
                              self.locals.get("dones", [])):
            if not done:
                continue
            self.ep += 1
            parts = info.get("reward_parts") or {}
            ret = float(sum(parts.values()))
            reason = "FINISH" if info.get("finished") else info.get("reason", "?")

            metrics = {}
            for k in METRICS:
                v = self.model.logger.name_to_value.get(k)
                if v is not None:
                    metrics[k] = float(v)
                    self.hist[k].append(float(v))

            notes = self._explain(parts, reason, ret)
            self.returns.append(ret)
            self.reasons.append(reason)

            record = {
                "episode": self.ep,
                "step": int(self.num_timesteps),
                "reason": reason,
                "return": round(ret, 2),
                "race_time": info.get("race_time"),
                "cp": info.get("cp"),
                "cp_total": info.get("cp_total"),
                "distance": round(info.get("distance") or 0.0, 1),
                # Net progress cannot tell "never moved" from "went far and
                # came back", and it reads negative when an episode began
                # part-way along the line. These two say which.
                "start_distance": round(info.get("start_distance") or 0.0, 1),
                "max_distance": round(info.get("max_distance") or 0.0, 1),
                "reward_parts": {k: round(v, 3) for k, v in parts.items()},
                "metrics": {k: round(v, 5) for k, v in metrics.items()},
                "why": notes,
            }
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError:
                pass

            if parts:
                breakdown = "  ".join(f"{k}={v:+.1f}" for k, v in
                                      sorted(parts.items(), key=lambda kv: -abs(kv[1])))
                print(f"      reward: {breakdown}", flush=True)
            for note in notes:
                print(f"      why: {note}", flush=True)
        return True

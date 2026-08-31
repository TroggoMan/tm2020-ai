"""When it gets worse instead of better, notice - and say why.

Reinforcement learning goes backwards sometimes, and on a run measured in days
you will not be watching when it happens. By the time you look, the model on
disk is the regressed one and the good policy is gone.

Three causes, in the order they actually bite on this project:

**1. You changed the reward while it was running.**
This is the one nobody expects, because the hot-reload is so convenient. The
replay buffer holds ~300k transitions and every one of them carries the reward
it was scored with *at the time*. Change `w_gear` mid-run and the critic is now
learning from a buffer where the same state-action pair is labelled two
different ways. It does not blend; it thrashes. The buffer needs about as many
new transitions as it has old ones before the change fully takes - at 40Hz
that is over an hour of driving per 150k stale transitions.

**2. The policy walked off a cliff.**
SAC's actor can move into a region the critic has overvalued, and the data it
then collects confirms the mistake because it is the only data being collected.
This is what the rollback is for.

**3. The entropy coefficient climbed.**
`ent_coef="auto"` raises exploration when the policy gets too deterministic.
That looks exactly like a regression and is not one - it is temporary and it
will come back. Check `train/ent_coef` in the WHY log before rolling anything
back.

This callback watches for (2), tells you about (1), and refuses to confuse
either with (3).
"""
from __future__ import annotations

import os
import time
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class RegressionGuard(BaseCallback):
    """Watch a rolling mean of episode return; warn or roll back on a fall.

    :param window: episodes in the rolling mean. Too short and normal
        exploration noise looks like a regression; 20 is about the floor.
    :param drop: fractional fall below the best rolling mean that counts as a
        regression. 0.25 means "a quarter of the return has gone".
    :param patience: consecutive windows below that line before acting. A
        single bad window is a bad run, not a trend.
    :param rollback_to: model path to restore from, or None to only warn.
        Restoring loads the *policy* and leaves the replay buffer alone, which
        is what you want: the experience is still valid, it was the weights
        that went wrong.
    """

    def __init__(self, window: int = 20, drop: float = 0.25,
                 patience: int = 3, rollback_to: str | None = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.window = int(window)
        self.drop = float(drop)
        self.patience = int(patience)
        self.rollback_to = rollback_to
        self.returns: deque[float] = deque(maxlen=self.window)
        self.best_mean: float | None = None
        self.bad = 0
        self.rollbacks = 0
        self.last_gen = None
        self.gen_changed_at = None
        self.ep = 0

    # -- reward-config changes -------------------------------------------

    def _check_generation(self, info: dict) -> None:
        gen = info.get("cfg_gen")
        if gen is None or gen == self.last_gen:
            return
        if self.last_gen is not None:
            self.gen_changed_at = self.num_timesteps
            size = getattr(self.model, "replay_buffer", None)
            n = size.size() if size is not None else 0
            print(f"  reward config changed (gen {self.last_gen} -> {gen}). "
                  f"{n:,} transitions in the buffer were scored under the OLD "
                  f"reward and are not relabelled - expect a wobble while they "
                  f"age out.", flush=True)
        self.last_gen = gen

    # -- entropy, so a deliberate exploration burst is not misread --------

    def _ent_coef(self) -> float | None:
        try:
            return float(self.model.logger.name_to_value.get("train/ent_coef"))
        except (TypeError, ValueError):
            return None

    def _rollback(self) -> None:
        path = self.rollback_to
        if not path or not os.path.isfile(path + ".zip"):
            print("  no snapshot to roll back to - leaving it alone",
                  flush=True)
            return
        try:
            # set_parameters swaps the weights in place and leaves the replay
            # buffer untouched. Reloading the whole model would throw away the
            # experience, which is the expensive part and is still valid.
            self.model.set_parameters(path, exact_match=False)
            self.rollbacks += 1
            self.returns.clear()
            self.bad = 0
            print(f"  ROLLED BACK to {os.path.basename(path)}.zip "
                  f"(rollback #{self.rollbacks}). The replay buffer is kept.",
                  flush=True)
        except Exception as ex:
            print(f"  rollback failed: {ex}", flush=True)

    def _on_step(self) -> bool:
        # `or []` is wrong on `dones`: SB3 hands it over as a numpy array, and
        # truthiness on an array of more than one element raises rather than
        # falling back. It only shows up with more than one env, which is
        # exactly when this callback matters.
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        infos = [] if infos is None else infos
        dones = [] if dones is None else dones
        for info, done in zip(infos, dones):
            if not info:
                continue
            self._check_generation(info)
            if not done:
                continue
            self.ep += 1
            parts = info.get("reward_parts") or {}
            if not parts:
                continue
            self.returns.append(float(sum(parts.values())))
            if len(self.returns) < self.window:
                continue

            mean = sum(self.returns) / len(self.returns)
            if self.best_mean is None or mean > self.best_mean:
                self.best_mean = mean
                self.bad = 0
                continue

            floor = self.best_mean - abs(self.best_mean) * self.drop
            if mean >= floor:
                self.bad = 0
                continue

            self.bad += 1
            ent = self._ent_coef()
            note = ""
            if ent is not None and ent > 0.1:
                note = (f" - but ent_coef is {ent:.3f}, so this may be a "
                        f"deliberate exploration burst rather than a "
                        f"regression")
            print(f"  REGRESSION: {self.window}-episode mean {mean:.1f} vs "
                  f"best {self.best_mean:.1f} ({self.bad}/{self.patience})"
                  f"{note}", flush=True)
            if self.gen_changed_at is not None and \
                    self.num_timesteps - self.gen_changed_at < 100_000:
                print("    the reward config changed recently - the buffer is "
                      "still mixed. Wait it out before rolling back.",
                      flush=True)
            if self.bad >= self.patience:
                if self.rollback_to:
                    self._rollback()
                else:
                    print("    (--auto-rollback would restore the best "
                          "snapshot here; restoring by hand from the panel "
                          "does the same thing)", flush=True)
                    self.bad = 0
        return True

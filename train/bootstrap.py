"""SAC with a driven warm-up instead of a random one.

Stock SB3 fills the first `learning_starts` transitions by sampling uniformly
from the action space. That is the right default for a simulator you can run a
million steps through overnight. Here every step is a real second of a real
game, so spending the first 2000 of them - eighty seconds of wall clock per
instance, and the eighty seconds that shape the critic's first opinion of the
world - watching the car snake randomly and stop is expensive.

`BootstrapSAC` replaces those actions with `train.scripted.drive`, mixed with
some genuine randomness so the buffer still contains variety. Nothing else
changes: no imitation loss, no behavioural cloning, no constraint on the
policy. SAC is off-policy, so transitions from a scripted driver are exactly
as usable as transitions from a random one - they are just better.

After warm-up this is stock SAC, byte for byte.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC

from stable_baselines3.common.callbacks import BaseCallback

from env.hints import parse as parse_hints
from train.ghost_driver import GhostDriver, load as load_ghost
from train.scripted import drive_batch


class BootstrapSAC(SAC):
    """SAC whose warm-up drives rather than flails.

    :param bootstrap: "pursuit", "straight", or "off" for stock behaviour.
    :param bootstrap_random: fraction of warm-up steps that stay uniform
        random. Some is important: a buffer containing only one trajectory
        teaches the critic nothing about the alternatives, and SAC's entropy
        term needs to see that other actions exist.
    :param bootstrap_jitter: steering noise added to the scripted action, so
        even the scripted steps are not all identical.
    """

    def __init__(self, *args, bootstrap: str = "pursuit",
                 bootstrap_random: float = 0.25,
                 bootstrap_jitter: float = 0.15,
                 hints=None, control_hz: float = 40.0,
                 ghost_file: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrap = bootstrap
        # "ghost": replay a recorded lap's inputs during the warm-up, so the
        # buffer fills with a good driver's transitions scored by the env's own
        # reward. See train/ghost_driver.py for why this beats injecting rows
        # into the buffer directly.
        self.ghost = None
        if ghost_file:
            try:
                # Resampled onto the control clock: the cursor advances one
                # row per step, so a 20 Hz ghost against a 40 Hz control rate
                # would otherwise replay the lap at double speed.
                acts = load_ghost(ghost_file, hz=control_hz)
                if len(acts):
                    self.ghost = GhostDriver(acts, hz=control_hz)
                    print(f"ghost warm-up: {len(acts)} inputs from "
                          f"{ghost_file} at {control_hz:g}Hz "
                          f"({len(acts)/control_hz:.1f}s)", flush=True)
                else:
                    print(f"ghost file {ghost_file} had no usable inputs - "
                          f"falling back to '{bootstrap}'", flush=True)
            except Exception as ex:                            # noqa: BLE001
                print(f"could not load ghost {ghost_file}: {ex} - "
                      f"falling back to '{bootstrap}'", flush=True)
        # Per-seat race clock, filled from info in _store_transition.
        self._ghost_clock: dict = {}
        self.bootstrap_random = float(bootstrap_random)
        self.bootstrap_jitter = float(bootstrap_jitter)
        # Hints are performed during the warm-up so the buffer contains real
        # tapped transitions from the first minute. The clock is counted in
        # steps rather than read from time.perf_counter(): a pulse train has
        # to line up with the *control* period, and the wall clock drifts off
        # it whenever a gradient step overruns.
        self.hints = parse_hints(hints)
        self.step_ms = 1000.0 / max(float(control_hz), 1e-6)
        # Per-env overrides published by the environments through `info`. The
        # env is the only thing that knows arc length and checkpoint count -
        # and under SubprocVecEnv it is in a different process - so section
        # hints arrive here rather than being recomputed.
        self._hint_actions: list = []
        self._bootstrap_used = 0
        # Control steps, not transitions. _bootstrap_used counts
        # transitions (n_envs per step), and driving the pulse clock
        # off that would run it n_envs times too fast - a 120ms tap
        # would come out 40ms long on three instances.
        self._bootstrap_steps = 0

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward,
                          dones, infos):
        """Record each seat's race clock, so the ghost can be indexed on it.

        This is the only hook that sees `infos`, and the env already puts
        `race_time` there. Captured here and applied in _sample_action, which
        is one control step of latency - 25ms at 40Hz, far less than the drift
        a step-counted cursor accumulated over a lap.
        """
        if self.ghost is not None:
            for i, inf in enumerate(infos or ()):
                # Record it even when it is None: None means "clock not
                # running", which sync() needs to see so it can hold the ghost
                # at the start line instead of letting it run on.
                self._ghost_clock[i] = (
                    (inf or {}).get("race_time") if isinstance(inf, dict) else None)
        return super()._store_transition(replay_buffer, buffer_action, new_obs,
                                         reward, dones, infos)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        warming = self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup)
        if not (warming and (self.bootstrap != "off" or self.ghost is not None)
                and self._last_obs is not None):
            return super()._sample_action(learning_starts, action_noise, n_envs)

        # The ghost drives whatever seats it still has samples for; the
        # scripted driver picks up the rest. These used to be exclusive - a
        # ghost REPLACED pursuit entirely - which meant a 28.9s world record
        # left every seat parked in neutral for the remainder of a long
        # episode, and made "use the WR" and "use pursuit" a choice you had to
        # get right up front. Seeding with the record and continuing under
        # pursuit is strictly better than either alone.
        ghost_unscaled = ghost_live = None
        if self.ghost is not None:
            for i, rt in self._ghost_clock.items():
                self.ghost.sync(i, rt)
            ghost_unscaled, ghost_live = self.ghost.batch(n_envs)
            ghost_unscaled = ghost_unscaled.astype(np.float64)
            if not ghost_live.any():
                if self.bootstrap == "off":
                    if isinstance(self.action_space, spaces.Box):
                        scaled = self.policy.scale_action(ghost_unscaled)
                        return self.policy.unscale_action(scaled), scaled
                    return ghost_unscaled, ghost_unscaled
                ghost_unscaled = ghost_live = None

        if ghost_live is not None and self.bootstrap == "off":
            if isinstance(self.action_space, spaces.Box):
                scaled = self.policy.scale_action(ghost_unscaled)
                return self.policy.unscale_action(scaled), scaled
            return ghost_unscaled, ghost_unscaled

        try:
            unscaled = drive_batch(self._last_obs, self.bootstrap,
                                   hints=self.hints,
                                   step_ms=self._bootstrap_steps * self.step_ms)
        except Exception as ex:
            # A broken bootstrap must never take a run down; fall back to the
            # behaviour that has always worked and say so once.
            if self.bootstrap != "off":
                print(f"bootstrap disabled: {ex}", flush=True)
                self.bootstrap = "off"
            return super()._sample_action(learning_starts, action_noise, n_envs)

        if unscaled.shape[0] != n_envs:
            unscaled = np.repeat(unscaled[:1], n_envs, axis=0)

        # Seats the ghost is still driving take its inputs verbatim. Done
        # BEFORE the hint overlay and the random/jitter pass below, so a ghost
        # lap goes into the buffer as the world record actually drove it -
        # jittering a known-good reference would only blur it.
        if ghost_live is not None and ghost_live.any():
            unscaled[ghost_live] = ghost_unscaled[ghost_live]

        # Overlay whatever the environments asked for, axis by axis. A hint
        # that only sets `gas` leaves the driver's steering alone, which is
        # what makes "hold flat through sector 2" a one-line hint rather than
        # a whole hand-written driver.
        for i, override in enumerate(self._hint_actions[:n_envs]):
            if not override:
                continue
            for axis in range(3):
                v = override[axis] if axis < len(override) else None
                if v is not None:
                    unscaled[i, axis] = float(np.clip(v, -1.0, 1.0))

        rng = np.random.random(n_envs)
        for i in range(n_envs):
            if ghost_live is not None and ghost_live[i]:
                continue
            if rng[i] < self.bootstrap_random:
                unscaled[i] = self.action_space.sample()
            elif self.bootstrap_jitter:
                unscaled[i, 0] = np.clip(
                    unscaled[i, 0]
                    + np.random.randn() * self.bootstrap_jitter, -1.0, 1.0)
        self._bootstrap_used += n_envs
        self._bootstrap_steps += 1

        # Same tail as the base class: normalise, apply any action noise, and
        # keep the scaled copy for the buffer.
        if isinstance(self.action_space, spaces.Box):
            scaled = self.policy.scale_action(unscaled)
            if action_noise is not None:
                scaled = np.clip(scaled + action_noise(), -1, 1)
            return self.policy.unscale_action(scaled), scaled
        return unscaled, unscaled


class HintRelay(BaseCallback):
    """Carries each env's requested warm-up action back to the sampler.

    SB3's rollout loop samples an action and *then* steps, so the override
    used on step k is the one the environment published on step k-1. At 40Hz
    that is 25ms of latency on a brake tap - well inside the pulse - and it is
    the honest price of the env and the driver living in different processes.

    Why relay rather than let the env apply the override itself: the replay
    buffer records the action SAC chose. An env that quietly drove something
    else would fill the buffer with transitions that never happened as
    described, and the critic would learn from them.
    """

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        self.model._hint_actions = [
            (i or {}).get("hint_action") for i in infos]
        # Rewind that seat's ghost when its episode ends.
        #
        # GhostDriver's cursor only advances, so without this a 1204-input
        # recording is spent after one episode and every remaining warm-up step
        # is neutral - a 200k warm-up would be 1204 useful transitions and
        # 198,796 of a parked car. Each seat rewinds on its OWN done, because
        # seats finish at different moments and a shared reset would hand one
        # of them the start of the lap while it is halfway round.
        ghost = getattr(self.model, "ghost", None)
        if ghost is not None:
            dones = self.locals.get("dones")
            if dones is not None:
                for i, d in enumerate(dones):
                    if d:
                        ghost.reset_env(i)
                        # Drop the finished episode's clock too. Left behind,
                        # the next episode's first _sample_action would sync
                        # the cursor to the END of the old lap before the new
                        # episode has reported a clock of its own.
                        self.model._ghost_clock.pop(i, None)
        return True

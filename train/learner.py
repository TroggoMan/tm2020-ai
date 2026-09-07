"""The learner, off the control loop.

THE PROBLEM THIS SOLVES

SB3 runs its gradient update on the thread that steps the environment, so
every update happens *inside* the 25ms control period:

    every tick:  obs = env.step(a)         # blocks on the game
                 buffer.add(...)
                 model.train(gradient_steps=N)   # must finish before the next tick

Half the period is the budget and one gradient step measures ~3-4ms, so two
or three fit - *however many cars are driving*. The intended ratio has always
been two updates per transition (`auto_gradient_steps` asks for
`2 * instances`), and the clock refuses it the moment there is more than one
car:

    cars    transitions/s    updates/s (inline)   updates per transition
      1            40                80                 2.00   <- as intended
      3           120                80                 0.67
     12           480                80                 0.17

Updates-per-transition is what off-policy sample efficiency actually tracks,
so a twelve-car fleet was collecting twelve times the data and learning from
each piece of it twelve times *less*. Batch scaling (`auto_batch_size`) makes
each of those two updates better informed, which is worth having, but it
cannot put back update frequency.

WHY A THREAD AND NOT A PROCESS

The obvious design is a separate learner process with a shared-memory replay
buffer and published weights. It is also a lot of machinery: IPC for ~1KB per
transition, a weight-publishing channel, policy staleness to reason about, and
SB3's `learn()` loop - with every callback, checkpoint, regression guard and
handover hanging off it - rewritten around it.

None of that is necessary here, because the rollout thread is not doing any
Python work to be blocked by. It spends its whole period in `time.sleep`
(DummyVecEnv) or a blocking pipe read (SubprocVecEnv), both of which release
the GIL, and torch releases the GIL around its ATen ops and runs backward on
its own thread. Measured on this 4080 with a 40Hz loop alongside:

    arrangement                          grad/s   control period (med/p95/max)
    inline, 2 steps per tick               81      25.0 / 25.0 / 25.0 ms
    background thread                     431      25.0 / 25.0 / 25.1 ms
    background thread, nothing else       419      -

The thread runs at *full speed* - the same rate it manages with no control
loop at all - and perturbs the period by 0.1ms at the 95th percentile, with
zero overruns. Identical at 1, 4 and 12 envs, and on both blocking styles.
5.3x the updates for about 200 lines, no IPC, no staleness, and every existing
callback still works. The process split buys nothing on top of that until the
GPU itself is saturated, which is a different problem with a different fix
(see "WHEN THE GPU IS THE LIMIT" below).

WHAT PACES IT

Not "as fast as possible". Free-running on a single car gives ~10 updates per
transition, which is far past the point of diminishing returns and into the
territory where SAC needs specific tricks (REDQ/DroQ) not to destabilise. The
target is the ratio the project already intended - `--utd`, default 2.0 - so
decoupling *restores* the designed behaviour at every fleet size instead of
introducing new behaviour at one of them. One car is unchanged in intent and
in effect.

WHEN THE GPU IS THE LIMIT

The learner sustains ~250-300 real SAC updates/s. Twelve cars at 40Hz produce
480 transitions/s, so UTD 2.0 there would need 960 updates/s and is simply not
available. `capacity_report()` says so at startup rather than letting the run
quietly under-train, and batch size is the compensation: past the point where
updates/s runs out, a wider batch is nearly free (kernel-launch bound, not
compute) and each update at least sees more of the buffer.

THREAD SAFETY, precisely

Three things are shared, and each is handled:

  * the replay buffer - `add()` (rollout) against `sample()` (learner). Locked.
    The window is sub-millisecond on both sides, so the contention is ~0.05ms
    per tick, but an unlocked torn read would mix one transition's observation
    with another's action and there is no way to notice that afterwards.
  * the logger - `train()` records keys the main thread's `_dump_logs()`
    iterates. A key appearing mid-iteration raises "dictionary changed size";
    it can only happen on the very first update, which is exactly the sort of
    once-per-run failure that gets blamed on something else. `_dump_logs` is
    wrapped in the same lock the learner holds around `train()`.
  * the network weights - the rollout reads them while the learner writes.
    Deliberately NOT locked: this is Hogwild, it is what every distributed RL
    setup does, and the worst case is one forward pass seeing a layer from
    before an update and a layer from after. Locking here would put the full
    gradient step back inside the control period, which is the entire thing
    being fixed.

`set_training_mode` is toggled by both threads. With the default `[256, 256]`
MlpPolicy there is no dropout and no batch norm, so the flag has no
behavioural effect; if a policy with either is ever used, this needs revisiting.

SAVING

`model.save()` and `save_replay_buffer()` must not run while the learner is
mid-update, or the checkpoint is a mix of pre- and post-update tensors and the
buffer pickle can catch a half-written transition. `set_parameters()` (the
regression guard's rollback) is worse - restoring weights underneath a running
optimiser step. Use `paused(model)` around all of them; it is a no-op when
there is no learner, so callers do not need to know.
"""
from __future__ import annotations

import contextlib
import math
import threading
import time
import traceback

# Sustained gradient steps per second, measured 2026-09-07 on this 4080 with a
# REAL SB3 SAC - 119-dim observation, [256,256] net, twin critics, targets and
# the entropy update - running on the learner thread with a 40Hz rollout in the
# foreground. That last part matters: these are the rates in the arrangement
# actually shipped, not a solo microbenchmark (which reads ~3% higher).
#
# An earlier table in this project recorded 3.99ms/step at batch 256 and the
# handoff quoted 3.4ms. Both were optimistic - a bare net timing, missing the
# log-prob, the entropy coefficient and the target polyak. The honest figure is
# 4.6ms, so plan against ~215/s and not ~290/s.
#
# The step is kernel-LAUNCH bound rather than compute bound below ~4096, which
# is why 16x the batch costs only 18% more time - that is what makes a wider
# batch the right consolation when updates/s runs out.
STEPS_PER_S = {256: 217.5, 512: 215.5, 1024: 201.7, 2048: 185.0, 4096: 183.2}


def updates_per_s(batch: int) -> float:
    """Interpolate the measured sustained update rate at `batch`."""
    sizes = sorted(STEPS_PER_S)
    if batch <= sizes[0]:
        return STEPS_PER_S[sizes[0]]
    for lo, hi in zip(sizes, sizes[1:]):
        if batch <= hi:
            f = (batch - lo) / (hi - lo)
            return STEPS_PER_S[lo] + f * (STEPS_PER_S[hi] - STEPS_PER_S[lo])
    # Past the measured range the cost is genuinely compute-bound, so it
    # degrades with the batch rather than staying flat.
    return STEPS_PER_S[sizes[-1]] * sizes[-1] / batch


def auto_batch(cars: int, control_hz: float, utd: float,
               base: int = 256, cap: int = 4096) -> int:
    """Batch size for a decoupled run: `base`, widened only if starved.

    Once the learner is off the control loop, the clock no longer caps the
    update count, so the reason `auto_batch_size` existed - buying back
    throughput the clock refused - is gone at small fleet sizes and the batch
    goes back to 256 exactly.

    It comes back at LARGE fleet sizes for a different reason. Twelve cars at
    40Hz want 960 updates/s at UTD 2 and the GPU sustains ~250, so those
    updates are genuinely unaffordable. Widening the batch by the shortfall
    means each update it CAN afford sees proportionally more of the buffer,
    which is nearly free below ~4096 (kernel-launch bound). It is a
    consolation, not a substitute: more data per update is not the same thing
    as more updates.
    """
    want = cars * control_hz * utd
    have = updates_per_s(base)
    if want <= have:
        return base
    # Round UP to a multiple of the base. Up rather than nearest because the
    # cost curve is nearly flat to ~4096, so over-widening is close to free
    # while under-widening leaves a real shortfall uncompensated; and a
    # multiple of 256 rather than an odd 653, which is no better than 512 and
    # makes two runs harder to compare.
    mult = max(1, math.ceil(want / have))
    return int(min(cap, base * mult))


def capacity_report(cars: int, control_hz: float, utd: float,
                    batch: int) -> str:
    """What this fleet size wants from the learner, and whether it can have it.

    Printed at startup so an under-trained run announces itself, rather than
    being discovered later as "it learns worse with more instances".
    """
    want = cars * control_hz * utd
    have = updates_per_s(batch)
    inline = _inline_utd(cars, control_hz)
    lines = [f"learner: DECOUPLED (own thread), target {utd:g} "
             f"updates/transition, batch {batch}",
             f"  {cars} car(s) at {control_hz:g}Hz produce "
             f"{cars * control_hz:g} transitions/s, so UTD {utd:g} wants "
             f"{want:.0f} updates/s",
             f"  measured sustained rate at batch {batch}: ~{have:.0f}/s"]
    if want <= have:
        lines.append(f"  -> fits, with {(have - want) / have:.0%} headroom. "
                     f"Inline would have managed {inline:.2f}.")
    else:
        reach = have / (cars * control_hz)
        lines.append(
            f"  -> UPDATE-BOUND: the real ratio will be about {reach:.2f}, "
            f"not {utd:g} - the GPU cannot issue updates faster. That is "
            f"still {reach / max(inline, 1e-9):.1f}x the inline loop's "
            f"{inline:.2f}.")
        if batch > 256:
            lines.append(
                f"  -> batch widened {batch // 256}x to compensate: each of "
                f"the updates it CAN afford sees {batch} samples instead of "
                f"256, which is nearly free below ~4096. More data per update "
                f"is not the same as more updates, but it is what is on offer.")
    return "\n".join(lines)


def _inline_utd(cars: int, control_hz: float) -> float:
    """What the old inline arrangement achieved, for the comparison above."""
    fits = max(1, int((1000.0 / control_hz * 0.5) / 6.0))
    return min(2 * cars, fits) / max(1, cars)


class DecoupledLearner:
    """Runs `model.train()` continuously on its own thread.

    :param model: the SB3 off-policy model. Its `gradient_steps` is set to 0
        so `learn()` stops training inline - this object owns training now.
    :param batch_size: samples per update.
    :param utd: target updates per transition collected.
    :param max_backlog_s: how much of a debt to carry when the GPU cannot keep
        up. Without a cap the shortfall accumulates for the whole run and the
        learner would sprint through hours of owed updates the moment the
        fleet shrank, hammering a buffer whose data had moved on.
    """

    def __init__(self, model, batch_size: int, utd: float = 2.0,
                 max_backlog_s: float = 2.0):
        self.model = model
        self.batch_size = int(batch_size)
        self.utd = float(utd)
        self.max_backlog = max(1.0, utd * max_backlog_s * 40.0)

        self.updates = 0
        self.credits = 0.0
        self.failed = None                  # the traceback, if the thread died
        self._last_ts = None
        # num_timesteps when THIS learner started, so the achieved ratio is
        # measured over this process. `num_timesteps` carries across a
        # --resume, so counting from learning_starts would divide this run's
        # updates by the whole lineage's transitions and report half the truth.
        self._ts0 = None
        self._t_start = None
        self._busy_s = 0.0

        self._stop = threading.Event()
        self._thread = None
        self._patched = False
        self._orig = {}
        # Held around a whole gradient step. Also taken by paused() and by
        # _dump_logs, both of which must not overlap an update.
        self._train_lock = threading.RLock()
        # Held around buffer add/sample only - sub-millisecond on both sides.
        self._buf_lock = threading.Lock()

    # -- wiring -----------------------------------------------------------

    def _patch(self) -> None:
        """Make the buffer, the logger and the actor safe to share.

        Patched onto the instances rather than subclassed: the replay buffer is
        built inside SB3's constructor and `_dump_logs` and the actor's methods
        are bound methods, so there is nothing to subclass without
        reimplementing all three.

        Instance patching does collide with one thing, and it is not obvious
        until it bites: `model.save()` cloudpickles the model's `__dict__` and
        `save_replay_buffer()` pickles the buffer, so a closure holding a
        `threading.Lock` makes both raise

            TypeError: cannot pickle '_thread.lock' object

        - i.e. the first checkpoint of the run dies, five minutes in. Hence
        `_unpatch()`, and hence `paused()` calling it: every save is already
        bracketed by `paused()`, so stripping the wrappers there fixes saving
        without any call site knowing about it.
        """
        if self._patched:
            return
        model, buf = self.model, self.model.replay_buffer
        blk, tlk = self._buf_lock, self._train_lock
        self._orig = {"add": buf.__dict__.get("add"),
                      "sample": buf.__dict__.get("sample"),
                      "dump": model.__dict__.get("_dump_logs")}

        raw_add, raw_sample, raw_dump = buf.add, buf.sample, model._dump_logs

        def add(*a, **kw):
            with blk:
                return raw_add(*a, **kw)

        def sample(*a, **kw):
            with blk:
                return raw_sample(*a, **kw)

        def dump(*a, **kw):
            with tlk:
                return raw_dump(*a, **kw)

        buf.add, buf.sample, model._dump_logs = add, sample, dump
        self._thread_local_action_dist()
        # So paused() and the episode log can find us without being passed in.
        model.learner = self
        self._patched = True

    def _unpatch(self) -> None:
        """Put the instances back the way SB3 built them.

        Only ever called with both locks held, so nothing is mid-call. The
        actor's methods go back to the shared-distribution versions, which is
        correct precisely because the learner is stalled while they are.
        """
        if not self._patched:
            return
        model, buf, actor = (self.model, self.model.replay_buffer,
                             self.model.policy.actor)
        for obj, name in ((buf, "add"), (buf, "sample"),
                          (model, "_dump_logs"), (model, "learner"),
                          (actor, "forward"), (actor, "action_log_prob")):
            obj.__dict__.pop(name, None)
        self._patched = False

    def _thread_local_action_dist(self) -> None:
        """Stop the two threads corrupting SB3's ONE shared distribution.

        This is not a theoretical race. SAC's actor holds a single
        `action_dist` object and both code paths mutate it in place:

            forward()          -> action_dist.actions_from_params(...)
            action_log_prob()  -> action_dist.log_prob_from_params(...)

        and `SquashedDiagGaussianDistribution` stores `self.distribution` (a
        Normal built from that call's mean and std) and `self.gaussian_actions`
        on itself. So the rollout's `predict()` re-points the distribution at
        its own batch while the learner is between sampling an action and
        taking its log-prob.

        Measured before this fix, with a 40Hz rollout and a learner thread:

            ValueError: Value is not broadcastable with batch_shape+event_shape:
              torch.Size([8192, 3]) vs torch.Size([4, 3])

        - the learner's 8192-row batch meeting a distribution the rollout had
        just rebuilt for its 4 envs. It killed the learner thread within
        seconds, on every batch size tried. A silent version of the same race
        (matching shapes, wrong numbers) would have been far worse: log-probs
        computed against another thread's distribution, entropy tuned on them,
        and nothing to see but a policy that would not improve.

        The fix is one distribution per thread. They are stateless between
        calls apart from the two attributes above, so a private copy per
        thread is exactly equivalent and costs one object.
        """
        actor = self.model.policy.actor
        if getattr(self.model, "use_sde", False):
            # gSDE's distribution carries an exploration matrix that train()
            # resamples deliberately; per-thread copies would diverge from it.
            print("  WARNING: use_sde=True - the actor's distribution cannot "
                  "be made thread-local safely. Run with --no-decouple.",
                  flush=True)
            return
        import copy as _copy
        proto = actor.action_dist
        local = threading.local()

        def dist():
            d = getattr(local, "d", None)
            if d is None:
                d = local.d = _copy.deepcopy(proto)
            return d

        def forward(obs, deterministic: bool = False):
            mean, log_std, kw = actor.get_action_dist_params(obs)
            return dist().actions_from_params(mean, log_std,
                                              deterministic=deterministic, **kw)

        def action_log_prob(obs):
            mean, log_std, kw = actor.get_action_dist_params(obs)
            return dist().log_prob_from_params(mean, log_std, **kw)

        # Instance attributes, so nn.Module.__call__ picks them up: its
        # _call_impl reads `self.forward`, and a plain function is not a
        # Parameter/Module/Tensor so nn.Module.__setattr__ passes it through.
        actor.forward = forward
        actor.action_log_prob = action_log_prob

    # -- the thread -------------------------------------------------------

    def _ready(self) -> bool:
        """The same gate SB3's inline loop uses, and for the same reason.

        Two traps here, both found by measurement:

        * `replay_buffer.size()` counts SLOTS, not transitions - each slot
          holds one row per env. Gating on `size() > batch_size` therefore
          asked for `batch_size * n_envs` transitions, and a twelve-car run at
          batch 1280 sat at zero updates for 15,360 steps while looking
          perfectly healthy. `sample()` draws (slot, env) pairs, so the
          transitions available are `size() * n_envs`; the honest gate is
          `learning_starts`, which is already denominated in transitions, and
          that is exactly what SB3 gates on inline.
        * `train()` needs `model._logger`, which only exists once `learn()`
          has called `_setup_learn()`. The learner thread is started first, so
          without this check a run with `learning_starts=0` would die on
          AttributeError before the rollout had begun.
        """
        return (getattr(self.model, "_logger", None) is not None
                and self.model.num_timesteps > self.model.learning_starts
                and self.model.replay_buffer.size() > 0)

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._ready():
                    time.sleep(0.05)
                    continue
                ts = self.model.num_timesteps
                if self._last_ts is None:
                    self._last_ts = self._ts0 = ts
                    self._t_start = time.perf_counter()
                self.credits += self.utd * max(0, ts - self._last_ts)
                self._last_ts = ts
                if self.credits < 1.0:
                    # Ahead of the data. Sleep well under a control period so
                    # the next transition is picked up promptly.
                    time.sleep(0.002)
                    continue
                self.credits = min(self.credits, self.max_backlog)
                t0 = time.perf_counter()
                with self._train_lock:
                    self.model.train(gradient_steps=1,
                                     batch_size=self.batch_size)
                self._busy_s += time.perf_counter() - t0
                self.updates += 1
                self.credits -= 1.0
        except BaseException:                                  # noqa: BLE001
            # A learner thread that dies silently is the worst failure mode
            # available here: the cars keep driving, the episode log keeps
            # printing, checkpoints keep being written, and nothing is being
            # learned. Record it and shout - stats() surfaces it every episode.
            self.failed = traceback.format_exc()
            print("\n*** LEARNER THREAD DIED - training has STOPPED while the "
                  "cars keep driving ***\n" + self.failed, flush=True)

    def start(self) -> "DecoupledLearner":
        self._patch()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="sac-learner")
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stop training and leave the model exactly as SB3 built it.

        Unpatching on the way out matters: after this the trainer saves, and a
        model still carrying the lock-holding wrappers cannot be pickled.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        with self._train_lock, self._buf_lock:
            self._unpatch()

    # -- reporting --------------------------------------------------------

    def stats(self) -> dict:
        """Achieved ratio and how hard the GPU is working for it."""
        ts = self.model.num_timesteps
        # Transitions collected since THIS learner started, which is what its
        # updates are per.
        collected = max(1, ts - (self._ts0 if self._ts0 is not None else ts))
        elapsed = (time.perf_counter() - self._t_start) if self._t_start else 0.0
        return {"updates": self.updates,
                "utd": self.updates / collected,
                "updates_s": self.updates / elapsed if elapsed > 0 else 0.0,
                "busy": self._busy_s / elapsed if elapsed > 0 else 0.0,
                "batch": self.batch_size,
                "alive": bool(self._thread and self._thread.is_alive()),
                "failed": self.failed is not None}

    def summary(self) -> str:
        s = self.stats()
        if s["failed"]:
            return "learner: DEAD - nothing is being learned"
        return (f"learner: {s['updates']:,} updates, "
                f"{s['updates_s']:.0f}/s, {s['utd']:.2f} per transition, "
                f"GPU busy {s['busy']:.0%}")


@contextlib.contextmanager
def paused(model):
    """Hold the learner still - for save, load, or a parameter swap.

    A no-op when training is inline, so every call site can use it
    unconditionally and nothing has to branch on whether decoupling is on.
    """
    learner = getattr(model, "learner", None)
    if learner is None:
        yield
        return
    with learner._train_lock, learner._buf_lock:
        # Strip the wrappers for the duration. Both locks are held, so the
        # learner is stalled and nothing is mid-call - and without this a
        # model.save() or save_replay_buffer() inside the block would raise
        # "cannot pickle '_thread.lock' object" on the closures.
        learner._unpatch()
        try:
            yield
        finally:
            if not learner._stop.is_set():
                learner._patch()

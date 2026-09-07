# Session handoff — 2026-09-07 (later)

Previous handoffs: `SESSION_HANDOFF-2026-08-31.md` (the steering-saw finding,
still relevant to driving quality). The earlier part of today was telemetry —
RAM reader, RL Connect, School mode — and is written up in **RAM.md**.

---

## DONE: the learner is decoupled

`train/learner.py`. It was the priority in the previous handoff and it is
built, measured and documented in README's "Running several games into one
model". On by default; `--no-decouple` restores the old behaviour.

**It is a thread, not a process.** The previous handoff scoped actor processes
feeding a shared buffer with published weights. That turned out to be
unnecessary: the rollout thread does no Python work worth blocking — it sits
in `time.sleep` or a blocking pipe read, both releasing the GIL — so a
background thread already gets the whole period. Measured with a real SB3 SAC
on the 4080:

| arrangement | updates/s | control period med/p95/max |
|---|---|---|
| inline, 2 steps/tick | 80 | 25.0 / 25.0 / 25.0 ms |
| decoupled thread | 217 | 25.0 / 25.0 / 25.1 ms |
| thread, no control loop | 224 | — |

97% of full rate, 0.1 ms of jitter, zero overruns, identical at 1/4/12 envs.
The process split would add an IPC replay buffer, a weight channel and policy
staleness for no measured gain. Do not build it.

**Paced to `--utd`, default 2.0**, not free-running. 2.0 is what the trainer
always intended (`auto_gradient_steps` asks for `2 * instances`) and what one
instance already achieved, so a single-instance run is unchanged at 2.00.
Free-running would put one car at ~5, which needs REDQ/DroQ tricks not to
destabilise.

### Three real bugs this turned up, all fixed — do not reintroduce

1. **SB3's SAC actor shares ONE mutable `action_dist` between `predict()` and
   `train()`.** `SquashedDiagGaussianDistribution` stores `self.distribution`
   and `self.gaussian_actions` on itself, and both threads rebuild them. It
   killed the learner thread within seconds on every batch size:

       ValueError: Value is not broadcastable with batch_shape+event_shape:
         torch.Size([8192, 3]) vs torch.Size([4, 3])

   Fixed with a thread-local distribution
   (`DecoupledLearner._thread_local_action_dist`). A *silent* version of this
   race — matching shapes, wrong numbers — would have been far worse: log-probs
   taken against another thread's distribution and nothing to see but a policy
   that would not improve.
2. **The lock-holding wrappers made the model unpicklable** —
   `TypeError: cannot pickle '_thread.lock' object` on the first checkpoint,
   five minutes into a run. `paused()` now strips the patches for the duration
   of the block (it holds both locks, so nothing is mid-call) and `stop()`
   strips them permanently.
3. **`replay_buffer.size()` counts SLOTS, not transitions.** One slot holds one
   row per env, so gating readiness on `size() > batch_size` demanded
   `batch_size * n_envs` transitions: twelve cars at batch 1280 sat at **zero
   updates** for 15,360 steps looking perfectly healthy. Gate on
   `learning_starts`, which is already in transitions, exactly as SB3 does.

Also fixed in passing: `--resume` never applied the run's computed
`batch_size` to the loaded model (harmless decoupled, a real bug inline).

### Everything that saves is bracketed

`paused(model)` — a no-op when training is inline — wraps every `model.save`,
`save_replay_buffer` and `set_parameters` (`EpisodeLog`'s checkpoint / `_best` /
`_archive` / `_promote`, the trainer's exit paths, and `RegressionGuard`'s
rollback). Without it a checkpoint mixes pre- and post-update tensors, and a
rollback restores weights underneath a running optimiser step.

### Verified

Real trainer path, real callbacks, DummyVecEnv and SubprocVecEnv:

    1 car  batch  256 : rollout 40.0 steps/s, UTD 2.00, GPU 45%
    4 cars batch  512 : rollout 40.0 steps/s/car, UTD 1.31 (ceiling 1.35), GPU 97%
    12 cars batch 1280: rollout 40.0 steps/s/car, UTD 0.41 (ceiling 0.41), GPU 96%

Save, `--resume` (weights + buffer) and rollback all exercised. The startup
banner reports which regime the run is in.

### Still unverified

That any of this **trains better**. It is more gradient steps per transition,
which is the thing sample efficiency tracks, but only a real run answers
whether the policy improves faster per wall-clock hour. That needs a live
A/B — same map, same seed, `--no-decouple` against default — judged on
checkpoint-reach rate per hour, not on reward.

---

## THE NEW CEILING: ~217 updates/s, and it is the GPU

Decoupling moved the bottleneck rather than removing it. Total gradient work is
now fixed at what one 4080 can issue, so **more cars buy data diversity and
wall-clock coverage, not more learning**:

| cars | transitions/s | UTD inline | UTD decoupled | gain |
|---|---|---|---|---|
| 1 | 40 | 2.00 | 2.00 | — |
| 4 | 160 | 0.50 | 1.31 | 2.6× |
| 8 | 320 | 0.25 | 0.68 | 2.7× |
| 12 | 480 | 0.17 | 0.41 | 2.5× |

`auto_batch()` widens the batch past that point (nearly free below ~4096 —
kernel-launch bound, not compute bound), so each affordable update sees more of
the buffer. It is a consolation, not a substitute.

**What would actually raise the ceiling, in order of value:**

1. **torch.compile — 20%**, previously rejected because it prefixes
   `state_dict` keys with `_orig_mod.` and orphans every checkpoint. Now worth
   revisiting *with* a prefix-stripping save hook plus a migration, because the
   update rate is the binding constraint rather than a nicety.
2. **CUDA graphs** for the gradient step. The step is kernel-launch bound —
   that is the whole reason batch 4096 costs only 18% more than 256 — so
   capturing it is aimed directly at the actual cost.
3. **A second GPU** for the learner. The games need VRAM, not compute; the
   learner needs the opposite.
4. Lowering `control_hz`, which lowers transitions/s and raises UTD for free —
   but see README's "Frame rate is the control rate": below ~45 fps the game
   discards policy decisions, so this trades one kind of fidelity for another.

Do **not** reach for more learner threads: they contend for the same device and
the same GIL.

---

## State of the machines

Carried forward from this morning's handoff, which this file replaced.

* `:99` dev/paid instance — game DOWN, plumbing (pads, broker, panel) up.
  Graphics set to lowest by `tools/gfx-low.py`; `--restore` undoes it.
* `:100` tmai01 — free Starter account (TAS01-TROGGOBOT), game running, Steam
  logged in. **Plugins were installed by hand and are hash-invalid, and
  `MLHook` is missing** — reinstall through the in-game Plugin Manager before
  drawing any conclusion from this instance.
* `:101` tmai02 — Steam at the sign-in screen, no Wine prefix yet. After the
  first game launch, run `tools/sync-maps.sh 2` and
  `python3 tools/gfx-low.py 2`.
* `tools/instances.sh` lists them all with VNC ports (localhost-only; tunnel).

## Unchanged from this morning's handoff

Telemetry (RAM + RL Connect), the School-mode blocker, the per-wheel ring
buffer, and the state of `:99` / `:100` / `:101` are all as written in the
earlier section of RAM.md and the `project_tm2020_ram_telemetry` memory. The
School-mode F3 toggle is still the biggest single unlock for the fleet, and it
is now also the thing standing between this learner and having twelve cars to
feed it.

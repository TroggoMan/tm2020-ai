# Session handoff — 2026-08-30 (late) / 2026-08-31

Read `README.md`, `TRAINING.md` and the `project_tm2020_ai_driver` memory entry
first. This is the short version.

## THE FINDING: the car was sawing at the wheel, not steering

The user said the track is easy for a human and the cars "are just not turning
the correct way". Measured on the continuous action space:

* the steering command **changed sign between 41.3% of consecutive control
  steps**;
* it swung by **more than lock-to-lock on 22%** of them;
* i.e. the wheel was being sawed **8.3 times a second**.

A car steered like that barely turns at all, whatever the policy intends - the
inputs average out. It also explains why the game rarely showed full lock
(`in_steer` reached it 0.5% of the time): TM2020's steering ramp was always
chasing a command that reversed before it arrived.

**Fix (the user's idea):** quantise the steering to notches and ramp toward
them - which is exactly how a keyboard drives TM2020, and keyboard players are
competitive, so it is known to be enough control.

* `action.steer_levels: 10` - notches every 10%, "ten steering keys per side".
* `action.steer_rate: 8.0` - slew limit, full lock in 125 ms. **This was
  already implemented in `tm_env` and simply switched off (`0.0`).**
* The cap is **per stage, not a schedule**: `<uid>.explore.json` sets
  `steer_rate: 8.0`, `<uid>.race.json` sets `0.0`. A real keyboard can flick
  lock-to-lock in one frame and some TM technique needs that, so the racer must
  not be capped; the explorer only needs to find the route.

**Correction, and it was in the project memory wrongly too:** the racer does
NOT start from fresh weights. `train_sac.py` execs the race stage with
`--init-from <explore model>`, which `set_parameters`-copies actor and critics
(the replay buffer is deliberately left empty). So the racer **inherits the
explorer's driving**, including a policy trained while its steering output was
a target the env ramped toward - and then runs with that ramp gone. Expect
oversteer until it adapts. Accepted because the mismatch is in the forgiving
direction, the buffer refills entirely under the new regime, and the handover
already changes reward + line + offsets anyway. If a handover ever visibly
collapses into oversteer, ramp `steer_rate` off over the early race stage
instead of dropping it at once.

First measurement after turning it on (at `steer_levels: 2`): sign flips
**41.3% -> 0.8%**, sawing **8.3/s -> 0.2/s**, swings >1.0 units **22% -> 0%**.
Whether it makes the car *drive* was still unproven at handoff.

## THE OTHER BIG ONE: the replay buffer was deleting every good episode

**It drives the whole track now** - 11 completed laps in one overnight run,
best **41.3 s**, improving 61.7 s -> 41.3 s as it learned. Then it slowly got
worse, and this is why.

`buffer_size` was **300_000**. That run reached **2,476,912 steps**, so the
buffer held only the **last 12%** of it. All eleven finishes happened between
steps 758k and 1,939k; the buffer window started at step 2,177k. So:

```
  0 of 11 finishes were still in the replay buffer
```

The policy was training on a buffer containing **no example of ever finishing
the track**, and it is self-reinforcing: it drifts down, the buffer refills
with the worse episodes, and those become the only thing left to learn from.
Mean distance peaked at 415 m while the finishes were in memory and fell to
272 m once they had aged out.

Fixed: `--buffer-size`, now defaulting to **2M** (~1.9 GB; a transition is
~1 KB with the 119-dim obs stored twice). 2M covers a full overnight run.

**Three explanations that were checked and did NOT fit** - do not re-run them:

* *entropy bonus overtaking the reward* - no: `ent_coef` was FALLING
  (0.169 -> 0.137) while performance also fell, and pooled over all runs the
  0.12-0.18 band is the best-performing one, not the worst;
* *the episode-cap feedback loop* - real, still unfixed, but not this: all 600
  recent episodes ended `stuck`, never `timeout`, so the cap never truncated
  anything, and `unused_time` barely moved (-39.8 -> -38.4);
* *a distorted reward term* - no: every term scaled down together, which is a
  policy getting worse rather than a reward getting warped.

## Still unfixed: the episode-cap ratchet cuts BOTH ways

`_recent_cp` is a 12-episode deque **per seat**; the cap is
`90 + 30 * max(_recent_cp)` clamped to 240 s. Twelve bad episodes on one seat
drop its cap from 240 s to 90 s, and three of the eleven finishes took 90 s,
93 s and 118 s - so a seat in that state cannot complete a slow lap even if it
drives one. The comment says the fall-back is deliberate (one fluke CP2 should
not grant forever) and that intent is right; the bug is that it can trap a
policy that WAS finishing. Suggested fix: floor the frontier at
`all_time_best_cp - 1`, so a run that has reached CP5 never drops below CP4's
210 s, while a fluke CP2 still only buys CP1's cap.

## Unexplained: two seats overrun 30x more than the other two

```
  i0 slip=416   i2 slip=488      i1 slip=16   i3 slip=13
```

Same seats, same ~30x ratio, reproduced on a completely fresh process - so it
is structural, not accumulated state. It is only ~0.6% of steps and both
finishes came from i2, so it is not crippling, but those two seats' transitions
are timed differently from the other two and nothing downstream knows that.

## The crawling scare was a false alarm

An earlier section of this file said the cars were crawling (median 12.7 km/h,
never above 142 km/h) and to start there. **That was a snapshot of an early,
untrained policy, not a structural problem** - the same run later drove
complete 41 s laps. Measure speed on a trained policy before concluding
anything from it. Kept here because the reasoning is instructive, not because
the conclusion stands.

## Older note, superseded: the cars are crawling

Everything above is about *where* the car stops and *how* it steers. The thing
that was walked past all session is that it is barely moving at all. Measured
over 5441 steps of recent traces:

```
  speed        mean 24.6 km/h   median 12.7 km/h   max 142.1 km/h
  above 100 km/h:  1.6% of the time
  above 200 km/h:  0.0%
  above 300 km/h:  0.0%
  throttle on:    60.0% of steps
```

Median speed is **walking pace**, and the car never once exceeds 142 km/h on a
campaign track a human is over 200 km/h for most of. This reframes the earlier
analysis: the cars are not crashing at speed into the hairpin, they are
trundling and getting caught on scenery, and "a corner it takes too fast" only
ever described the handful of fastest episodes rather than the typical one.

**The question is why it will not hold the throttle down.** Concrete, checkable
candidates, none of them yet tested:

* `reward.w_gas` is 0.01 - trivial next to a `step_cost` that is charged
  whatever the car does, so there is little to gain from accelerating;
* `w_both_pedals` / `brake_threshold 0.7` may still be interrupting throttle -
  measure the throttle duty cycle directly, not the reward term;
* **`stuck` is `speed 0.5 for 3 s`, so a car creeping at 2 km/h is never
  "stuck"** and can burn a whole episode without the detector firing;
* `w_accel` 0.05 pays for acceleration, which a braking policy loses - check it
  is not simply cheaper to sit still.

## Honest scoreboard for 2026-08-30/31

Three real bugs found and fixed, each verified by measurement (line height,
projection folds, wheel sawing). **None of them moved the mean distance.**
Policy-phase mean is 78 m against the pre-fix baseline's 85 m - the same. An
early "45% improvement" reported at 26 episodes did NOT survive to 161; do not
trust a per-run mean before ~150 policy episodes, the early sample flatters it.

The entropy collapse recurs in every configuration tried: `ent_coef` runs
0.86 -> ~0.02 within roughly 150 episodes, and every run's best stretch is the
one where it is still around 0.05-0.1. Raising `target_entropy` is NOT the
answer (measured worse - see below). This remains unexplained.

## Dead end, recorded so it is not re-run

The `INPUT FIDELITY: the game is applying N% of the steering we send` warning
looks like a smoking gun and **is not**. Checked properly: the applied-steering
ramp reaches full lock in ~11 ms (measured live off `in_steer`), there is no
speed-based steering cap (max applied stays ~0.98 at every speed band), and the
window IS focused on `:99`. The warning fired 68 times across ~2500 episodes.
The apparent "75% applied" came from comparing commanded-steering-during-the-
scripted-warm-up against applied-steering-during-policy-driving, which is not
like-for-like. **NB `trace` records the COMMANDED steer, not `in_steer`** - do
not read that column as what the game did.

## The headline

**The brake was disconnected for ~7 hours.** `action.brake_threshold` was
`1.2`, and `_decode()` clips the network output to `[-1, 1]` before testing
`a[2] > threshold` — so the condition was unsatisfiable and the brake could
never fire. Measured from telemetry: **0.00% braking in every trace after
15:00**, against 35-44% earlier in the day.

This was set deliberately (to stop cars sitting on the brake) without it being
obvious that 1.2 is unreachable. Two consequences, and both match the symptoms
exactly:

* the car could not slow down; and
* **TM2020's brake is reverse when stationary**, so a car wedged against a
  wall had no way to back off. Only 7% of wedges ever recovered, and 100% of
  episodes ended `stuck`.

Now `0.0` in the affected configs, with two guards so it cannot recur silently:
`_apply_config` prints a loud warning if a binary pedal's threshold is >= 1.0,
and the panel clamps those two fields and explains why.

## Also fixed: the traced line floated above the road

`env/roadtrace.py` took a block's deck to be its TOP cell (`y + sy - 1`). True
for ordinary road, but a **checkpoint block is a road piece with a gantry arch
over it**, so its box is 2-3 cells tall and the tarmac is the BOTTOM one. Plus
`world_centre` used the cell centre, another +2 m.

Measured against 16 854 real driven positions, the line sat a **median 5.2 m
(up to 9.9 m) above the road**, and every gate read 9-14 m off the line it was
meant to sit on. Since 18 of the policy's inputs are "where is the road ahead,
in my frame", it was being aimed at a road hovering above the real one.

Rebuilt as: spawn -> **the midpoint of the face each pair of consecutive blocks
shares** -> finish, with deck heights anchored to blocks that know their own
level (gantries, single-layer pieces) and interpolated across ramps, and with
**checkpoint heights taken from the game's own landmark**. Results:

| | before | after |
|---|---|---|
| vertical error vs driven laps | +5.20 m | **-0.02 m** |
| gate off-line distance | 9.4 / 9.5 / 9.7 / 13.9 / 9.6 m | **0.7 / 1.3 / 2.8 / 5.4 / 1.0 m** |
| coverage / CP order | 98% / [0,1,3,4,2] | unchanged |
| gate accounting vs the game | 207/207 | **207/207** |

## Also fixed: 32 m progress jumps at hairpins

`Centerline.project()` is a global nearest-point search, so where the road
folds back on itself the arc position could snap to the other branch: **32 m of
progress for 0.8 m of movement**, at five places on the lap. New
`Centerline.project_near(pos, last_index, moved_m)` searches near the previous
index with a window that follows how far the car actually moved. Max
single-step arc gain **32.0 m -> 6.0 m** (6 m per 50 ms step = 432 km/h, i.e.
the remaining worst case is physically achievable) with **total progress paid
unchanged**. The env keeps `_line_idx`, reset to None at each episode start.

## Things measured and deliberately NOT changed

* **Gate geometry is exact.** Replayed against 207 traces, `Gates.crossed` with
  `cp_half_width 18 / cp_height 10` reproduces the game's own checkpoint count
  **207/207, no false positives, zero lag**. The previous handoff's suspicion
  about gate 0 is closed — it was never the problem.
* **Gate planes from the line tangent** were tried (the chord estimate is
  22-40 deg off the road at every gate) and are **worse**: 206/207, one false
  positive, up to 8 steps of lag. Reverted; a note in `mapdata.py` says why.
* **`target_entropy` -3 -> -1** to stop the entropy collapse: measured **worse**
  over 70 episodes against an otherwise identical baseline (31% of the way to
  CP1 vs 55%, zero checkpoints vs five). Reverted to `auto`. The collapse is
  real (ent_coef 0.86 -> 0.02 by ep 100) but more randomness is not the cure.
* **`--gradient-steps 8`** (what `auto` picked) overran 5.4% of control steps;
  4 overran 1.3%; 1 overran 0.04%. `auto_gradient_steps` is now calibrated from
  those overruns rather than a benchmark.
* **LIDAR is healthy** — all 16 beams vary over a real lap, 0.1% saturated.
* **Steering is not inverted.** Verified end to end: the game's `vis.Left` is
  exactly `cross(up, dir)` (dot = 1.000 over 1600 live samples), and the
  pursuit driver run through the real observation pipeline on 12 783 real
  positions steers the correct way **100.0%** of the time. Cars *can* turn
  correctly; the learned policy simply had not got there.
* **`stuck` 12 s -> 3 s.** Only 7% of >=3 s wedges ever regained >5 m, median
  gain 0 — so ending sooner costs nothing and 50% of wall-clock (and 52% of all
  recorded steps) was a motionless car.

## What is running

```
train/train_sac.py --stage explore --seats 4 --handover 20 --handover-patience 25
  --then-race --bootstrap pursuit --learning-starts 20000
  --buffer-size 2000000 --init-from models/sac_steerstage_best.zip
  --steps 5000000 --race-steps 10000000 --control-hz 20 --gradient-steps 1
  --name sac_bigbuf --promote-to models/driver.zip
```

**Warm-started from the 41.3 s model**, not fresh - `--init-from` copies actor
and critics and leaves the buffer empty on purpose. That restore works: first
finish at **episode 534** against **episode 1921** for the run that started
from random weights.

Trend at ~660 episodes: mean arc 205 -> 376 m, `cp>=4` 10% -> 20%, `ent_coef`
settled at 0.093 (the band the previous run performed best in, and not drifting
up the way it did before). One finish so far, 51.79 s.

**The buffer fix is NOT yet proven.** At 300k steps against a 2M buffer nothing
has aged out - the buffer holds 100% of the run, exactly as the old one did at
this stage. It only proves itself if the curve keeps rising past where the old
run peaked (mean 415 m, around episodes 2300-3700) instead of decaying.

`--learning-starts 20000` is the current experiment: the hand-written pursuit
driver provably steers correctly, but it only used to run for 2000 transitions
(~25 s across four seats) before handing over to a random-init network. 20 000
is ~4 minutes of competent line-following seeded into the buffer. **Watch
whether the warm-up episodes themselves reach CP1** — if pursuit can, the line
and controls are good and it is purely a learning problem; if pursuit cannot,
the corner genuinely needs braking and `env/hints.py` is the tool for it.

`logs/trainer.pid` is written by hand at launch — **nothing writes it
automatically**, and a stale one nearly caused two trainers to drive the same
four pads at once. Check `pgrep -af train_sac.py` before trusting it.

## Where the cars actually fail

One dominant wedge point: 12 of 45 recent episodes ended inside a 20 m box at
**(1180, 18.0, 806)**, arc ~215-224, coming south down the hairpin and running
out of road ~42 m short of CP1 in x. Gate 0 is at arc 232 of 1836.

## Scaffolding on Summer 2026-06 that must be REMOVED (2026-08-31)

Three per-track reward aids exist only to get the policy across the 194m
unbarriered platform run between CP4 and CP5 (line 1356m -> 1549m). None of
them belongs in a general driver, and all three live in
`configs/mIAmhaktmdIeoA9nVTQarm8Ck91.explore.json`, never in
`env/config.py` DEFAULTS:

* `reward.w_platform: 1.0` - doubles pay-per-metre on unbarriered sections.
  Side effect: halves the relative weight of `step_cost` there, so the car
  crosses slowly and wanders. Observed doing exactly that.
* `markers: [1400m +150, 1460m +200, 1520m +250]` - one-off bonuses inside the
  gap, to shorten how far the critic must carry CP5's value backwards.
* `surfaces._non_road: -0.5` (and Grass/Water/DirtRoad at -0.5) - keep this
  one; it is the general "no progress on sand/water" rule and is sized to
  cancel progress (~+1/step), not to dominate.

REMOVAL TRIGGER: once CP5 is reached reliably (say >30% of CP4 episodes),
drop `w_platform` to 0 first - it is the one distorting driving style - then
clear the markers with `tools/set_marker.py --clear`. Re-check that CP5
conversion holds without them; if it collapses, the policy learned the
scaffold rather than the crossing.

WHY THE SCAFFOLD WAS NEEDED: with the earlier -5/wheel off-road penalty,
stopping at the platform entrance cost -20 while attempting and falling cost
up to -940, so giving up was 47x cheaper and the policy learned it within an
hour. The penalty was resized before the scaffold was added.

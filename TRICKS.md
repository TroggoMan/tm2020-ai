# Teaching it tricks

A reward that says "go fast" will find the racing line. It will not find a
speed slide, a wallride, or a reverse bank, because those are not on the way to
anything — they are isolated islands in action space that a random policy
cannot walk to. Every trick below is taught with the same four-rung ladder,
and the interesting part is that **most of the work is rungs 1 and 2, not the
reward**.

## The ladder

**1. Can it see the thing?**
If the number that distinguishes the trick from a mistake is not in the
observation, no reward will teach it — the policy is being asked to condition
on something it cannot perceive, and the best it can do is memorise where on
the track the reward happened. Check `OBS_GROUPS` in `env/tm_env.py` first,
every time.

**2. Will exploration ever produce it?**
SAC's warm-up samples uniformly on three axes at 40Hz. "A 120ms brake pulse at
430km/h followed by a specific steering hold" has a probability of essentially
zero, and no amount of training time changes that. This is what
`env/hints.py` exists for: a hint performs the entry for the first few
thousand transitions, so the buffer contains at least *some* examples of the
trick to compare against. Without this rung the reward is a term that never
fires.

**3. Make it pay — as a band, not a threshold.**
A trick has a right amount. Too little slip is understeer, too much is a spin;
both are slower. So the reward is a ramp that peaks across the correct band and
falls to nothing outside it, never a step function — a step gives the policy no
direction to move in, so it cannot climb toward the band from outside it. See
`env/speedslide.py` for the worked example.

**4. Take the shaping away.**
The shaped term is a subsidy for a behaviour that has not yet paid for itself.
Once the trick appears reliably, turn `w` down and let lap time decide. If the
trick was actually slower, it should die — and you want to find that out.

> Keep every `w` small. The progress term pays ~1.0 per metre; a trick bonus of
> 0.05 per step is a nudge, and 5.0 per step is a policy that slides on the
> spot and never finishes. If a shaped term is more than a few percent of
> episode return in `logs/why.jsonl`, it is not shaping any more, it is the
> objective.

---

## Where each trick sits today

| trick | can it see it? | can it reach it? | band known? |
|---|---|---|---|
| straight-line accel | yes | yes | n/a — **do this first** |
| speed slide (road) | yes — `side`, `speed`, `surface` | yes — brake-tap hint | **yes, from SDHelper** |
| grass / dirt slide | yes | yes | yes, from SDHelper |
| reverse slide | yes | needs a reverse hint | yes, from SDHelper |
| wallride | partly — `up` tilt, `surface` | needs a wall and a hint | no, needs measuring |
| turbo chaining | yes — `turbo`, `fx_ahead` | yes | n/a, it is just "be on it" |
| air control / landings | partly — `flying`, `airbrake` | yes | no |
| bugslide / neo slide | **no** | — | — |

The one hard "no" is per-wheel ground contact. The plugin streams a single
`ground` boolean for the whole car, and every bug-slide variant is defined by
*which* wheels are off the ground. That is a small plugin change
(`CSceneVehicleVisState` exposes it per wheel) plus an observation width
change, which retires every existing model — worth doing deliberately, not as
a side effect of chasing one trick.

---

## Trick 1 — drive in a straight line

This is where to start, and there was a real reason it was not working: the
reference line was the problem, not the policy.

`lines/ExploreMode.json` is a recording of the **straight** road map, and it
measures 1534m across a 1483m chord, wandering up to **11m sideways** through
836° of cumulative heading change. Two consequences, both of which punish
driving straight:

* the six lookahead points the policy is steered by swing sideways by up to
  11m, so the thing it is being pointed at is a slalom; and
* `soft_offset` is 8m, so a car driving perfectly straight down the middle of
  the road is *outside the line's own wobble* for much of the map, and is
  charged the off-line penalty for it.

The policy was not misbehaving. It was doing what the line asked.

`lines/straight-road.json` is that line with the wobble taken out
(`tools/straighten_line.py --axis auto`): dead straight, 0m of wasted length,
and stamped with the map uid so it cannot be loaded against the wrong track.

    ./tm2020-sacAI
    # then, in the panel or directly:
    .venv/bin/python train/train_sac.py \
        --line lines/straight-road.json --name straight --bootstrap straight

Expect: the warm-up drives straight at full throttle immediately, `progress`
dominates the WHY log, and episodes end in `FINISH` rather than `off_line`.
If it still snakes with a straight line loaded, the cause is downstream of the
line and worth chasing — but check `w_weave` first, which is `0.0` by default,
meaning **nothing currently penalises snaking at all**.

## Trick 2 — the speed slide

The numbers are not ours. `env/speedslide.py` carries SDHelper's own band
table, because SDHelper is what human SD players steer by.

Worth being clear about what SDHelper does, because it looks like it reads
skidmarks and it does not. It reads sideways speed, forward speed, and the
material under the front-left wheel, and then **swaps the skidmark texture on
disk** so the marks the game draws come out green / yellow / orange / blue.
The colour is its output. Both of its inputs have been in our telemetry all
along, which is why this is learnable without seeing a single skidmark.

Road, forward, above 400km/h, in km/h of sideways speed:

    < 7      nothing useful
    7 – 13   orange
    13 – 19  yellow
    19 – 22  GREEN — this is the target
    22 – 28  yellow
    28 – 34  orange
    > 34     spinning

Grass and dirt slide from 200km/h and want far less sideways speed (green at
7–10 and 9–12 respectively); reversing shifts every band up by about 10.

**Rung 2 is the whole problem here.** At 450km/h in a straight line the car
has zero sideways speed and there is no gradient toward 19km/h of it — the
band is unreachable by drifting toward it. It is reached by a brake tap. So:

```json
"hints": [
  {"name": "sd-entry", "speed_from_kmh": 400, "speed_to_kmh": 600,
   "hold_ms": 120, "gap_ms": 700, "w": 0.05}
],
"speedslide": { "w": 0.15, "w_blue": 0.0 },
"enabled": { "speedslide": true }
```

`hold_ms` is the tap length you want tried — that is the knob you asked for.
The warm-up performs it, and the reward pays a little for tapping inside the
window afterwards so the behaviour has a reason to persist. Both read the
same numbers from the same file, so the tap and the payment cannot disagree.

Start with `w_blue` at 0. It charges the car for being fast and *not* sliding,
which is a second force pulling in the same direction as `w`, and two terms
pulling at once is how you get a policy that spins on the spot.

Use the `SD Trainer` map — you already built it, and its recorded line is dead
straight (0.0m wasted, 2° of total turning), which is exactly what you want
under a slide experiment.

## Trick 3 — grass and dirt slides

No new machinery: same table, different row, and the floor drops to 200km/h.
The only change is that the hint's speed window should move down with it
(`"speed_from_kmh": 200, "speed_to_kmh": 400`) and the map needs grass or dirt
on it. This is the cheapest second trick precisely because rungs 1–3 are
already built.

## Trick 4 — reverse slide

The band table already has the backward rows and SDHelper applies no speed
floor at all going backwards. What is missing is rung 2: nothing in the
warm-up ever reverses. A hint with `"control": "brake"` at low forward speed
gets the car turned around; after that the same slide reward applies with the
sign of `speed` selecting the backward row automatically.

## Trick 5 — wallride

Rung 1 is partly there and rung 3 is not. `world_up_local` in the observation
already tells the policy how far the car is tilted out of the world's
horizontal — a wallride is that value going hard sideways while `ground` stays
true — so it can *see* the state. What does not exist is a band: nobody has
measured what tilt and what speed make a wallride carry rather than scrub.

The order of work is therefore backwards from the others: build the map, drive
some wallrides by hand, record the traces, and read the tilt/speed/side-speed
numbers out of them. Only then write the band. Guessing this one is how you
get a policy that drives into walls confidently.

## Trick 6 — turbo chaining

The easiest of the lot and the one most likely to already work. `fx_ahead`
carries distance and lateral offset to the next boost, `turbo` carries the
current level and remaining time, and `w_turbo_use` already exists as a small
bonus for holding throttle while a boost is live. There is no band and no
entry trick — the whole skill is "line up with the next one", which is
ordinary racing that ordinary progress reward already pays for. Turn
`turbo_bonus` on, use the `Booster left right` map, and mostly watch.

---

## Running the curriculum

You already built the right maps. Use them in this order, one trick at a time,
and keep a separate model per stage rather than one model that has seen
everything:

    Straight Line - Road       accel, gears, not snaking
    SD Trainer                 the speed slide, on a straight road
    Wide left, mid right       carrying a slide through a corner
    Booster left right         boost pickup and chaining

Getting a map onto every instance is a file copy plus one command — no menus,
no clicking, and a Starter Access account plays local maps perfectly well:

    tools/maps.py push "path/to/Some Map.Map.Gbx" -n 3
    tools/maps.py play "Downloaded/Some Map.Map.Gbx" -n 3

Trackmania Exchange works the same way — `tools/maps.py tmx <id>` downloads
the `.Map.Gbx` over HTTP and pushes it. The in-game TMX browser is the thing
that would need clicking; the file does not.

After each stage, read `logs/why.jsonl`. The question is never "did the return
go up" — it is "which term did the return come from". A trick that is being
paid for by its own shaping term and nothing else has not been learned, it has
been bought.

---

## Air control, and why the policy CANNOT currently learn it

User's technique knowledge, 2026-08-31:

> A brake tap in the air stops **pitch**, but not **yaw**. Yaw is controlled by
> countersteering.

Both of those are **rate** control: damping pitch means reacting to how fast
the nose is rotating, countersteering yaw means reacting to yaw rate. And the
observation contains **no angular velocity of any kind**.

What it does have: `up` (3 dims - world up in the car's frame, so attitude) and
`grnd` (1 dim - on the ground or not). Attitude is not rate. There is no frame
stacking and no RNN, so a single observation physically cannot express "I am
pitching nose-up at N degrees per second". In the air `slip` and `side_speed`
are meaningless too - no wheel contact - so the ground slip signal does not
substitute.

**Conclusion: no amount of reward shaping will teach these two techniques with
the current observation.** A reward for "landing flat" would be paying for an
outcome the policy has no input to control. This is the same class of problem
as the neoslide note above (a one-frame release timing trick that single-frame
observations cannot capture).

### Two fields the telemetry ALREADY sends and the observation throws away

```
  flying        airtime duration
  ground_dist   height above ground - i.e. how long until you land
```

`ground_dist` is the single most useful one: timing a pitch correction is
entirely a question of how long you have left. Both are per-car fields already
arriving every frame; adding them is two observation dims, not new plumbing.

Also dropped: `steer_angle`, `damper` (per-wheel suspension travel),
`skidding`, `top_contact`, `wear`, `brake_coef`, `sim_coef`.

### If air control is ever wanted, in order

1. **Add `ground_dist` and `flying`** - cheap, and `ground_dist` alone makes
   "am I about to land" learnable.
2. **Add angular velocity** - the honest fix for pitch/yaw damping. Not in the
   telemetry today; the plugin would need to ship it, or it can be differenced
   from `dir`/`up` across frames on the Python side (which is really a
   two-frame observation in disguise, so option 3 may be cleaner).
3. **Frame-stack the observation** (2-3 frames) - makes every rate learnable at
   once, including the neoslide timing, at the cost of 2-3x the input width and
   retiring the model. This is the general fix and probably the right one if
   tricks matter.

Nothing here is implemented. Recorded because the technique knowledge is
correct and the reason it cannot currently be used is structural, not a tuning
problem - so it should not be attempted with reward shaping.

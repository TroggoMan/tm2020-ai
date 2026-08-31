# How to actually train this thing

Read this one top to bottom. It assumes nothing.

---

## The one-paragraph version

There are **two models**, and they do different jobs. The **explorer** has never
seen the track and its only goal is to reach the finish. The **racer** is
handed what the explorer learned about the *track* — a route and the geometry —
and its goal is to be fast. The racer never copies the explorer's driving. You
run the explorer once per new map, then the racer forever after.

---

## Part 1 — the pieces, in plain terms

**The reference line** is a string of points through the track, every 2 metres.
It does three jobs and only three:

1. *How far along am I?* — your position projected onto it. This is `progress`,
   the main reward.
2. *Where does the track go next?* — six points ahead of you, handed to the
   network. This is its entire sense of direction.
3. *Am I miles off course?* — distance from it, which can cost you a little
   (`soft_offset`) or end the episode (`max_offset`).

**This is why the line matters so much.** It is not a suggestion the policy can
ignore — points 2 and 3 mean the line is *the instruction*. A line that wobbles
teaches the car to wobble. That was the actual cause of the snaking, and
`tools/straighten_line.py` is the fix.

**The lidar** is 24 rays cast against the map's block grid, telling the car
which directions are open. It comes from the map dump, not from the line, and
it is objective — it is the map, not anybody's driving.

**The checkpoints** come from the game's own landmarks. A checkpoint counts when
you cross its plane, inside its width. Sections are numbered between them:
`cp0->cp1` is the second section.

**The reward** is a sum of named terms, and every one is logged separately in
`logs/why.jsonl`. Progress dominates by design. Everything else is a nudge.

---

## Part 2 — the two stages

### Stage one: the explorer

    train/train_sac.py --stage explore --handover 3 --then-race

That is the whole command. What happens:

1. It asks the game for the map's checkpoint and finish positions and draws a
   **provisional line** straight through them. This line runs through scenery —
   that is fine and expected. It is a statement about *where the finish is*, not
   about where the road is.
2. Because the line is nonsense as a driving line, the off-line penalty is
   switched off and the offset limit is opened right up. The car is free to go
   anywhere. The lidar is what tells it where the road is.
3. The reward is progress toward the next checkpoint, plus a bonus per
   checkpoint taken, plus the finish bonus.
4. Every run that gets further than the previous best is saved as a **trace** in
   `runs/<map>/traces/`.
5. `--handover 3` means: once it has finished **3 times** and then gone 25 more
   finishes without a new best time, stop. Both conditions matter — "finished
   once" might be a fluke it cannot repeat, and "stopped improving" alone would
   never fire on a stage that has not finished at all.

You do not have to sit and watch. It stops itself.

### The handover — what actually crosses over

This is the part you asked about, so here it is precisely. **Three things move,
and only one of them is the explorer's driving:**

| what | how it is used | changed on the way? |
|---|---|---|
| the explored path | the racer's progress axis and lookahead | **yes — smoothed** |
| the occupancy grid | the lidar | no. It is the map. |
| the landmarks | the checkpoint gates | no |

And two things deliberately **do not** cross over:

- **The explorer's weights.** The racer starts from scratch. The explorer was
  trained on a different reward for a different job; seeding from it is how you
  accidentally inherit the wandering.
- **The explorer's exact path as a target.** The trace is smoothed with a 30m
  moving average, and the race config is rewritten to
  `soft_offset: 28m, w_soft: 0.002` — wide and nearly free. The line then says
  *which way round the track goes* and nothing more. Where the road is comes
  from the lidar; how fast you get round is the racer's problem.

You will see this printed when it happens:

    handover: ep0431_step0187k_d1120m.json -> lines/<uid>-explored.json
      1534m of driving, smoothed to 1493m; 836 degrees of steering became 210
      the racer gets this as a ROUTE, not a line to trace: soft_offset 28m at w_soft 0.002

That "degrees of steering" line is the number to watch. If it barely drops, the
smoothing window is too small and the racer is being handed a slalom.

### Stage two: the racer

`--then-race` starts it automatically. If you would rather look first, leave
`--then-race` off and it prints the command.

    train/train_sac.py --stage race --line lines/<uid>-explored.json

Now progress, lap time and the shaped terms take over.

---

## Part 3 — tweaking, in the order you should do it

Everything below is in the map's config file, `configs/<map uid>.json`, and it
is **re-read while training runs** — no restart. The panel edits the same file.

### Step 1: get it driving straight before anything else

Turn everything optional off. `w_gear`, `w_gas`, `w_accel` at their defaults are
fine; set `speedslide.w` and every hint to 0. You want the WHY log to say
progress is 90%+ of the return. If it does not, nothing else you tune will mean
anything.

If it snakes: check the line first, not the reward.

    tools/straighten_line.py lines/<the line>.json --dry-run

If "wander sideways" is more than a few metres on a straight, the line is the
problem. Fix it before touching a single weight.

### Step 2: find out where the time is going

    tools/splits.py --theoretical

Read the `gap` column: median minus best, per section. That is how much is
sitting on the table in each piece of track. The biggest one is where to work.
A section with a low `reached` count is not slow — it is where the car keeps
dying, which is a different problem with a different fix.

### Step 3: work one section at a time

Sections can be named three ways. Use whichever you actually know:

**By checkpoint** — the natural unit, and the same numbers `splits.py` uses:

    {"name": "sector-2", "cp_from": 1, "cp_to": 2, ...}

**By a place you drove to** — when the interesting bit is halfway through a
section. Drive there, then press **Mark here** in the panel (or
`tools/mark.py before-the-jump`), and refer to it by name:

    {"name": "the-jump", "mark_from": "before-the-jump",
     "mark_to": "after-the-jump", ...}

A mark stores the **world position**, not a distance along the line. That is
deliberate: distances belong to whichever line is loaded, and lines get
re-recorded. A mark you drove to last week still points at the same corner
after the explore stage rebuilds the line through it.

**By speed** — for things that are about the car rather than the track, like a
speed slide. Transfers to every map:

    {"speed_from_kmh": 400, "speed_to_kmh": 600, ...}

### Step 4: within a section, nudge or force

A hint does one of two things.

**Nudge** — pay a little for a behaviour so it gets a fair trial:

    {"name": "sd-entry", "cp_from": 1, "cp_to": 2,
     "hold_ms": 120, "gap_ms": 700, "w": 0.05}

`hold_ms` is how long a tap lasts. It pays for a *tap*, not for the pedal being
down — the clock stops at `hold_ms` whether or not the policy lets go, because
a term that paid per step for a held pedal would be maximised by standing on
the brake.

**Force** — put a known input sequence into the buffer for one piece of track:

    {"name": "flat-through-1", "cp_from": 0, "cp_to": 1,
     "gas": 1.0, "brake": -1.0, "steer": 0.0}

Any axis you leave out is left to the driver, so "hold flat through sector 2" is
one line. **Forcing applies to the warm-up only** — never on top of the trained
policy, because the replay buffer records the action the policy chose and an
environment quietly driving something else would fill it with transitions that
never happened as described.

### Step 5: take the shaping away

Once the behaviour appears reliably, turn `w` down toward zero and let lap time
decide. If the trick was actually slower, it should die — and you want to find
that out. A shaped term that is more than a few percent of episode return in
`logs/why.jsonl` is not shaping any more; it *is* the objective.

---

## Part 4 — reading the logs

**The episode line:**

    ep    12  step   45200  FINISH  t= 24.412s  cp= 3  best=24.412s  [18.3m]

`ep` resets each run. `step` is cumulative across resumes and is what SAC's
schedules key off. `t` is the game's race clock. `best` is the best finish
*this process* has seen.

**`logs/why.jsonl`** is the one that matters. It answers "which term is actually
driving behaviour", not "did the number go up". Ask it, every time:

- Is progress still dominant? If a shaped term has taken over, turn it down.
- Is `off_line` large? The car is being punished for leaving the line — check
  whether the line deserves to be followed that closely.
- Is `ent_coef` still high? It is deliberately exploring. Give it time.

**`tools/splits.py`** for where, **`tools/models.py`** for which model is which.

---

## Part 4b — when it gets worse instead of better

It will, sometimes. There are three causes and they need three different
responses, so the first job is telling them apart.

### 1. You changed the reward while it was running

This is the one that catches people, precisely because the hot-reload is so
convenient. The replay buffer holds up to 300,000 transitions and **every one
carries the reward it was scored with at the time**. Change `w_gear` mid-run
and the critic is now learning from a buffer where the same state and action
are labelled two different ways. It does not average them out gracefully — it
thrashes.

Nothing relabels the buffer, and nothing can: the old reward for a transition
depended on state the environment no longer has. The buffer simply has to age
out, which needs roughly as many new transitions as there are stale ones. At
40Hz, 150,000 stale transitions is over an hour of driving.

**The response is to wait.** Rolling back here makes it worse — you would be
restoring a policy trained on the reward you just decided you did not want.
The trainer now prints the transition count when the config changes so you know
how big the wobble should be.

If you are making a *big* reward change, it is cleaner to start a new model
(`--name`) than to drag a buffer scored under the old one along with it.

### 2. The policy walked off a cliff

SAC's actor can move into a region the critic has overvalued, and then the only
data being collected is data that confirms the mistake. This is a genuine
regression and it does not fix itself.

**The response is to roll back.** Snapshots are already being written to
`models/archive/<name>/` on every new best, and the panel's Checkpoints &
snapshots section restores one. Or let it happen automatically:

    --auto-rollback

Rolling back restores the **weights only** and keeps the replay buffer. That is
deliberate: the experience is still valid off-policy data, and it is the
expensive part — every transition in it cost real seconds of real driving.

### 3. The entropy coefficient climbed

`ent_coef="auto"` raises exploration when the policy gets too deterministic.
On the graph that looks exactly like a regression. It is not one — it is
temporary and it comes back better.

**The response is to do nothing.** Check `train/ent_coef` in `logs/why.jsonl`
before touching anything. The guard checks it too and says so in the warning
rather than rolling back over a deliberate exploration burst.

### The guard

On by default, watching a 20-episode rolling mean:

    --regress-window 20     episodes in the mean (0 turns it off)
    --regress-drop 0.25     fraction of return that must vanish to count
    --auto-rollback         restore <name>_best.zip instead of only warning

It prints which of the three it thinks is happening:

    REGRESSION: 20-episode mean 640.1 vs best 1000.4 (1/3) - but ent_coef is
    0.184, so this may be a deliberate exploration burst rather than a regression

`--auto-rollback` is off by default on purpose. An automatic rollback during
cause 1 or 3 undoes progress you wanted to keep, and both are common.

## Part 4c — the window has to be focused (or have its own display)

If the game window is not focused, **it applies less input than you send**.
Around 90% of a command held at full lock, in practice.

The mechanism: TM2020 ramps analogue steering *per frame*. Unfocused, the
frame rate is throttled, so the ramp gets fewer steps and never reaches full
deflection. Physics still runs at 100Hz; it is the input ramp that is
frame-bound.

This is worse than it sounds for a fleet. Transitions recorded while
unfocused have a different action-to-effect mapping from focused ones, they go
into the same replay buffer, and nothing afterwards can tell them apart.

**The detector.** The environment now compares what we sent against
`in_steer` — what the game says it received — sampled only while a command has
been held steady, so it measures a deficit rather than the ramp itself:

    INPUT FIDELITY: the game is applying 90% of the steering we send.

`applied_ratio` also rides along in `info` every step. Tune it under
`input_fidelity` in the config (`window`, `settle_steps`, `warn_below`).

**The fix is structural: give every instance its own X display.** On its own
display it is always the focused window, and N instances stop fighting over
one focus:

    tools/steam-instance 1 --vnc      # Xvfb :101, VNC on 127.0.0.1:5901

Games launched from that Steam client inherit the display, so the whole chain
is covered by starting Steam this way. On a single shared desktop the only
alternative is to keep the game focused and not touch anything else, which is
not a fleet.

## Part 5 — the short answers

**Do I switch models by hand?** No. `--stage race` uses `sac_tm_v2`,
`--stage explore` uses `sac_explore`. `--name` is for running a deliberate
experiment alongside.

**Do I need a recorded lap?** Not for the explore stage. That is its whole
point.

**Will the racer copy the explorer?** No, and three separate things prevent it:
fresh weights, a smoothed line, and an off-line penalty relaxed to almost
nothing. What it inherits is the *map*, not the driving.

**How long?** The explorer needs to finish a few times, which on a simple map
is minutes and on a hard one can be a long while. The racer improves for as
long as you leave it.

**What if the explorer never finishes?** It cannot see the route. Check the
checkpoint order it printed at startup (`--gate-order` overrides it), and check
that the map has an occupancy dump so the lidar has something to cast against —
without one, every beam reads "nothing in range" and the car is blind.

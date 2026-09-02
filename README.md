# TM2020 AI driver

A reinforcement-learning driver for **Trackmania 2020** on Linux/Proton.

- **Control in** through a uinput virtual Xbox 360 gamepad — no injection, no
  memory hooks.
- **Telemetry out** through a custom Openplanet AngelScript plugin — no screen
  capture.
- **Algorithm:** Soft Actor-Critic (SB3), one replay buffer, optionally fed by
  several game instances at once.
- **Nothing** from this pipeline is ever submitted to Nadeo's servers; the car
  always races as `TAS`.

The rest of this file is the design record: every non-obvious decision, and the
debugging that forced it. If you just want to train the thing, read
[`TRAINING.md`](TRAINING.md) instead.

---

## Contents

- [Status](#status)
- [The mental model](#the-mental-model)
- [Quickstart](#quickstart)
- [Repo layout](#repo-layout)
- [Companion docs](#companion-docs)
- [How it's wired](#how-its-wired)
  - [The broker and the socket directions](#the-broker-and-the-socket-directions)
  - [Ports, when running several games](#ports-when-running-several-games)
  - [Packaging: `.op`, not a folder](#packaging-op-not-a-folder)
  - [Signature Mode: School (required, every launch)](#signature-mode-school-required-every-launch)
  - [Map control: no clicking needed](#map-control-no-clicking-needed)
  - [CLI reference](#cli-reference)
- [Reference lines](#reference-lines)
  - [From a replay, or from driving](#from-a-replay-or-from-driving)
  - [How ghost capture actually works (three traps)](#how-ghost-capture-actually-works-three-traps)
  - [A recorded line can teach it to snake](#a-recorded-line-can-teach-it-to-snake)
- [Perception: what the policy can see](#perception-what-the-policy-can-see)
  - [The 106-dim observation](#the-106-dim-observation)
  - [Lidar: ground rays, not wall rays](#lidar-ground-rays-not-wall-rays)
  - [Surfaces](#surfaces)
  - [The network view](#the-network-view)
- [Reward: what pays, and what only looks like it pays](#reward-what-pays-and-what-only-looks-like-it-pays)
- [Training](#training)
  - [The warm-up drives instead of flailing](#the-warm-up-drives-instead-of-flailing)
  - [Two stages: finding the route, then racing it](#two-stages-finding-the-route-then-racing-it)
  - [Which model am I supposed to use?](#which-model-am-i-supposed-to-use)
  - [What loads is not the same as what still means the same thing](#what-loads-is-not-the-same-as-what-still-means-the-same-thing)
  - [Running several games into one model](#running-several-games-into-one-model)
  - [Tuning: a config file per map](#tuning-a-config-file-per-map)
- [Episode lifecycle](#episode-lifecycle)
  - [Episode reset: cheapest button first](#episode-reset-cheapest-button-first)
  - [Checkpoints: count them yourself](#checkpoints-count-them-yourself)
  - [Replays: taken off disk, never off the menu](#replays-taken-off-disk-never-off-the-menu)
- [Actions: throttle and brake are on/off switches](#actions-throttle-and-brake-are-onoff-switches)
- [Operational gotchas](#operational-gotchas)
- [Hard constraints](#hard-constraints)
- [Phase plan](#phase-plan)
- [Dead ends](#dead-ends-confirmed-by-api-inspection-not-assumption)
- [Open lead](#open-lead)
- [Reference / prior art](#reference--prior-art)

---

## Status

Control and telemetry are working end-to-end. Training runs today on a single
paid instance in both the explore and race stages. The fleet (many instances,
one model) is designed and partially built — see [`FLEET.md`](FLEET.md) /
[`FLEET_V2.md`](FLEET_V2.md). See [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) for
the latest working state and the current open thread.

Full checklist in [Phase plan](#phase-plan).

## The mental model

There are **two models**, doing different jobs:

| | knows the track? | goal | run it |
|---|---|---|---|
| **explorer** (`--stage explore`) | no | reach the finish | once per new map |
| **racer** (`--stage race`) | yes — a route + geometry the explorer found | be fast | forever after |

The racer never copies the explorer's *driving*; it inherits what the explorer
learned about the *track*. See [Two stages](#two-stages-finding-the-route-then-racing-it).

Everything spatial the policy knows comes off a **reference line** — a string of
points through the track. The line is an *instruction*, not a suggestion: the
lookahead points the network steers toward and the off-line penalty both come
straight off it. A bad line trains a bad policy. See
[Reference lines](#reference-lines).

## Quickstart

**Prerequisites**

- Trackmania 2020 running under Proton, with Openplanet installed.
- The telemetry plugin deployed into the prefix: `./deploy_plugin.sh`.
- Openplanet **Signature Mode → School** set after every game launch
  ([why](#signature-mode-school-required-every-launch)).
- A Python venv at `.venv/` for the trainer (SB3, Gymnasium, torch). The pad
  server runs on **system** Python — it needs `evdev` and uinput permission.

**Bring it up** — four processes, in this order:

    python3 control/virtual_pad_server.py &    # 8765  control in
    python3 telemetry/broker.py &              # 8767  fan-out of the plugin's 8766
    python3 web/server.py &                    # 8080  control panel
    # then open http://127.0.0.1:8080

Or let the launcher do it: `./tm2020-sacAI` (add `--instances N` for a fleet,
`--mock N` to run without the game).

**Then, from the panel:** get a reference line, then start training. Or from the
CLI:

    # stage one - no recorded lap needed
    .venv/bin/python train/train_sac.py --stage explore --name sac_explore

    # stage two - refine on a recorded/derived line
    .venv/bin/python train/train_sac.py --line lines/newtrack.json --name sac_tm_v2

## Repo layout

    tm2020-sacAI            one entry point: plumbing + panel, then use the browser
    plugin/TMAITelemetry/   Openplanet AngelScript plugin (source of truth)
    deploy_plugin.sh        packages it as a .op and installs it into the prefix
    control/                uinput virtual Xbox 360 pad + a CLI to poke it
    telemetry/              broker (fan-out) + a client for the JSON-lines stream
    env/                    Gymnasium environment, reference line, surfaces,
                            map data (gates + effects), per-map tuning config
    env/ports.py            which ports belong to which instance (stdlib only)
    env/speedslide.py       SDHelper's own speed-slide bands, lifted verbatim
    env/hints.py            "tap the brake here, for this long" - see TRICKS.md
    configs/<map uid>.json  live-reloaded tuning, one file per map
    train/train_sac.py      SAC training entry point
    train/bootstrap.py      SAC whose warm-up drives instead of flailing
    train/scripted.py       the hand-written driver that warm-up uses
    train/why.py            per-episode reward decomposition in English
    web/                    local control panel (stdlib only)
    tools/fleet.py          starts and supervises N pad servers + N brokers
    tools/mock_game.py      a fake game, for testing the fleet without accounts
    tools/models.py         what every model on disk is, and which one is live
    tools/maps.py           push a .Map.Gbx to every instance, and load it
    tools/straighten_line.py  take the wobble out of a recorded reference line
    tools/                  closed-loop test, command CLI, line + replay capture
    lines/  models/  logs/  reference lines, checkpoints, logs
    maps/<uid>.materials.json  surfaces actually driven on, for the tuning UI
    models/<name>.meta.json    what a model was trained AGAINST, checked on resume
    runs/<map>/replays/     .Replay.Gbx lifted from the game on every PB
    runs/<map>/traces/      JSON traces of runs that got further without finishing
    models/archive/<name>/  rollback snapshots

## Companion docs

| file | what's in it |
|---|---|
| [`TRAINING.md`](TRAINING.md) | how to actually train it, top to bottom, assumes nothing |
| [`TRICKS.md`](TRICKS.md) | teaching a technique (slide, wallride…) with the see-it / reach-it / pay-for-it ladder |
| [`FLEET.md`](FLEET.md) | scaling to N containerised instances into one buffer |
| [`FLEET_V2.md`](FLEET_V2.md) | the privileged-surveyor architecture for scaling past one paid account |
| [`GRID.md`](GRID.md) | the live video grid of every instance in one page |
| [`ACCOUNTS.md`](ACCOUNTS.md) | the manual Ubisoft-account signup runbook |
| [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) | latest working state and current open thread |

---

## How it's wired

### The broker and the socket directions

The **plugin listens** on 8766 and Python connects to it. So Python must not
hold 8766 itself, and the policy process can restart freely without the game
noticing. Control is the other way round — the pad server listens on 8765 and
Python connects to send inputs.

The **broker** exists because the plugin serves exactly one client: it holds
that slot and re-serves the stream locally, so the trainer, the panel and any
listener can coexist. Everything except the broker should talk to **8767**, not
8766. Commands written by any client are forwarded upstream to the plugin.

### Ports, when running several games

The three base ports are adjacent, so per-instance they step by **ten**, not
one. Stepping by one puts instance 1's pad on instance 0's plugin.

    instance 0    pad 8765   plugin 8766   broker 8767
    instance 1    pad 8775   plugin 8776   broker 8777
    instance 2    pad 8785   plugin 8786   broker 8787

Instance 0 is unchanged, so a single-game setup needs nothing new. The mapping
is in `env/ports.py` — stdlib only, because the pad server runs on the *system*
python and cannot import from the venv. `tools/fleet.py --instances N` starts
and supervises the pad servers and brokers; `--check` reports which ports are
listening. The plugin port lives in each game's own Openplanet config, inside
that game's Wine prefix, and is the one thing fleet.py cannot set.

### Packaging: `.op`, not a folder

Openplanet silently ignored the unpacked plugin folder — no load line and no
error in `Openplanet.log`, as if it were never scanned. Every working
third-party plugin in this prefix is a `.op` (a plain zip of `info.toml` +
`main.as`), so `deploy_plugin.sh` builds one. Loading folders may need
Openplanet's Developer Mode (`Settings.ini` has `DeveloperMode=false`), but the
`.op` path is proven and avoids the question.

### Signature Mode: School (required, every launch)

The `VehicleState` API is School-mode only. In game: `F3` overlay → `Openplanet`
→ `Signature Mode` → `School`. **It does not persist across relaunches.** This is
also what guarantees nothing reaches Nadeo's leaderboards.

This is the blocker on unattended multi-instance training — a relaunched worker
needs a human to re-toggle it. It appears to be deliberately non-persistent:
nothing in `Settings.ini` records the signature mode.

**Planned workaround — script the overlay click.** Not anti-cheat evasion: School
is the *more* restrictive mode, and automating the toggle only ever moves the
game into the state that disables leaderboard submission. Groundwork confirmed:

- The session is Wayland, but the game is an XWayland client, so `xdotool`
  can see it: window name `Trackmania`, geometry a fixed **1920x1080 at (4,30)**.
  The window id changes per launch, so search by name each time.
- `ydotool` (uinput-based, works natively on Wayland) is installed as a
  fallback if XTEST events don't reach the overlay.
- `OverlayScale=1` in `Settings.ini`, so overlay coordinates should be stable
  across launches.

The eventual multi-instance workers run on the **Ubuntu server (X11)**, not this
Wayland desktop, so the click automation has to be written for X11 anyway.
That's the easier target: `xdotool` is native there with no XWayland caveat, and
with `Xvfb` per instance each worker owns its own display, so clicks can't
collide or steal focus. Keep the toggle script display-scoped (`DISPLAY=:N`)
rather than assuming a single global desktop.

### Map control: no clicking needed

`ManiaExchange.op` unzips to a plugin whose exports are info/UI only, but the
useful part is *how it plays a map*, in `src/Utils/Game/Methods.as`, which we
copy directly. All verified present in `Openplanet.h`:

    app.BackToMainMenu()                                  // else you stick on the current map
    while (!app.ManiaTitleControlScriptAPI.IsReady) yield()
    app.ManiaTitleControlScriptAPI.PlayMap(url, mode, "") // mode "" = default on TM2020
    app.ManiaTitleControlScriptAPI.PlayMapList(list, ...) // a whole curriculum in one call

    app.Network.PlaygroundClientScriptAPI.RequestRestartMap()   // episode reset
    app.Network.PlaygroundClientScriptAPI.RequestGotoMap(uid)
    app.Network.PlaygroundClientScriptAPI.RequestNextMap()

`PlayMap` takes a URL, so a TMX map id is enough to load anything:
`https://trackmania.exchange/mapgbx/<id>`. That gives map selection, episode
reset and curriculum rotation **through real APIs, with no simulated input** —
the click automation is now only needed for the School Mode toggle.

Caveat carried over from ManiaExchange: it gates on
`Permissions::PlayLocalMap()`, so check that permission before relying on this in
an unattended worker.

The plugin's command channel exposes all of this on the same socket as telemetry
(`restart`, `goto <uid>`, `playmap <url>`, `menu`, `rate <hz>`, `ping`), driven
by `tools/tmai_cmd.py`.

### CLI reference

    ./deploy_plugin.sh                              # after editing the plugin
    python3 control/virtual_pad_server.py &         # listens on 8765, control in
    python3 telemetry/broker.py &                   # listens on 8767, fan-out of 8766

    python3 control/padctl.py demo                  # scripted wiggle
    python3 telemetry/telemetry_listener.py --stats # measure real tick rate
    python3 telemetry/telemetry_listener.py --once  # one line, then exit
    python3 tools/closed_loop_test.py               # verify both halves are one loop
    python3 tools/tmai_cmd.py restart               # episode reset
    python3 tools/tmai_cmd.py landmarks             # checkpoint/finish positions
    python3 tools/tmai_cmd.py dumpmap               # effect blocks (boosters etc)
    python3 tools/tmai_cmd.py dumpmap occupancy     # + solid grid cells (lidar)
    python3 tools/replay_watcher.py runs/manual     # capture PB replays by hand

---

## Reference lines

### From a replay, or from driving

You do **not** have to drive a lap to get a line. Openplanet's `VehicleState`
reads whatever vehicle is currently *viewed*, and its own export docs say the
state stays valid while spectating — one function is even documented as the
exception that "doesn't work when watching a replay", which implies the rest do.
So playing a replay produces the same telemetry as driving, through the same
code path.

`tools/record_line.py` therefore has one implementation and three modes:

    auto    (default) whatever is being viewed, once it moves - replay or live
    replay  only when it is NOT your own spawned run
    live    only your own spawned run

The panel has a replay browser (search, run times parsed out of the filenames,
newest first, plus a paste-a-path box). Picking one names the line and switches
the mode. It cannot auto-load the replay — no Openplanet API for that was found,
only menu dialog callbacks — so open it in game and press play, then hit Record.

Recording also optionally captures the viewed car's **inputs**, which makes a
replay recording double as demonstration data for seeding SAC's buffer later.

Verified against a known replay: captured **24.747s vs the replay's 24.595s
(+0.152s)**, 627 samples at 2m spacing over 1253m — consistent with the 182 km/h
average. Pass `--expect-time` and it prints that check itself.

### How ghost capture actually works (three traps)

Watching a ghost works, but not the obvious way. All three of these cost real
debugging time:

1. **`ViewingPlayerState()` returns null while watching a ghost.** There is no
   viewing player, and its fallback `GetSingularVis()` only works when the scene
   holds *exactly one* car. With player + ghost present it returns null. The fix
   is `VehicleState::GetAllVis(GetApp().GameScene)`, which enumerates every
   vehicle; each has `.AsyncState`. The recorder takes the fastest distinct one.
2. **Do not gate the vehicle read on the playground.** Watching a replay from
   the menu creates no `CSmArenaClient` at all, so an early `return` on a null
   playground reports "no car" for every frame. Read the vehicle first,
   unconditionally; treat race context as a bonus. There is no `race_time`
   without a player API, so the recorder falls back to the plugin's own clock.
3. **There is no `finished` flag either**, so a ghost on repeat records forever,
   concatenating laps into one nonsense line. The recorder detects the restart
   as a positional jump (`--lap-gap`, default 30m) and stops after `--laps`.

**`pkill`/SIGINT and background jobs.** A process started with `&` from a
non-interactive shell inherits **SIGINT ignored**, so "stop" signals silently do
nothing and the job records forever. `record_line.py` installs handlers for both
SIGTERM and SIGINT that exit cleanly and still write the file.

Offline `.Replay.Gbx` parsing (no game running) is possible but not done: the
files are `CGameCtnReplayRecord`, class `0x03093000`, with an **LZO-compressed
body**, so it needs a real Gbx parser. That folds into the Gbx work already
planned for track geometry.

### A recorded line can teach it to snake

This one cost a lot of confused training time, so it is worth stating plainly:
**the reference line is an instruction, not a suggestion**, and a line recorded
by driving contains every correction the driver made.

`lines/ExploreMode.json` is a recording of the *straight* road map. It measures

    length 1534.2m   chord 1482.9m   wanders up to 11.0m sideways
    836 degrees of cumulative heading change, on a straight road

Both of the policy's spatial inputs come off that line, and both of them punish
driving straight:

* the six lookahead points swing sideways by up to 11m, so the thing the policy
  is being aimed at is a slalom; and
* `soft_offset` is 8m, so a car driving perfectly straight down the middle of
  the road is outside the line's own wobble for much of the map and is charged
  the off-line penalty for it.

Progress itself is *not* the culprit and never was — it is arc length along the
line, measured by projecting the car onto it. Simulated at a fixed 60m/s of real
travel, a hard weave scores 31 m/s of progress against 60 m/s for driving
straight, and pure sideways wobble with no forward motion nets exactly 0.00m.
Weaving has always cost progress. What paid for it was the line.

    tools/straighten_line.py lines/ExploreMode.json --axis auto \
        -o lines/straight-road.json --map 3TKGZO6WiXCd8QLO8LtC2gD2Ow0

`--axis` forces a perfectly straight line at the recording's *median* lateral
position, which is what a straight-road map wants. `--smooth W` is the general
case for real tracks; `--dry-run` reports what it would change without writing.
The report to read is "moved sideways from the recording" — if that is wider
than a road block, the window is too big and the line may no longer be on the
track.

Two related things worth knowing:

* **None of the three recorded lines had a map uid stamped in it.** That is the
  field which stops a line being loaded against the wrong track, and it was
  empty on all of them. `--map <uid>` stamps it.
* **Nothing penalises snaking.** `w_weave` and `w_reversal` exist and both
  default to `0.0`, with `enabled.weave` false. Straightening the line removes
  the *reason* to snake; turning those on is what removes the *freedom* to.

---

## Perception: what the policy can see

### The 106-dim observation

The v1 observation was 37 dims and contained no surface information at all —
the plugin streamed `mat` and the env dropped it. v2 adds:

- **grip class per wheel** (6-way one-hot ×4). Grouped, not raw: there are 81
  materials, a track uses about six, and grip class is the part that transfers
  to a map with a different palette. Per-wheel because two wheels on tarmac and
  two on grass is the situation that spins you.
- per-wheel icing and dirt, plus car wetness
- **effects acting now**: turbo + level + remaining, reactor level/type/timer,
  cruise, slow-motion, air brake, side speed
- **effects ahead**: distance and lateral offset to the next boost, handicap,
  danger and bumper, projected onto the reference line from `dumpmap`. This is
  the half that matters — reacting to a booster once you are on it is too late
  to have chosen a line into it.
- **16 lidar beams** (see below)

v2 is a different shape, so it trains as `sac_tm_v2`. `sac_tm.zip` and its
156k-transition buffer are untouched and still runnable with `--name sac_tm`.

Boosters are a **carrot, not a rule**: `w_turbo_use` rewards throttle while on a
boost it already reached. It is never told it must take one, and lap time stays
the dominant term, so a detour to farm a booster still loses.

### Lidar: ground rays, not wall rays

There is no raycast API in TM2020 and screen capture would need a GPU budget
and a trained encoder to learn what the game already knows. But every block
exposes the exact grid cells it occupies (`BlockUnitsE` → `AbsoluteOffset`), so
`dumpmap occupancy` exports the whole grid once per map, cached to
`maps/<uid>.json`, and `env/lidar.py` marches 16 forward-weighted beams against
it — 0.36ms per sweep against a 25ms control period.

**They measure distance until the ground runs out, not wall distance.** A block
unit marks a cell the block occupies in the editor, not where the drivable
surface sits inside it, so a road block's cell reads "solid" even though you
drive through it. Casting for the first *solid* cell would report a hit
instantly in every direction. Inverting the test — march until the cell is
*empty* — gives how far you can go that way before there is no track under you.

Cells are 32×8×32m and the march step is 16m, so a beam resolves to 16m. That
answers "which way does the track continue" and "how many blocks wide is it
here", which is what stage one needs. It is **not** precise enough to place a
car against a barrier; the reference line and the live per-wheel surface
readings do that, and the lidar sits alongside them.

Two details that cost real debugging: the beams check a **three-cell vertical
window**, because the car's reported Position lands one cell either side of the
block it is on depending on ride height and camber — looking only at the car's
own layer reports "no ground" in every direction at once. And a map with no
occupancy dump reports 1.0 on every beam rather than a shorter vector, so the
observation keeps its shape whether or not the scan succeeded.

### Surfaces

`EPlugSurfaceMaterialId` is the whole **Nadeo engine's** enum — 81 entries
shared with ShootMania and with editor test surfaces. A Trackmania track uses
about six of them. The panel used to list all of them, which meant listing ~75
rows that could never fire.

The tuning UI now has three tiers, narrowest first:

| tier | what it is |
|---|---|
| **on this map** | materials the environment has *actually* recorded under the wheels while driving. Evidence, not a guess. |
| **Trackmania surfaces** | the hand-picked 22 any track plausibly uses. |
| **everything** | the full enum, for something exotic. |

The environment writes `maps/<uid>.materials.json` as it drives, so the first
tier fills in the moment a trainer touches a map. Anything already given a
weight stays visible whatever the tier is set to, or saving would silently drop
a setting you cannot see.

### The network view

`train/nn_probe.py` hooks the actor's hidden layers and writes
`logs/nn_state.json` atomically at 8Hz; the panel renders it. Everything drawn
is real:

- node brightness is the neuron's **actual activation** on the observation the
  policy is acting on right now;
- every edge is one of the **strongest real weights** into that neuron, pulled
  out of the weight matrices at training start, tinted by its true sign;
- the **critic's Q estimate** for the current state and action is sampled too —
  the min of the twin critics, since that is the one SAC trains on.

Two deliberate reductions. The hidden layers are 256 wide, so a fixed-stride
sample of 24 is drawn — fixed, so a given dot is always the same neuron and you
can watch it over time. And the 90 inputs are drawn one node per observation
*group*, because 90 dots in a column is unreadable and the groups are what you
reason about anyway. Edges in the later layers are restricted to neurons that
are themselves drawn, so an edge never points at something off screen.

---

## Reward: what pays, and what only looks like it pays

### Progress is already distance along the path

`progress` is `Δs` where `s` is **arc length along the reference line**, from
`Centerline.project()`. It is not distance travelled. Cutting a corner advances
it by exactly as much as going the long way round, so there is no incentive
anywhere to add distance — driving in circles earns nothing.

What progress does *not* do on its own is make being slow hurt. At 100 km/h a
step earns ~0.7 of progress against a `step_cost` of 0.02 — a 35:1 ratio, so
crawling is barely worse than flying.

`reward.par_speed` sets that ratio directly. Above 0 it replaces the flat step
cost with one measured in the same units as progress, so the two terms sum to:

    w_progress * dt * (speed_along_line - par_speed)

Driving at par earns nothing, faster earns, slower actively loses. For roughly
the 2:1 you want at a target speed, set `par_speed` to half of it. It is 0 by
default; the panel edits it in km/h.

### The off-line terms are about the LINE, not about sliding

`soft_offset`, `w_soft` and `max_offset` are all lateral distance from the
reference line in metres. **Nothing in the reward penalises drifting the car.**
Per-wheel slip is an observation input and never a reward term, so a speedslide
that carries speed is strictly better here — it finishes sooner, and lap time
is what pays.

The one place a good technique could get trained out is `w_reversal`, which
charges for a full steering sign flip at speed. That is also what a neoslide
looks like from the outside. Both anti-twitch terms default to 0; if you want
to kill brute-force flutter, raise `w_weave` (size of the steering change) and
leave `w_reversal` alone.

Worth knowing about `w_soft` in the other direction too: the reference line
came from a human lap, so leaning on the off-line penalty hard caps the policy
at *copying* that lap. Widen the corridor rather than raising the penalty if
you want it to find its own line.

### "road borders: -500"

`surfaces` maps an `EPlugSurfaceMaterialId` name to a per-wheel, per-step
reward. But a -500 per-step penalty swamps every other term and teaches the
policy that the reward function is broken, not that grass is bad. What you
actually want is `terminate_on_surface: ["Grass"]` with `surface_grace_steps`,
which says "that was a crash" cleanly. Empty the list later to let it hug the
edges. Both are one click in the panel.

### Crashing must not be a way to stop the clock

A per-step time cost makes crashing *profitable*: dying at step 100 of 1800
avoids 1700 steps of charge, which is worth far more than any crash penalty, so
the policy learns to drive into the nearest wall.

`reward.charge_unused_time` (on by default) charges a **failed** episode for the
time it did not use. A finish is never charged — beating the clock is the
entire objective. For a failed episode `step_cost + unused_time` is therefore a
constant, identical whether it died on step 100 or step 1500, which is the
point: it removes the incentive without adding one.

The WHY log reports the two together as one *time* charge for exactly this
reason. Ranking them separately made it announce that "78% of this episode was
the unused_time term, not lap time", which is precisely backwards —
`unused_time` **is** lap time.

### Gears, without the flip-flop

`reward.w_gear` pays per step, scaled `gear / top_gear` — but only once the gear
has been **held** for `gear_hold_steps` (8 steps, 0.2 s). That one condition is
what kills the flip-flop: an oscillation between two gears never accumulates
hold time, so it never earns, and there is nothing to farm by bouncing off a
shift point. Committing to the higher gear is the only way to collect.

Deliberately small (0.02). Gear is a proxy for speed and progress already pays
for speed; this is a nudge toward committing, not a second speed reward.
`w_downshift` exists and defaults to 0 — downshifting is *correct* into a slow
corner, and punishing it teaches the car not to brake.

### Acceleration

Two separate carrots, both small and both on by default:

* `w_gas` (0.01) pays for holding the throttle at all — the "just accelerate"
  prior a person tries first;
* `w_accel` (0.05) pays for the metres per second **actually gained**, so
  holding the throttle while pinned against a wall earns the first and not the
  second.

### Speedslides, above 400 km/h only

Below the threshold, sliding is a mistake. Above it a controlled slide *is* the
fast line. So the target is a **band**, not a minimum: too little slip and it is
understeering wide, too much and it is scrubbing speed off in a spin.

    "speedslide": {"min_speed_kmh": 400, "slip_lo": 0.15, "slip_hi": 0.45,
                   "angle_lo_deg": 4, "angle_hi_deg": 20, "w": 0, "w_outside": 0}

Below `min_speed_kmh` this term does nothing at all. It is **off** by default,
because turning it on before the car is that quick costs nothing and does
nothing except add a term to the WHY log.

Honest caveat: **we cannot see skidmarks.** There is no API for them. The
closest measures the game gives us are per-wheel `slip` and the car's sideways
velocity, so the band is set on mean wheel slip and on slip angle
(`atan(side_speed / speed)`). Those are what the skidmark renderer is itself
keyed off, but they are a proxy and the numbers will need tuning against
footage.

**SDHelper's numbers, not ours.** The skidmark question, answered: SDHelper does
not read skidmarks either. It reads `VehicleState::GetSideSpeed`, `FrontSpeed`,
and the material under the front-left wheel, and then swaps the skidmark
*texture on disk* so the marks the game draws come out green / yellow / orange /
blue. The colour is its output, not its input — and both of its inputs are
already in our observation (`side` and `speed`).

`env/speedslide.py` carries the table verbatim. Road, forward, above 400km/h,
in km/h of sideways speed: green (the target) is **19–22**, yellow out to 13
and 28, orange out to 7 and 34. Grass and dirt slide from 200km/h and want far
less sideways speed; reversing shifts every band up and drops the floor
entirely. The reward is a ramp that peaks across green and tapers to nothing at
the orange edges — a step function would give the policy no direction to climb
in from outside the band.

    "speedslide": { "w": 0.15, "w_blue": 0.0, "speed_floor_kmh": 0.0 }

`speed_floor_kmh` at 0 means "use SDHelper's own per-surface floor"; set it to
override, which is the only way to test a slide reward on a car that cannot
reach 400km/h yet.

### Hints: "tap the brake here, for about this long"

A speed slide is not on the way to anything. At 450km/h in a straight line the
car has zero sideways speed, and there is no gradient toward 19km/h of it —
the band is reached by a brake tap, not by drifting toward it. Uniform
exploration at 40Hz will not roll a 120ms pulse at the right speed and then
roll the steering that keeps it.

So a hint says where to try and how long the pulse should be:

    "hints": [
      {"name": "sd-entry", "speed_from_kmh": 400, "speed_to_kmh": 600,
       "hold_ms": 120, "gap_ms": 700, "w": 0.05}
    ]

The scripted warm-up **performs** it, so the buffer contains real tapped
transitions from the first minute, and the reward pays a little for tapping
inside the window so the behaviour has a reason to persist once the warm-up
ends. Both read the same file, so the tap and the payment cannot disagree
about how long a tap is.

It pays for a *tap*, not for the pedal being down: the clock starts when the
pedal goes down and payment stops at `hold_ms` whether or not the policy lets
go. A term that paid per step for a held pedal would be maximised by standing
on the brake.

Scope a hint by a speed window, by an arc-length section (`s_from`/`s_to`), or
both. Speed windows are usually what you want — "in the 400–600 range" is a
statement about the car and transfers to every map. Note that the scripted
warm-up can only perform *speed-window* hints: arc length is not in the
observation, and under `SubprocVecEnv` the env that knows it is in another
process. Section hints still work as the reward term, inside the env.

See [`TRICKS.md`](TRICKS.md) for the full see-it / reach-it / pay-for-it ladder.

---

## Training

### The warm-up drives instead of flailing

> *"the ai hasn't even tried the most basic option of just driving in a
> complete straight line and just accelerating"*

Correct, and it was not a fault in the policy — it is what SAC's warm-up does.
For the first `learning_starts` steps SB3 ignores the network entirely and
samples uniformly from the action space. With three independent uniform axes,
"full throttle, straight, for two seconds" has a probability near zero of ever
appearing: every step re-rolls the steering, so the car snakes at a random
average throttle and stops. The buffer then fills with thousands of
transitions all saying *flailing goes nowhere*, and that is what the critic's
first opinion of the world is built from.

`train/bootstrap.py` replaces those actions with a hand-written driver
(`train/scripted.py`):

| `--bootstrap` | what it does |
|---|---|
| `pursuit` *(default)* | full throttle plus pure-pursuit steering toward the 10 m lookahead point, braking only for a sharp corner at speed |
| `straight` | full throttle, no steering. Type A. |
| `off` | SB3's uniform random flailing |

A quarter of warm-up steps stay uniform random (`--bootstrap-random`) and the
scripted steering carries jitter, so the buffer still contains alternatives to
compare against — SAC's entropy term needs to see that other actions exist.

This is **not** imitation learning. There is no behavioural-cloning loss and no
constraint on the policy anywhere. SAC is off-policy, so transitions from a
scripted driver are exactly as usable as transitions from a random one — they
are just better. After warm-up this is stock SAC, byte for byte, and a
`--resume` turns it off entirely because a trained policy has nothing to warm
up.

### Two stages: finding the route, then racing it

    # stage one - no recorded lap needed at all
    python3 train/train_sac.py --stage explore --name sac_explore

    # hand over: best run -> a real reference line
    python3 tools/line_from_trace.py --map <uid> lines/newtrack.json

    # stage two - refine on the route it found
    python3 train/train_sac.py --line lines/newtrack.json --name sac_tm_v2

The trick in stage one is that **the route order is static map data**.
`Arena.MapLandmarks` gives spawn, every checkpoint and the finish as world
positions, so "find the ending" needs no exploration — what is unknown is where
the road physically goes between them, and that is the lidar's job.

`provisional_line()` strings those positions together into a `Centerline`, so
stage one needs no new environment: progress along that line is progress toward
the finish, and every lookahead, projection and reward term already written
keeps working. It is a terrible racing line — it cuts through scenery and
ignores every corner — so `--stage explore` opens `max_offset` to 250m and the
episode cap to 120s automatically.

`tools/line_from_trace.py` then converts the best trace (a finished run beats
any unfinished one, however far it got) into a line file of exactly the same
format a recorded human lap produces, so nothing downstream changes.

### Which model am I supposed to use?

    python3 tools/models.py

Short answer: **none of it needs switching by hand — the stage picks it.**

| | model | |
|---|---|---|
| `--stage race` | `models/sac_tm_v2.zip` | the racing model |
| `--stage explore` | `models/sac_explore.zip` | needs no recorded lap |

`sac_tm.zip` is the 37-dim ancestor. It cannot be loaded any more — the
observation is 106 dims now — and it does not need to be: `sac_tm_v2` was
produced from it by `tools/transplant.py`, which copied every weight into the
wider layers and zeroed the new columns, so at step one the wider network
computed exactly what the old one did. Verified over 200 probes: actor
difference 0.00e+00, critic 8.4e-07 relative.

`_best` files are written automatically whenever a finished lap beats the
previous best. They are never resumed from unless you ask for them by name,
because the newest model has more experience even when it is momentarily
slower. Roll one back from the panel's *Checkpoints & snapshots*, or with
`--name`.

### What loads is not the same as what still means the same thing

SB3 checks the observation width on load, so a mismatch fails loudly. Nothing
checks the rest, and the rest matters just as much. With a **binary throttle**,
an actor output of 0.3 used to mean 65% throttle and now means fully on: the
model loads perfectly and drives differently.

Every save now writes a `models/<name>.meta.json` sidecar recording the action
convention, the checkpoint mode and the control rate, and `--resume` compares
it and says so:

    trained 2026-08-26 21:40 under different rules:
      - throttle was continuous and is now on/off - the same actor output means
        a different pedal position, so expect a dip before it re-adapts

The weights are still worth far more than a fresh start. It is a warning, not a
problem.

### Running several games into one model

    ./tm2020-sacAI --instances 3        # plumbing + panel
    # then pick "instances: 3" in the panel, or:
    python3 train/train_sac.py --instances 3 --line lines/spring2026-03.json

**N simulations, ONE model.** Not N models. Each instance runs its own
`TrackmaniaEnv` against its own game, and all of them push into the *same*
replay buffer with the *same* weights being updated from all of it. SAC is
off-policy, so it does not care which game a transition came from. The obvious
alternative — running N copies of the trainer — gives N separate models each
learning from a fraction of the data, and they cannot be merged afterwards.

`SubprocVecEnv`, not `DummyVecEnv`: each env spends its whole step blocked on a
socket and a sleep, so in one process they would serialise and N instances
would each run at 40/N Hz. In separate processes they overlap and N games cost
the same wall-clock as one. One instance stays `DummyVecEnv` — no fork, no
pipe, and a traceback you can read.

**The learner is inside the control loop.** This is the one thing that does
*not* scale with instances. SB3 runs its gradient update on the same thread
that steps the environment, so every update happens **inside** the 25 ms
control period. Measured here, batch 256 over a 256×256 net:

| gradient steps | CUDA | CPU |
|---|---|---|
| 1 | 2.9 ms | 11.9 ms |
| 2 | 5.9 ms | 25.7 ms |
| 4 | 12.1 ms | 46.1 ms |
| 6 | 18.3 ms | 59.0 ms |
| 8 | 27.2 ms | 89.9 ms |

Overrun the period and the car keeps driving while nothing is being sent to it
— and worse, every transition still *claims* to be a 25 ms one, so the model
learns the dynamics of a game running at a rate it never ran at. Three
instances at 6 updates a round measured **30 Hz, not 40**, with ~300 overruns
per episode.

So `--gradient-steps 0` (the default) caps the count at what fits in half the
period, and the episode log prints `slip=N` when it overran anyway. The
consequence is real and accepted: adding instances raises transitions per
second without raising updates per second, which *lowers* the
updates-per-transition ratio. That is the right side of the trade — more
diverse data with fewer updates each beats hammering a small buffer.
`env/tm_env.py` also paces against an **absolute** clock rather than sleeping
`dt` from the top of `step()`, so the learner's time is absorbed into the
control period instead of added on top of it.

**Accounts are the part that is not automated.** One Ubisoft account can only be
signed in once at a time, so N concurrent games need N accounts, created and
signed in **by hand**, once each per Wine prefix, after which the prefix is
snapshotted and reused. Scripting that login is the single thing most likely to
get the accounts banned. Full runbook in [`ACCOUNTS.md`](ACCOUNTS.md); the
scaling architecture is in [`FLEET.md`](FLEET.md) / [`FLEET_V2.md`](FLEET_V2.md).

**Testing without N copies of Trackmania:** `./tm2020-sacAI --mock 3`.
`tools/mock_game.py` speaks both protocols and simulates a car with enough
physics to be driven. It exists purely to verify that N envs, N pads and N
brokers are wired to N *separate* simulations. The physics are a toy; a policy
trained against it has learned nothing about Trackmania.

### Tuning: a config file per map

`configs/<map uid>.json`, created from the CLI values the first time a map is
seen and owned by the panel after that. `TrackmaniaEnv` stats it about once a
second and applies changes **on the next step** — so you can watch a run, see
it clip the same barrier every lap, raise the grass penalty, and have it take
effect without restarting and losing the warm-up.

Everything that can change mid-episode lives there: reward weights, the stuck
and off-line thresholds, the reset timings, per-surface rewards, and the
`enabled` toggles. `control_hz` deliberately does not — it sets the action
repeat the model was trained against, so changing it would alter the meaning of
every transition already in the buffer.

CLI flags on `train_sac.py` only **seed** a config that does not exist yet. Once
the file is on disk it wins, because otherwise every panel edit would look like
it had been silently ignored.

---

## Episode lifecycle

### Episode reset: cheapest button first

`ui` in the telemetry stream is `CGamePlaygroundUIConfig::EUISequence`. The
ladder in `env/tm_env.py:_reset_sequence()` acts on two values and waits out
the rest:

| `ui` | state | what it does |
|---|---|---|
| 1 | Playing | press the give-up button (`b`) |
| 11 | Finish | press `b`; after `finish_fallback_after` tries, `a` ("Improve") |
| 2 / 3 | Intro, Outro | mash `skip_button` — any button skips the fly-in |
| anything else | loading | wait; a button fired at a menu that isn't listening is lost |

**Order matters more than it looks.** Give-up respawns instantly. Improve
always works but goes through the full restart *with* the ~8s intro — the exact
cost give-up exists to avoid. Reaching for Improve first made every reset after
a personal best eight seconds slower, and because the wait was shorter than the
intro, the ladder then retried on top of a reset that was already running.
Measured on the stubbed game: **0.6s when `b` takes, 5.4s via the Improve
fallback, 15.4s if the intro is not skipped.**

Attempts escalate the *settle delay*, not the button: the usual failure is
pressing before the screen has accepted input, not pressing the wrong thing.

Success requires **the race clock going backwards AND being out of the Intro
sequence**. The clock reads ~0 during the fly-in, so testing the clock alone
hands back control before the car can be steered. Do not reintroduce
`SpawnStatus`: it reads `NotSpawned` while the car is demonstrably driving, and
requiring it made every successful give-up look like a failure.

### Checkpoints: count them yourself

`RaceWaypointTimes.Length` reads **0 in Time Attack** no matter how many
checkpoints have been passed — this is why every log line said `cp=0` during
runs that were plainly reaching checkpoints. It is a reporting bug, never a
driving failure; the reward has always used reference-line progress.

The fix does not go through the player API at all. `CSmArena.MapLandmarks`
gives every checkpoint and the finish as a **world position**, which the plugin
serves via the `landmarks` command.

**Going *through* the gate, not near it.** Proximity is not the same test as the
game's. A 20m sphere around a checkpoint credits it for driving *past* it on the
road running alongside, or on a bridge above it.

`cp_mode: "gate"` (the default) requires a genuine transit. Between two
telemetry samples, all three of these must hold:

* the step **crosses the gate's plane** — the signed distance along the gate
  normal changes sign, in either direction, because TM credits a checkpoint
  however you go through it;
* at the interpolated crossing point the car is within `cp_half_width` of the
  gate line, measured sideways from the *nearest landmark in the cluster* — so
  a wide multi-block gate counts at its edge as well as its middle;
* and within `cp_height` vertically, so a road stacked over the checkpoint
  cannot collect it from above.

The crossing point is interpolated rather than tested at either end: at 40Hz
and 400 km/h the car moves ~2.8 m per step and a gate is a plane with no
thickness. The landmark gives a position and no rotation, so the plane's
**normal is inferred from the route** — a checkpoint faces along the track, and
the track there runs from whatever precedes it to whatever follows. Horizontal
only. Reordering the gates (`--gate-order`) recomputes the normals.

`cp_mode: "sphere"` restores the old test; it is only useful for comparing
against traces recorded before this changed.

### Replays: taken off disk, never off the menu

The finish screen offers Improve / Save Replay / Exit, and we touch none of it.
The game already autosaves every personal best to

    Documents/Trackmania/Replays/Autosaves/<login>_<map>_PersonalBest_TimeAttack.Replay.Gbx

— one file per map, with no time in the name, so the **next** PB overwrites it.
`tools/replay_watcher.py` watches that directory and copies the file out the
moment its mtime moves, named for the episode that produced it. No D-pad
navigation, and nothing breaks when Nadeo moves a button.

The autosave only fires on a *finished* PB, so the env also writes its own JSON
trace whenever a run gets **further** than any before it — otherwise the runs
that die two thirds of the way round, which are most of them and the
interesting ones, would leave nothing behind.

---

## Actions: throttle and brake are on/off switches

`action.binary_gas` / `action.binary_brake`, both on by default. The three
network outputs stay continuous — SAC's squashed Gaussian needs that — but the
pedals threshold at zero.

TM2020's throttle is not really analogue: on a keyboard it is a switch, and the
fast line is very nearly always fully on or fully off. Letting SAC hold 0.43 gas
gives it a whole continuum of mediocre options to get lost in, every one of them
slower than the two ends, and 0.47 looks much like 0.43 to the critic. Steering
stays analogue, because it genuinely is.

Steering is additionally **quantised into notches and slew-limited** on the race
stage — see [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md). The continuous command
was changing sign between 41% of consecutive control steps (sawing the wheel
8×/s), which barely turns the car whatever the policy intends.
`action.steer_levels` and `action.steer_rate` are set **per stage**: the
explorer gets a slew limit, the racer gets lock-to-lock in one frame because
some TM technique needs it.

---

## Operational gotchas

**The plugin serves one client at a time.** `Accept()` is called once per
connection cycle, so a second consumer sits unaccepted in the backlog and looks
like a silent hang. This is the whole reason the broker exists — everything
should connect to **8767**, not 8766. Stop any direct 8766 consumer before
running `closed_loop_test.py` or `tmai_cmd.py` against the plugin.

**Telemetry is capped by render framerate, not the physics tick.** Measured
**72.8 lines/s with 13-15ms gaps** despite the 100Hz setting, because the
coroutine yields once per frame. TM's physics runs at 100Hz, so we sample below
it. Fine for a 20Hz control loop, but don't assume `S_RateHz` is achievable —
measure with `telemetry_listener.py --stats`.

**`pkill -f` kills the shell running it**, because the pattern matches that
shell's own command line. Bracketing (`[t]elemetry`) does not help when the same
command line also launches the target. Kill by PID.

**Background `&` jobs inherit SIGINT ignored** from a non-interactive shell, so
"stop" signals silently do nothing. The line/replay tools install their own
SIGTERM+SIGINT handlers that exit cleanly and still write the file.

---

## Hard constraints

- The car's in-game name/plate must always read **TAS**, so any footage is
  self-evidently tool-assisted.
- Nothing from this pipeline is ever submitted to Nadeo's servers or
  leaderboards. School signature mode enforces this at the engine level.

## Phase plan

- [x] **Phase 0 — control.** uinput pad, confirmed live in-game (steer, gas on
      R2, brake on L2; L2 doubles as reverse when stationary, which is a real
      game mechanic, not a bug).
- [x] **Phase 1 — telemetry. Working end-to-end 2026-08-26.** 5850 lines, zero
      parse failures, 33 fields. Plugin serves position, velocity, orientation
      basis (Dir/Up/Left), speed, rpm, gear, inputs, adherence, skidding wheel
      count, flying duration, ground contact, per-wheel slip/material/damper,
      plus race context (UI sequence, race time, map uid, checkpoint count,
      respawns, finished flag) for episode boundaries.
- [x] **Closed-loop test.** `tools/closed_loop_test.py` drives the pad and
      watches the inputs return in telemetry: **7/7 steps, round-trip median
      31ms, max 54ms**. At a 20Hz control rate (50ms) an action's effect lands
      roughly one control step late, so the action delay belongs in the MDP.
- [x] **Reset / episode structure.** State machine on top of the plugin's
      command channel (`restart` / `goto` / `playmap`).
- [x] **Algorithm chosen: SAC**, not NEAT. The game runs at 1x wall-clock, so
      real seconds are the scarce resource: NEAT extracts one fitness scalar
      from a 30s run and discards ~2200 samples; SAC keeps and replays all of
      them. NEAT's one edge — discovering odd input patterns gradient
      exploration won't stumble into — is covered by hints + scripted warm-up.
- [x] **Spatial awareness.** No raycast API exists; block-occupancy grid parsed
      from map data, marched in Python — see [Lidar](#lidar-ground-rays-not-wall-rays).
- [x] **Single-instance training**, explore + race stages.
- [ ] **Scale to multiple instances.** Designed and partially built; blocked on
      the School-mode toggle needing a human, and on account provisioning. See
      [`FLEET.md`](FLEET.md) / [`FLEET_V2.md`](FLEET_V2.md).
- [ ] **Offline `.Ghost.Gbx` / `.Replay.Gbx` parsing** for harvesting top
      players' inputs as demonstration data. Folds into the Gbx work.

### Low resolution on workers (later)

Running workers at minimal resolution is worth doing, but not for speed:
TM2020's physics is locked to 100Hz wall-clock, so a 30s lap takes 30s at any
framerate. What it buys is **more instances per GPU** (raising the scaling
ceiling) and **a higher telemetry rate** — we under-sample the 100Hz sim at
72.8 samples/s today because the plugin yields once per frame, and more frames
closes that gap. Only the input ring (see [Open lead](#open-lead)) could break
the 1x wall-clock barrier.

## Dead ends (confirmed by API inspection, not assumption)

- **TMT SDK** (`TMT.Sdk.dll`, inspected with `dnfile`) — read-only telemetry and
  game-action hooks, no live control injection.
- **TMInterface** — has exactly the right API (`SimulationManager.SetInputState`,
  `Net::Socket`) but is Nations/United Forever only, not TM2020.
- **ViGEmBus** — Windows-only, can't run under Wine.
- **Ghost inputs via Openplanet** — `CGameCtnGhost` is metadata only
  (`Duration`, `RaceTime`, validation fields). The inputs are in the
  `.Ghost.Gbx` on disk.

## Open lead

TMT's creator (`nieqtv` on the TMT Discord) agreed to share their internal
physics hook — "the input ring" — used for TMT's own bruteforce. If it
materialises it likely replaces the uinput layer outright: tick-accurate, no OS
input-timing jitter, possibly fast-forwardable. The telemetry side wouldn't need
to change.

## Reference / prior art

[`tmrl`](https://github.com/trackmania-rl/tmrl) — the real working prior art for
TM2020 AI driving (SAC + LIDAR-from-screenshots + Openplanet telemetry +
vgamepad, strictly real-time). **Do not copy its 20Hz control rate** — 50ms per
action is fine for finishing a track and too coarse for timing-sensitive
technique, which needs input changes on the order of a few physics ticks.

**`SAC_GetData.op`** is already installed in this prefix — it unzips to
"TrackmaniaRL Connect" by Palamabron / AITrackmania, a working telemetry bridge
serving a 33-float binary stream on :9000 plus a JSONL readiness channel on
:9001. It's what our plugin is modelled on:

- Use `Main()` as a coroutine with `yield()`, not `Update(float dt)`.
- The plugin `Listen`s and the Python side connects, not the reverse.
- Prefer `CSmScriptPlayer` (`api.Speed`, `EngineRpm`, `EngineCurGear`,
  `AdherenceCoef`, `WheelsSkiddingCount`, `FlyingDuration`) over digging the
  equivalents out of `CSceneVehicleVisState`. Use VisState for what only it has:
  `Dir`/`Up`/`Left`, per-wheel slip/material/damper, ground contact.
- Its readiness handshake (`verify_loaded_map` / `confirm_ready`) is a good model
  for our episode-reset protocol.

Its header comment claims Openplanet has "no documented API to safely load an
arbitrary .Map.Gbx" — that is wrong (see
[Map control](#map-control-no-clicking-needed)). It is undocumented, not absent,
and the ManiaExchange plugin uses it in production.

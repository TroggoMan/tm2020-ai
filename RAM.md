# Reading vehicle state out of the game's RAM

**Status: working and measured, 2026-09-07.** `env/ram_state.py` locates the
vehicle struct with no help from anything, and `telemetry/ram_adapter.py`
re-serves it in the plugin's own JSON schema.

## Why

TMAITelemetry is unsigned → Openplanet **Developer mode** → a **paid** account.
That single chain is what stops the free fleet from producing telemetry
(FLEET_V2.md). `/proc/<pid>/mem` is subject to none of it. A free School-mode
instance running **no Openplanet plugin at all** now yields real vehicle state.

## What it gives you

Every field below was placed by correlating memory against live plugin
telemetry over varied driving, then re-confirmed on an independent run. Offsets
are from the struct base, which is where `pos` sits.

| offset | type | field | agreement with the plugin |
|---|---|---|---|
| `-0x230` | u8 | `finished` | 540/540 frames |
| `-0x2c` | u32 | start-of-run clock stamp | — |
| `-0x10` | f32[4] | rotation quaternion `(w,x,y,z)` | median 2e-6 per basis component |
| `0x00` | f32[3] | `pos` | median 1.5 mm |
| `0x0c` | f32[3] | `vel` | median 0.008 |
| `0x18` | u32 | absolute sim clock, ms (copy at `0x3c`) | `race_time` median 6 ms |
| `0x40` | f32 | `in_steer` | 95.6% exact |
| `0x44` | f32 | `in_gas` | 100% exact |
| `0x48` | u32 | `in_brake` | 100% exact |
| `0x50` | f32 | `rpm` | 85.6% exact |
| `0x54` | u32 | `gear` | 98.9% exact |
| `0x60..0x6c` | u32[4] | per-wheel ground contact, 0/1 | 99.4% |
| `0x80..0x8c` | f32[4] | suspension travel | 93.1% within 1e-3 |

Derived rather than stored, and matching the plugin to ~1e-4: `speed` = ‖vel‖,
`side_speed` = vel·left, `front_speed` = vel·dir, and the `left`/`up`/`dir`
basis from the quaternion.

`race_time` = `[0x18] - [-0x2c]`. The clock at `0x18` never resets; the stamp at
`-0x2c` is rewritten on every respawn.

**This struct is republished once per rendered frame, not per physics tick.**
Measured: 41 updates/s uncapped, 19/s with the frame rate capped to 20, and the
clock at `0x18` stepping 24 ms and 50 ms to match. The physics very likely does
step at 100 Hz underneath — TICK's ticks are 10 ms — but what is *visible* here
is frame-paced, and so is the gamepad read. That makes the frame rate the
observation rate: see "Frame rate is the control rate" in README.md before
capping it.

### Wheel order

RAM runs **FL, FR, RR, RL**; the plugin and the policy run **FL, FR, RL, RR**.
`RAM_TO_PLUGIN_WHEEL = (0, 1, 3, 2)`. Proven twice over: only the two front
wheels carry a steer angle in the per-wheel array, and a frame with exactly one
wheel off the ground pinned the rear pair.

### Quaternion handedness

The stored quaternion is the **conjugate** of the body→world rotation, and the
basis vectors are the **rows** of the resulting matrix, not the columns. Both
mistakes are nearly invisible on a straight line — two of three components agree
— and catastrophic in a corner: with columns, `left` picks up the forward
direction and `side_speed` comes out as the full speed. Conjugating and taking
rows drops the median error to 3e-6, which is the plugin's own print rounding.

## SUPERSEDED, 2026-09-07: use RL Connect for the surface block

Most of the gaps listed below were closed the same day, not by more memory
work but by a SIGNED plugin: **TrackmaniaRL Connect** (siteid 421, author
Palamabron / AITrackmania), which serves a 33-float stream on 127.0.0.1:9000.
`telemetry/sac_getdata_adapter.py` already parses it. Verified live at 60 Hz:

    1501 frames in 25.0s = 60 Hz
    cp_count            : [0, 1, 2, 3, 4]
    mat combos (17)     : (16,4,16,4)  (16,16,80,80)  (9,80,16,80)
    slip combos (25)    : intermediate values, wheels disagreeing
    flying (ms airborne): 0 .. 227

It reads them straight off `CSceneVehicleVisState` through the Openplanet API:

    AppendInt(buf, uint(vis.FLGroundContactMaterial));  ... FR, RL, RR
    AppendFloat(buf, vis.FLSlipCoef);                   ... FR, RL, RR
    AppendFloat(buf, api.AdherenceCoef);
    AppendFloat(buf, raceData.dPlayerInfo.NumberOfCheckpointsPassed);

So per-wheel `mat[4]`, `slip[4]`, `adherence`, airborne time and a working `cp`
count are all READABLE - do not repeat the scans below for them.

**But it is not a drop-in replacement for the fleet.** RL Connect calls
`VehicleState::ViewingPlayerState()`, and per README.md the `VehicleState` API
is **School-mode only**, and School mode **does not persist across relaunches**
and is recorded nowhere in Settings.ini. So every instance needs a human to
re-toggle F3 -> Signature Mode -> School after every launch. That is fatal for
a fleet that restarts games on its own, until the toggle is scripted.

RAM needs no plugin, no signature mode and no toggle, so it is the only source
that survives an unattended relaunch. Treat RAM as the baseline and RL Connect
as a richer layer on top, available when a human is present or once the click
is automated.

NOTE on the 60 Hz verification above: it ran on the PAID instance, whose
Settings.ini says DeveloperMode=true. Since the live signature mode is not
recorded anywhere, that run does NOT establish what a free School-mode account
does. Still untested.
The conclusion that per-wheel material is unreachable by value scanning still
stands and is still correct - it is just no longer worth caring about.

What RAM still uniquely provides: `damper[4]`, `finished`, and needing NO
plugin at all. Everything else in the surface/tyre block comes from 421.

A gotcha that cost real time: **Wine-side sockets are owned by `wineserver`,
not the game pid.** A `/dev/tcp/127.0.0.1/9000` probe reported the port CLOSED
while `ss -ltnp` showed it listening as
`users:(("wineserver",pid=...))`. `tools/stack.sh`'s `port_up()` uses the same
`/dev/tcp` trick, so it will mis-report any in-Wine port the same way.

## Per-wheel material and slip: a RING BUFFER, found 2026-09-07

They are not at a fixed address, which is why every earlier scan "failed".
The game keeps a ring of recent per-wheel frames - measured 46 entries, ~0.77s
of history at 60 Hz. Block geometry, proven by a joint constraint (see below):

    stride 44 bytes per wheel, wheel order FL, FR, RR, RL
      block -16 : f32  damper
      block  -4 : f32  steer_angle      (nonzero on the front pair only)
      block  +0 : u32  ground material  (EPlugSurfaceMaterialId, plugin numbering)
      block  +4 : f32  slip coefficient

How it was found, after single-key scans left hundreds of candidates: park the
car STRADDLED (front wheels on one surface, rear on another) and require
mat[4] AND damper[4] to match *in the same block, at a consistent stride and
wheel order*. Parked makes an 8 s full-memory scan coherent; straddled makes
both quartets asymmetric at once. 476,132 -> 46.

**The trap, and it caught me three times: a match while PARKED proves nothing.**
Stationary, every ring entry holds identical values, so a joint 8-value match
selects all 46 equally and not one of them tracks under motion (~30% each).
Intersecting two different straddles does NOT help either - the whole ring
updates, so all 46 survive. Only a motion test discriminates.

**The measurement that settled it:** while driving, ask whether AT LEAST ONE
block holds the live material. 996 samples, **100%**. No single entry tracks;
the set always contains the current frame.

So to read it: use the LIVE `damper[4]` at `struct+0x80` (which this file
already documents, and which needs no plugin) to select the ring entry whose
damper matches, then read material at +0 and slip at +4 of that entry. That
makes per-wheel surface and slip available with no plugin and no School-mode
toggle - the whole point for the free fleet.

STILL TO DO: the ring's base address needs a locator like the vehicle struct's,
and the entry count (46 here) should be confirmed rather than assumed.

## What it does NOT give you

- **`cp` / `lap`** — reads 0, and that is **correct, not a gap.** The game's
  `RaceWaypointTimes.Length` reads 0 in Time Attack however many checkpoints
  have been passed; our own plugin says so in a comment and the README has a
  section on it ("Checkpoints: count them yourself"). `env/tm_env.py` never
  reads `cp` off the wire — it counts gate-plane crossings from `pos` against
  `CSmArena.MapLandmarks`. So the RAM path emits exactly what the plugin emits,
  and the env derives the real count from a position this path already gives to
  the millimetre. Nothing to find. (Measured the hard way: 672 samples over
  several laps with checkpoints genuinely crossed, `cp` never left 0 — because
  it cannot.)
  What the survey still needs is the **landmark positions**, and those come from
  the plugin's `landmarks` command on the privileged instance, cached to disk.
- **ground contact material (`mat[4]`)** — narrowed from 685M slots to **6
  addresses**, but not yet usable. Method that worked, after the +/-8KB window
  round the car was (rightly) challenged as an unjustified assumption: park the
  car on a surface and scan ALL 2.6 GB, then intersect across surfaces. Parking
  is what makes it work - a full scan takes ~8 s and `mat` changes several times
  a second while driving, so a moving scan is coherent with nothing.

  | round | surface | value | survivors (u32) |
  |---|---|---|---|
  | 1 | road | 16 | 476,132 |
  | 2 | grass | 2 | 198 |
  | 3 | dirt road | 8 | 190 |
  | 4 | water | 13 | **6** |

  The RAM values use the plugin's own `EPlugSurfaceMaterialId` numbering - no
  translation table. Two traps this turned up:

  * After round 3 there were 45 clean clusters of 4 addresses at **stride 44**,
    which matched the per-wheel block already proven elsewhere (damper +0,
    steer_angle +12, material +16, slip +20). Every one of them was ELIMINATED
    by the water round. Stopping at three surfaces would have recorded a
    confidently wrong offset. Four distinct materials was the minimum.
  * The 6 survivors were then split by a STRADDLED PARK - front wheels on one
    material, rear on another, stationary. Two of them (`0x3fdacb4ec`,
    `0x3fdacb5fc`) turned out to hold garbage the moment the value changed:
    they had matched four parked scans by coincidence. The other four track it
    every time.

  **Result: a single contact-material scalar, four redundant copies, NOT a
  per-wheel array.** With telemetry reading `(22, 22, 6, 6)` all four addresses
  read `6`. The per-wheel `mat[4]` was not found: none of the 45 stride-44
  clusters matched that multiset either - they hold stale values from surfaces
  driven over minutes earlier (`[16,16,16,16]`, `[5,5,5,5]`, garbage), so they
  are leftovers or other vehicles, not live wheel state.

  What that costs: the 24-dim surface one-hot can be filled from one material
  for all four wheels - right whenever the car is fully on one surface, wrong
  while straddling an edge. Better than inferring asphalt-or-air from the
  contact flags, but not the real per-wheel signal.

  **Offsets are per-BUILD, not per-map.** These four addresses survived a map
  load, and the vehicle struct's offsets held across four maps and a second
  process at a different base. Only the base address changes per launch, which
  is what locate() is for.

  **Per-wheel `mat[4]` is a dead end for value scanning, concluded 2026-09-07.**
  With the car parked STRADDLED (front wheels on one material, rear on
  another - `mat = [6,6,22,22]`), the whole 2714 MB was scanned for four
  material values at every stride from 4 to 128 bytes, in both the wheel order
  (FL,FR,RR,RL) and the plugin order. Every small-count candidate failed
  verification: the one stride-36 hit - exactly the block size the class dump
  predicts - had the right four materials and 0.0 in all 32 other bytes of each
  block, i.e. four numbers in a zeroed region.

  The same scan keyed on the per-wheel DAMPER pattern (asymmetric floats, far
  more distinctive) immediately re-found `struct+0x80` at stride 4 - so the
  method is sound, and the negative result is real.

  Conclusion: the per-wheel material is almost certainly a POINTER to a
  material descriptor, which Openplanet resolves to an enum index via its class
  table. A value scan cannot find it; only pointer-chasing would. The car-level
  scalar is the cached enum and is separate.

  Practical alternative worth trying FIRST: `ezio416/tm-current-surfaces`
  reads per-wheel surfaces through the normal Openplanet API. If it is
  index-SIGNED it runs in School mode on a free account - which would give the
  fleet per-wheel surfaces the same way SAC_GetData gives it telemetry, with no
  reverse engineering at all. Check the Openplanet index before spending more
  time in memory.

  Also do not repeat this mistake: a float comparison of the form
  `abs(v - want) > tol` is FALSE for NaN, so NaN passes as a match. Four
  "verified" hits were all NaN before that guard was added.

- **`wetness` / `submerged`** — findable now, not yet hunted. Both were
  constant 0 in every earlier run, so nothing could be correlated. Driving
  through water gives `wetness` (`WetnessValue01`) a real series - 0.07, 0.34,
  0.63, 0.90, 1.0 - and `water` (`WaterImmersionCoef`, the `submerged` obs dim)
  reads -1.0 out of water and 0.8..1.0 in it.

- **`icing` / `wear` / `dirt`** — still constant 0 everywhere, including across
  road, grass, dirt road, sand and water. `Ice_trainer.Map.Gbx` is the untested
  map for icing; dirt presumably needs accumulated distance on a dirt surface.
  Per the wheel-block layout above, the slots right after `slip` (+24, +28,
  +32) are the natural candidates, but +24 reads a constant ~0.2944 which is
  not icing.

- **`ground_dist`** — no signal to correlate against: the test maps are walled,
  so the car never left the ground and `ground_dist` spanned 0…0.073 all
  session. Every high correlate was something tracking ride height. `contact[4]`
  covers the airborne case. Retry on a map with a jump.
- **`slip[4]`, icing, wear, dirt, `adherence`, wetness, turbo/reactor/cruise** —
  constant across every run. `slip` in particular saturates to exactly 0.0 or
  1.0, so it carries about one bit; several 0/1 arrays in the struct track it at
  r=1.0 and disagree on ~6% of frames, which is inside telemetry staleness. Not
  enough to call placed. On a plain road map these are all genuinely inactive,
  which is the same argument FLEET_V2 makes for the SAC_GetData adapter.
- **`ahead`, `offset`, `fx_ahead`, `lidar`, edge awareness** — these were never
  vehicle reads. They are computed from surveyed map geometry, which is what the
  plugin's `dumpmap`/`landmarks` commands are for. That survey is a one-off on
  the privileged instance, cached to disk, and reused. RAM neither can nor
  should replace it.
- **`playmap`** — loading a map is a plugin command. A free instance still needs
  its map chosen by hand or by menu navigation. Same gap the SAC_GetData route
  has.

## Finding the struct

Two locators, both in `env/ram_state.py`:

- `locate_structural(pid)` — **no telemetry at all.** Screens the whole address
  space for the struct's own shape (a unit quaternion, a sane position, rpm and
  gear in range, 0/1 contact flags, suspension in 0…2 m, inputs in range), then
  confirms survivors by physics: over ~70 ms, position must advance by velocity.
  Returns exactly one address. ~75 s.
- `locate(pid, telemetry_fn)` — the same shape screen, seeded by a plugin frame.
  ~30 s. For when the plugin is there anyway.

Both need the car **moving**. `ram_adapter.py` holds the gas itself while
locating rather than requiring someone to drive.

### The two traps that cost the most

1. **A whole-address-space scan takes 30–75 s.** A car at 45 m/s is two
   kilometres away by the end of it, so one seed position taken up front matches
   nothing in the regions read last — and it fails as *zero hits*, not as an
   error. `locate()` re-seeds per region; `locate_structural()`'s screen is
   position-independent on purpose.
2. **A contradiction is not proof of a wrong candidate.** Confirming by
   `pos += vel*dt` breaks for the *real* struct too whenever the car hits a wall
   inside the interval. Scoring (mostly-passing wins) finds it; a single-strike
   rule throws it away.

## Run control without the plugin

- **respawn / give up**: pad button `b` (or `y`). Works mid-race.
- **the finish screen**: no gamepad button dismisses it — `b`, `y`, `a`,
  `start`, `select` were all measured doing nothing. **Enter over XTEST** does,
  because the game lives on a plain Xvfb. That keystroke is the only reason
  `restart` works without the plugin.
- The adapter implements `restart` on its command channel using both.

## Using it

```
# one instance, telemetry on :8776 in the plugin's schema
python3 telemetry/ram_adapter.py --serve-port 8776 --pad-port 8775 \
        --display :100 --map-uid <uid>
python3 telemetry/broker.py --upstream-port 8776 --port 8777
```

`tools/stack.sh up --school` now takes this path by default;
`--school-sac` keeps the old signed-plugin route.

Reading **another user's** `/proc/<pid>/mem` needs `CAP_SYS_PTRACE`, so the
adapter runs under `sudo` for a fleet instance. It is the only part of the stack
that does, and it only ever reads.

## Proven on a free instance

2026-09-07, end to end, with **no Openplanet plugin of ours anywhere**:

- `tools/steam-instance 1 --vnc` → tmai01's game on `:100`, account
  **TAS01-TROGGOBOT, Starter Access**.
- `sudo .venv/bin/python telemetry/ram_adapter.py --user tmai01 --pad-port 0 \
   --serve-port 8779 --display :100` located the struct at `0x184d046c` on its
  own and served coherent frames: race_time resetting on respawn, 0 → 62.7 →
  118.4 km/h with the gear going 1 → 2, `ground` going false over a jump, gas
  1.00 while accelerate was held and 0.00 the instant it was released.
- Different process, different base address, **same offsets**. They are struct
  offsets, not a lucky allocation.

### Two things that instance needs and the dev one already had

- **A window manager on its display.** Without one nothing has X input focus, so
  the game never sees a keystroke — while mouse clicks, which are delivered by
  position, work fine. That combination reads as "the game ignores the
  keyboard", and it is not: it is `openbox` missing. `stack.sh` starts one;
  `tools/steam-instance` on its own does not.
- **Its own pad, or keyboard control.** The free instance answered the keyboard
  immediately (Up = accelerate, Delete = respawn); whether it also grabs one of
  the fleet's uinput pads is the usual seat-binding question and is not settled
  here. `--pad-port 0` disables the pad path, at the cost of `restart`.

### The wall that is NOT telemetry

A **Starter (free) account cannot play a local custom track at all** —
`PLAY → LOCAL → PLAY A TRACK` opens the "TRACKMANIA ACCESS" upsell and stops.
Local custom maps need **Club Access**, a paid subscription, on every account.

So the RAM reader removes the *telemetry* licensing constraint completely, and
leaves a different one standing: the free fleet can only drive **Campaign,
Weekly Tracks and Track of the Day** — not our test maps. That is not fatal
(the current training map is a Summer 2026 campaign track, which every free
account has), but any plan that assumed free instances would run our own maps
needs revisiting. See FLEET_V2.md.

## Not solved: stepping the physics

This is a **read** path. It does not pause, step, or rewind the simulation.
TICK does that from an injected DLL inside the game's tick loop
(`TickNative.dll`); nothing here approaches it, and `/proc/<pid>/mem` is the
wrong tool for it. See TICK.md.

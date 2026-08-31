# Block name -> surface, from driving it

Evidence collected by driving a surface test track (`lNbYztV6fMH_XsqXetu6a9PnFX`,
883 m straight strip, spawn (1520,10,784) -> finish (637,10,784)) and reading
`tools/whereami.py`, which prints the MATERIAL the game reports per wheel next
to the BLOCK NAME from the occupancy dump.

Why both: they answer different questions and they disagree. The material is
physics (grip, how it slides). The block name is geometry (does it have
barriers, is there ground beside it). `env/lidar.is_drivable()` classifies by
NAME and decides what enters the occupancy grid - which feeds the LIDAR and the
racing-line tracer. A block wrongly marked not-drivable is not reported as
"slow ground", it is reported as *nothing there at all*.

## Observed

Driven and read directly (parked on the block, all four wheels reporting)
unless marked "not driven".

| block | material | our `is_drivable` | correct? | notes |
|---|---|---|---|---|
| `RoadTechStart` | Asphalt | True | yes | road **with** barriers |
| `RoadTechStraight` | Asphalt | True | yes | road with barriers |
| `PlatformTechBase` | Asphalt | True | yes | platform, **no edges**, drop-off |
| `OpenTechRoadStraight` | Asphalt | True | yes | thin road, **no barrier**, penalty grass both sides |
| `RoadDirtStraight` | Dirt | True | yes | material DOES change - car can perceive this one |
| `PlatformDirtBase` | Dirt | True | yes | same material as above, no edges |
| `PlatformIceBase` | **RoadIce** | True | yes | block says Ice, material says RoadIce |
| `PlatformPlasticBase` | Plastic | True | yes | |
| `Grass` (terrain) | Grass | False | yes | the map's canvas; 2304 of 2335 blocks |
| `RoadWaterStraight` (channel) | **Water** | **False** | **NO** | parked in it with 4 wheels down |
| `RoadWaterStraight` (its lip) | **Rubber** | **False** | **NO** | the raised border; Rubber = kerb |
| `PlatformGrassBase` | **Green** | **False** | **NO** | parked on it with 4 wheels down |
| `PlatformWaterRampBase` | — | **False** | **NO** | not driven; expect same as RoadWaterStraight |
| `PlatformPlasticFinish` | — | True | yes | not driven |
| `RoadIceStraight` | — | True | yes | not driven |

## Names and materials do not correspond - read both

* `PlatformIceBase` reports **`RoadIce`**, not `Ice`.
* `PlatformGrassBase` reports **`Green`**, the open road's verge reports
  **`Grass`**. So `Green` is the drivable grass SURFACING and `Grass` is
  TERRAIN. That is a far better rule than any substring match - and it means
  the `DRIVABLE_HINTS` missing-comma bug is nearly a red herring, because
  `NOT_DRIVABLE` is checked first and would reject `PlatformGrassBase` anyway.
* Wheel order is confirmed **FL, FR, RL, RR** - user reported right-side tyres
  on the border and indices 1 and 3 read `Rubber`.

## Rubber is the kerb, and the policy cannot see it

`Rubber` is the road border - "generally we don't want the cars to touch it"
(user). It is the ONE fine-grained per-wheel "you are at the edge" signal the
game provides. `env.surfaces` groups it as **`road`**, identical to `Asphalt`,
so the 36-dim surface observation collapses kerb and racing line into the same
slot and throws the signal away. Giving Rubber its own group would let the
policy learn to avoid the edge from something it can actually perceive.

## Platform blocks have no sides. Road blocks do - unless they are Open*

User's rule, and it holds on both surveyed maps. The one exception is
`OpenTechRoadStraight` (road-named, no barrier, grass verges), so it cannot be
a plain prefix test.

**This explains the CP4 -> CP5 failure on the training track exactly.** Every
edgeless block on that entire circuit sits between those two checkpoints:

```
  [31] CHECKPOINT #4   RoadTechCheckpointTiltRight
       >>> 7 PLATFORM blocks (NO SIDES)
            ToRoadTech, Curve2In, BaseOnLandHill3, BaseOnLandHill3,
            Curve3In, BaseWithHole24m, ToRoadTech
  [44] CHECKPOINT #5   RoadTechCheckpoint
```

The other 43 blocks of the walk are `Road*` with barriers. So the car spends
four fifths of the lap learning that going wide is survivable, then meets seven
consecutive no-barrier pieces - one named `BaseWithHole24m` - with nothing in
its observation saying anything changed. Measured: reaches CP4 78% of episodes,
CP5 never.

## The rule this implies

**Bare terrain is not drivable; a road/platform PIECE with that surface is.**

`is_drivable()` currently matches the material word anywhere in the name, so
`"grass"` in NOT_DRIVABLE rejects `PlatformGrassBase` exactly as it rejects the
`Grass` canvas block. The piece prefixes (`Road…`, `Platform…`, `OpenTech…`)
have to be checked BEFORE the terrain keywords.

**User's point, and it is the reason this matters:** penalty grass is drivable,
just slow, and *sometimes you have to cross it to drive a track at all*. A car
told there is no ground there cannot plan to cross it.

## Related, already found

* **Missing comma bug** in `env/lidar.DRIVABLE_HINTS`: `"plastic", "grass"` and
  `"core", "base",` on consecutive lines with no comma between `"grass"` and
  `"core"`, so Python concatenates them to `"grasscore"` - neither `"grass"`
  nor `"core"` is actually a drivable hint.
* **`Green` (id 76) vs `Grass` (id 2)** are BOTH real and both used on this
  map - `Green` beside a Stadium road, `Grass` on the open-road strip. Both map
  to the `grass` physics group correctly, but `Green` was missing from
  `env.surfaces.TUNABLE`, so the panel could not tune the one you actually
  slide onto. Fixed.
* **Material cannot distinguish geometry.** `RoadTechStart` (barriers),
  `PlatformTechBase` (drop-off) and `OpenTechRoadStraight` (grass edges) all
  report plain `Asphalt`. Three completely different consequences for going
  wide, one identical observation. `roadtrace` does assign different widths
  (Platform 8 m, Road 6 m) as `_track_hw`, but **`_track_hw` is never fed to
  the policy** - it appears twice in `tm_env.py`, neither in `_observe`. That
  is a strong candidate for the CP4->5 platform failure on the training track.

## Wet wheels: PLASTIC DOES NOT SHED WATER

Measured 2026-08-31, ~1450 samples: drove out of a water block onto
`PlatformPlasticBase` with `wetness = 1.0000` and it stayed at **exactly
1.0000 for 19.3 s** at 20-42 km/h. Not decaying slowly - pinned, no movement in
the fourth decimal. **User confirms: there is no water decay on plastic.**

So the per-surface wet interaction the FLEET_V2 plan wanted calibrated is real,
and plastic behaves like the wood case in those notes ("wet dries fast on dirt,
holds on wood"). First measured constant of that table.

Telemetry fields for this (per car, all present and live):

```
  wetness   1.0      icing  [0,0,0,0]     adherence 1.0
  water     1.0      dirt   [0,0,0,0]     slip      [0,0,0,0]
```

**MEASURED, same session** - a high-speed pass across every strip settled it:

| surface | d(wetness)/dt | dries from fully wet in | samples |
|---|---|---|---|
| **Dirt** | **-6.99 /s** | **0.14 s** | 7 |
| **Green** (grass) | **-0.115 /s** | **8.7 s** | 47 |
| Plastic | 0.0000 | never | 2171 |
| RoadIce | 0.0000 | never | 58 |
| Water | 0.0000 (re-wets on entry) | never | 381 |

Dirt sheds water ~60x faster than grass. This also **rules out** the worry that
`wetness` might be a zone flag rather than real per-wheel water: it decays on
some surfaces and not others at wildly different rates, so the observation's
`wet` dim is genuine and worth tuning against.

Caveat on the numbers: **Dirt's -6.99/s rests on only 7 samples** - the whole
transition is over inside 0.14 s at ~50 Hz telemetry, so read it as
"essentially instant" rather than trusting the figure (fastest single sample
-13/s). Plastic's zero is solid: 2171 samples, ~58 s, up to 101 km/h, never
moving off 1.0000, so it is not a speed threshold either.

Also seen in the same capture: `Concrete` (grouped `road`), and
`Plastic + Rubber` when clipping the kerb - Rubber-as-border again.

## Reward side - deliberately leave alone

`terminate_on_surface: []` and `surfaces: {}` in every current config, so
nothing punishes grass and no surface ends an episode. That is right: the real
cost of grass is that it is slow, which `progress` already charges for. An
earlier config had `Grass: -9.2`, which would teach the car that a sometimes
necessary line is catastrophic.

## Snow is the ice penalty surface

User: *"snow is to ice what penalty grass is to grass. It's the ice based
penalty surface."* Confirmed the code already agrees - `Snow`, `Ice` and
`RoadIce` all map to the `ice` physics group, exactly as `Green` and `Grass`
both map to `grass`. Nothing to change.

So the penalty/surface pairs are:

| driving surface | its penalty verge | shared group |
|---|---|---|
| `RoadIce` | `Snow` | ice |
| `Asphalt` | `Green` / `Grass` | road / grass |

## Which telemetry fields actually carry information

Measured over 6359 samples of a human driving a multi-surface track with jumps,
then re-checked at 20 328 samples. **Do not add the constants** - an input that
never changes is parameters to train for no information.

| field | verdict |
|---|---|
| `damper` (4), `steer_angle`, `skidding`, `top_contact` | VARY - added |
| `ground_dist`, `flying` | VARY - added as `air` |
| `dirt`, `icing`, `air_brake` | vary, already in the observation |
| `adherence` | **CONSTANT 1.0** - and it is ALREADY an observation dim (`adher`). A dead input that has been there all along. Left in place only because removing it shifts every later index. |
| `brake_coef`, `sim_coef`, `wear`, `turbo` | CONSTANT - deliberately excluded |
| `water`, `wetness` | constant on THIS track (no water on it); genuinely informative elsewhere |
| `pos` | never add. Absolute world position teaches this track's coordinates instead of driving; everything spatial goes through `car_frame` first. |

### `flying` is MILLISECONDS

Max 3990 on a four-second jump, exactly 0 while grounded. A first cut divided
it by 3 as if it were seconds, which would have pegged that input at its
ceiling on every hop. Caught only because a human drove real jumps.

### `omega` is differenced per CONTROL STEP, not per telemetry frame

Raw telemetry arrives ~1 ms apart; `_observe` runs once per 50 ms control step,
so `_omega_obs` differences 50 ms-spaced orientations and `self.dt` is right.
Differencing raw 1 ms frames instead gives nonsense (469 rad/s peak) because
the rotation over 1 ms is noise. At the real 20 Hz: pitch never approaches the
+-3 clip, and only 5 frames in 4131 (0.12%) clip at all, on yaw and roll.

# Fleet v2 — the privileged-surveyor architecture

Status: **design + partial build, 2026-08-28.** Phase-0 (single paid instance,
explore stage) is unchanged and works today. Everything below is how the fleet
scales past one account without buying one paid licence per car.

---

## The constraint that forces this

Our telemetry plugin is unsigned → it needs Openplanet **Developer mode** →
Developer mode needs a **paid** account → free/Starter accounts get **School**
mode → the free fleet accounts cannot run our plugin at all.

Index-signing is blocked (ToS bans mostly-AI-generated plugins). An exception
request to Openplanet staff is **in flight, outcome unknown** — v2 is built so
that outcome doesn't gate progress:

- **exception granted** → revert to our plugin everywhere; the adapter below
  becomes a fallback, splitscreen + explore + full 106-dim obs on every account.
- **denied** → the adapter is permanent; a second paid account (~£20) becomes a
  *targeted* buy for explore + lidar on a second track, not a blocker.

## The way round: SAC_GetData

`SAC_GetData` / "TrackmaniaRL Connect" (siteid 421) is **signed**, confirmed
loading in School mode, and serves a fixed **33-float** binary stream on
`127.0.0.1:9000` + a JSONL readiness channel on `9001`.

`telemetry/sac_getdata_adapter.py` reads that stream and **re-serves it as
exactly the newline-JSON our plugin speaks**, on the broker's upstream port
(8766). So `telemetry/broker.py` and `env/tm_env.py` need no transport change —
downstream it looks like our plugin with fewer fields populated.

    python3 telemetry/sac_getdata_adapter.py --serve-port 8776   # per instance
    python3 telemetry/broker.py --upstream-port 8776 --port 8777

`--mock` synthesizes frames so the wire path is testable with no game/account.

### What survives, what does not

| group | dims | v2 source |
|---|---|---|
| speed, vel, up, gear, rpm | 9 | direct from stream |
| lookahead + offset | 19 | position + a **recorded** line (race stage only) |
| slip, adherence | 5 | direct |
| surface one-hot (grip group / wheel) | 24 | from the 4 ground-material enums |
| side speed | 1 | `dot(vel, cross(up, dir))`, computed in the adapter |
| prev action | 3 | ours |
| ground contact | 1 | `flying_duration == 0` (approximate) |
| **icing / dirt / wet** | 9 | **0** — correct on plain maps; RECONSTRUCT elsewhere |
| **turbo / reactor / cruise / slowmo / airbrake** | 11 | **inactive** — same |
| **fx_ahead** | 8 | needs `dumpmap` → SURVEY |
| **lidar** | 16 | needs occupancy grid → SURVEY |

On a plain road/asphalt test map every "lost" dim is genuinely 0/inactive for
the whole run, so the observation is **complete there with the adapter alone**.

## Determinism: placement is reusable, state is not

Trackmania physics is deterministic *given identical inputs from an identical
start*. The RL car drives different inputs every run, so:

- **invariant** (survey once, reuse): where walls / boost pads / reactor ramps
  / cruise gates / slow-mo zones / ice patches / surface regions are.
- **not invariant** (evolves along the trajectory): turbo time remaining,
  reactor timer, icing level, dirt buildup, wetness, airborne-ness.

The dynamic state is **re-simulated in the adapter** from (static trigger map)
+ (live position/velocity/time) + calibrated state-machine constants. Those
constants are ~global for the stadium car, with a small per-surface
interaction table (wet dries fast on dirt, holds on wood, medium on road →
`dry_rate[surface]`, `ice_fall[surface]`, `dirt_gain/loss[surface]`, ~20
numbers). Calibrate once, the first time a map with real effects is trained.

## Obs layout decision

Keep training the "good driver" against the **full 106-dim layout** (what our
plugin already emits, with honest zeros on plain maps). The adapter targets the
same layout — zeros now, reconstructions later. **No model reset ever.** Only
an interim SAC_GetData-native 62-dim layout would force a reset; skip it.

## Per-track lifecycle (`train/orchestrator.py` — skeleton)

| stage | account | does |
|---|---|---|
| **SURVEY** | privileged | offline Gbx parse + 1 scripted lap → `maps/<uid>.{occupancy,materials,effects}`, append constants to `calibration.json`. Idempotent. |
| **EXPLORE** | privileged | full obs, explore stage, landmark line → handover → `lines/<uid>-explored.json` |
| **RACE_BULK** | free fleet ×N | SAC_GetData + adapter, race stage on the explored line. Throughput. |
| **REFINE** | privileged | on a materially-new best line: one scripted survey pass along it, refresh effects, resume. Per-line, not a live loop. |
| **DONE** | — | splits plateau → freeze, next track |

The privileged account is a **scheduled shared resource** (SURVEY/EXPLORE/
REFINE for whichever track needs it; idle → one more full-obs race instance).
The free fleet only ever runs RACE_BULK. "Racer is confused" is a training-knob
problem (shaping, hints, time), **not** a reason to hand back to the surveyor.

## Controllers: routing N games to disjoint pad sets

XInput exposes ≤4 controllers per process, and Wine shows every game every
`/dev/input` node → two games on one box both grab the same first 4.

- **Pool, don't hotplug:** `control/virtual_pad_server.py --pool 16` — one
  process, 16 pads on ports 8765 step 10, started ONCE before any game. Adding
  a seat/instance later = use an already-present pad. A game never enumerates a
  device that appears after it starts, and restarting the pool re-orders
  devices + makes Steam re-grab.
- **Route by ownership:** `deploy/99-tmai-pads.rules` — udev assigns each pad
  to the linux user whose game owns it (troggoman 0-3 + 12-15, tmai01 4-7,
  tmai02 8-11). A game as `tmai01` can `open()` only its four; winebus skips
  the rest with EACCES. Kernel-enforced, matches the fleet's existing per-user
  isolation, survives Proton updates.
- **Fallback for same-user instances** (shouldn't occur — one login per
  account): `--distinct-pid` gives each pad product id `0x028e+i`, launch each
  game with `PROTON_USE_SDL=1` +
  `SDL_GAMECONTROLLER_IGNORE_DEVICES_EXCEPT=045e/028e,045e/028f,…`. Untested
  that TM2020-under-Proton honours the SDL hint; a non-standard pid also risks
  winexinput not treating the pad as XInput. Off by default.
- **Steam Input OFF** per game regardless.

## Open test items (need a School game up + pads restarted between runs)

1. Adapter end-to-end against a real `:9000` — one frame decoded, obs matches a
   full-plugin reference on the same map within noise.
2. `SAC_GetData.op` actually loads from a **file copy** in School mode (vs a
   fresh Plugin-Manager install).
3. Pad pool + udev ownership: game as `tmai01` sees exactly pads 4-7 as XInput.
4. Data Sender vs SAC_GetData for the multi-instance case (configurable port).
5. Decay/interaction constants: measure `dry_rate`/`ice_fall`/`dirt` per
   surface; confirm they're car-global not track-variant.
6. Re-simulation drift: adapter turbo/reactor countdown vs the game's, one map.

## Built this pass (2026-08-28)

- `telemetry/sac_getdata_adapter.py` — adapter + `--mock`
- `env/mapnames.py` + `maps/names.json` — uid↔alias, display only, not yet
  wired into the panel
- `control/virtual_pad_server.py` — `--pool N`, `--distinct-pid` (additive; the
  running 4-pad setup is untouched, takes effect next clean start)
- `deploy/99-tmai-pads.rules` — not installed
- `train/orchestrator.py` — dry-run skeleton, no stage execution wired
- `web/server.py` — fixed `NameError: MODEL_DIR` that blocked every
  start-from-scratch (resume unticked) from the panel

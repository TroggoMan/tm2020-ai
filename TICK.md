# TICK integration notes

Investigated 2026-09-06. TICK is the TAS tool in `~/.local/share/tick`
(launcher `TickLauncher`, data service on :46321, launcher API on :46322).

Status key: **[V]** verified empirically this session, **[?]** inferred, not proven.

## Why it matters for the AI driver

- **[V]** TICK injects via `CreateRemoteThread` (`TickInjector.exe`), NOT the
  `dinput8.dll` loader slot. It therefore **coexists with Openplanet** - our
  TMAITelemetry plugin kept running with TICK injected into the same process.
- **[V]** `injectionBackend: "Trackmania Wine/Proton environment"` - it finds
  and injects into the Proton game with no extra configuration.
- **[V]** It does NOT give richer telemetry than our plugin. Every field
  recovered from TICK's state blob was located *by* correlating against our
  plugin's output. The value is a capability we lack: save/restore of full
  simulation state, plus deterministic seeding.

## Architecture

Injected DLL <-> HTTP data service <-> UI. The DLL is a **worker on a command
queue**: POST enqueues a command (HTTP 201, `status: pending`), the DLL claims
it (`claimedBy: tick-native-<pid>`), executes in-game, writes back `result`.
Poll `GET /api/runtime/commands?limit=N&newestFirst=true`. **[V]**

Auth: header `X-Tick-Data-Service-Key` (data service), `X-Tick-Launcher-Session`
(launcher). **Both are regenerated on every launcher start** - always re-read
`~/.local/share/tick/ui/runtime-config.js`, never hardcode. **[V]**

`/events` websocket (`?afterSequence=N&accessToken=K`) is a durable sequenced
**change feed**, not telemetry: it says *what resource changed*, you then GET
it. `afterSequence=0` replays history from previous sessions. **[V]**

There is **no live vehicle-state read path**. The DLL POSTs to
`/api/runtime/vehicle-state` but that is a write-only sink (GET returns 405).
Car state is only obtainable via state clips (recorded, 100 frames). **[V]**

## Endpoints that matter

    GET  /api/runtime/status              currentMapUid, effectiveGameSpeed,
                                          loadedInputRevisionId, loadedActionCount,
                                          inputExecutionEffective
    GET/PUT /api/runtime/game-speed       {requestedGameSpeed: N}
    GET/PATCH /api/settings               needs {expectedRevision: <current>}
    POST /api/runtime/capture-state-clip  {expectedMapUid}        -> runtimeClipId
    POST /api/runtime/persist-state-clip  {runtimeClipId, expectedMapUid} -> stateClipId
    POST /api/runtime/replay-state-clip   {runtimeClipId, expectedMapUid}
    POST /api/input-collections           {parentId, name, mapUid, mapName}
    POST /api/input-collections/{id}/revisions?origin=X
                                          raw body, Content-Type text/plain
    GET  /api/input-revisions/{id}/source returns the TAS script as text
    POST /api/input-revisions/{id}/load?expectedCollectionRevision=N

Optimizer (`/api/runtime/start-optimization`, `/api/optimization/settings`) is
the TAS brute-forcer: random mutation of 1-15 inputs in a tick window, scored by
a declarative objective (`minimizeCheckpointTime` plus null-able conditions on
speed / yaw / pitch / roll / xyz / wheel contact / velocity / cuboid / trigger).
**It takes no external reward function and returns no trajectories, so SAC
cannot be plugged into it.** The frozen-game background simulator lives behind
this and is not otherwise reachable. **[V]**

## Input format - SOLVED, no RE needed

Text (what `/source` serves and the revisions endpoint accepts):

    0 seed 12345
    0 accel 1
    1000 steer 60
    5000 accel 0

Times are **milliseconds**. Binary form in `input_action_chunks.bytes` is
13 bytes/record: `[int32 tick][int32 reserved][uint8 type][int32 value]`,
type 1=accel 2=brake 3=steer 4=seed. **tick = ms / 10** (100 Hz). **[V]**

**Steering is +/-127**, and it maps exactly onto our normalized +/-1.0:
`steer 60` produced `in_steer 0.472` (60/127). Sits well with `steer_levels`.
**[V]**

## State clip format - PARTIALLY solved

Container: magic `TICKCLP1`, int32 version, length-prefixed map UID, then
checkpoint_bucket / race_time / lap_time (all matching the `state_clips` DB
columns), then the frames. Uncompressed, ~2.8 bits/byte. **[V]**

Field offsets **within a frame**, validated against live TMAITelemetry: **[V]**

    +68 .. +80   quaternion (4x f32)
    +84 .. +92   position   (3x f32)
    +96 .. +104  velocity   (3x f32)
    +108 .. +116 angular velocity (3x f32)  [?] plausible magnitudes only
    +64          NaN sentinel

**The container geometry is NOT solved.** Blob sizes differ between clips
(243651 vs 243642 bytes for the same map and frame_count=100), so the header is
variable-length. Do NOT assume a stride.

### The trap - read this before touching clip parsing

I twice "confirmed" a layout by finding (header, stride) that divided the blob
length exactly, and both times it was coincidence: the parse landed mid-struct
and produced garbage. On a 243 KB blob many pairs divide evenly. Worse, two
runs parsed with the same wrong offsets yield *identically* wrong data, which
looks exactly like a perfect match and silently fakes a passing test.

**Only correlation works.** Read a known value from TMAITelemetry, scan the blob
for that float, and derive the stride from the spacing of the hits.

## Gotchas

- `loadInputRevision` 400s without `?expectedCollectionRevision=<collection revision>`.
- Loading a revision does nothing unless `settings.activeInputCollectionId` is
  that revision's collection - the runtime silently stays on the old one.
- **Input playback only engages when the run is at tick 0.** Mid-race it reports
  `inputExecutionEffective: false` with `lastErrorCode: null` - no error at all.
  Restart the map (`tools/headless-main.sh map "<path>"`) to arm it.
- `game-speed` PUT applies asynchronously; poll `runtime/status` for
  `effectiveGameSpeed` rather than reading straight back.
- Running `TickDataService` directly (not via the launcher) writes
  `data/tas-data.db` into the install dir, which then **fails the launcher's own
  manifest check on next start** ("bundle contains an unlisted file"). Launch via
  `TickLauncher`, which points the DB at `~/.local/share/tick-tas-tool/data/`.
- `pgrep -f <name>` matches its own shell command line here; use `pgrep -x`.

## Game speed

**[V]** Real simulation acceleration, not playback: race_time advanced 2.97x at
requested 3 and 7.89x at requested 8, measured against wall clock. Ceiling
untested above 8.

Consequence: **unusable through the live uinput gamepad path**, because at 8x
the agent would control at 1/8 the tick rate. Exploiting it means pre-scheduled
input revisions (inputs are indexed by tick, not delivered live) and a
rollout-batched trainer - which SAC tolerates, being off-policy.

## Fidelity at 8x - RESOLVED: bit-exact **[V]**

Same seeded script (`0 seed 12345`, throttle held, four steering changes) run at
1x and 8x, each capturing a state clip ~8 s in with the car cornering at
38.4 m/s. Trajectories extracted with the correlation method below and compared
frame by frame:

    shift 0: mean err 0.997 m
    shift 1: mean err 0.665 m
    shift 2: mean err 0.332 m
    shift 3: mean err 0.000000 m   <-- exact, all 97 overlapping frames
    shift 4: mean err 0.332 m
    shift 5: mean err 0.664 m

The error is a clean V, rising by exactly one tick of travel (~0.33 m) per frame
of misalignment - the signature of a pure frame offset, not divergence. At the
correct alignment, position and velocity are **identical to the bit**, and the
error does not grow across the window (0.000000 at frame 0 and frame 90).

**Conclusion: 8x is free. Same seed + same inputs => same simulation.** So the
throughput gain is real and costs nothing in accuracy.

Caveats: tested at 8x only (ceiling above 8 untested), on one map, one script,
one 8-second window. Determinism assumes the `seed` action is pinned - the
script sets it explicitly at tick 0.

## Solved container geometry **[V]**

    header 203 bytes + 100 frames x 2434 bytes + trailer (input actions)

Verified, not fitted: the ground-truth position landed at byte 212045 in one
clip and 202309 in the other, differing by exactly 4 x 2434, and solving
`(offset - 84 - H) mod 2434 == 0` put both hits precisely on frame boundaries
(87 and 83). Trailer was 52 bytes on these clips; it varies between clips, which
is why total blob size is not constant (243642 / 243651 / 243655 all observed).

**Always re-derive H per clip by correlation.** Do not hardcode.

### Working recipe

1. Poll telemetry until the target `race_time`; keep that frame as ground truth.
2. Immediately `capture-state-clip` then `persist-state-clip`; read the blob.
3. Scan the blob for the float triple matching the ground-truth `pos`
   (tolerance ~0.02). Expect exactly one hit.
4. `H = (hit - 84) % 2434`; frame index `= (hit - 84 - H) // 2434`.
5. To compare two runs, sweep the frame shift and take the minimum - the
   capture-latency frame offset is NOT the trajectory alignment (they differed
   by one here: hits implied 4, true alignment was 3).

Note: no per-frame tick counter was found in the first 84 bytes, so alignment
has to be recovered by the shift sweep rather than read directly.

---

# Direct memory access (rebuilding what TICK does)

Goal: read/write the vehicle state ourselves, so save/restore does not depend on
TICK's whole-run pre-scheduled input model.

## Finding the game process

`pgrep -x Trackmania.exe` finds **nothing** - under Proton the process `comm` is
`MainThread`. TICK's `gameProcessId` is **Wine's internal PID** and is useless
against `/proc`. Locate it by cmdline + footprint:

    for d in /proc/[0-9]*: cmdline contains "Trackmania" and VmRSS > 500 MB

## Scanning recipe **[V]**

1. **Scan while parked.** A full rw scan is ~3 GB / ~10 s; a moving car gives an
   inconsistent snapshot across regions.
2. **Use a tolerance, never an exact byte match.** The car creeps even when
   "stopped" (speed ~0.008), so the low mantissa bits change and an exact
   12-byte `struct.pack("<3f", *pos)` search returns **zero hits**. Vectorise
   with numpy: `|arr[i]-x|<0.05` and the same for the next two elements.
3. **Then drive and filter.** A parked car cannot discriminate - stationary
   filtering alone leaves ~450 addresses. Movement kills stale copies fast.
4. Reverse into open ground first, then spin (full lock + throttle): on the
   start line the car just drives off down the track.

Measured 2026-09-06: 249 stationary candidates -> **25 survivors** after ~10 s
of reversing and spinning.

## The 25 survivors (this process instance only - addresses are not stable)

    0x1642046c  (wine-mapping, shared - probably not the physics body)
    0x18b28718  0x18b28808  0x18b28864
    0x305cbf748 0x30b077c70 0x30b077eac 0x30d79f564
    0x30dbe1384 0x30dbe13a4   <- 32 bytes apart
    0x30dd7ae1c 0x30ee2d3c4 0x30ee2d460 0x30ee38434
    0x312022000 0x3181e7b50 0x3181e7eb0 0x3181e8908 0x3181f77f4
    0x40320bc5c

They fall into value *generations*: ~10 hold bit-identical values, 3 hold a
slightly different y/z (a lagged or interpolated copy), etc. The authoritative
physics body is one generation; the rest are render transforms and caches.

**Caveat:** Openplanet (our plugin) and TickNative.dll both live in this same
process and hold their own copies of the position. Region backing does NOT
discriminate them - 19 of the 20 are plain `[anon]` heap.

**Next step: write-testing.** Reading buys little (the plugin already reports
pose, velocity, quaternion, slip, damper, wear). The prize is a *write* the
simulation respects - that is save/restore without TICK's scheduling limit.
Most candidates will ignore writes or only corrupt rendering.

## Driving the car: two independent paths

- **uinput pads** (`control/virtual_pad_server.py`, `act <steer> <gas> <brake>`
  on 8765/8775/8785/8795, deadman timer needs repeats). These require a
  controller to be **assigned to the seat inside the game's own settings** -
  an in-game step that `stack.sh up` does NOT perform (it only prints "confirm
  the game shows controllers"). With no binding, every pad reads as dead and
  `calibrate_seats.py --json` returns all zeros.
- **TICK scripts**, which inject below that layer and therefore drive the car
  with **no controller binding at all**. This is how the car was driven for the
  memory scan.

If the car ignores all four pads, check the binding before suspecting anything
else - it looks exactly like a broken policy or a broken tool.

## RAM vehicle state - WORKING **[V]**

`env/ram_state.py`. Validated live against TMAITelemetry while driving at
27.8 m/s: rpm and damper match EXACTLY, speed within 0.01, velocity within
0.02, position within 0.17 m (occasional ~0.9 m spikes are read/telemetry skew,
32 ms of travel at that speed - not address error).

### Verified offsets (from the vehicle struct base)

    +0x00   f32[3]  position          r=1.00
    +0x0c   f32[3]  velocity          r=1.00
    +0x50   f32     rpm               r=1.00
    +0x68   f32[3]  velocity (copy)   r=1.00
    +0x80   f32[4]  damper            r>=0.9994
    speed           NOT stored anywhere - derive as |velocity|

**Damper wheel order in RAM is not the plugin's order.** RAM +0x80/84/88/8c ==
plugin damper[0]/[1]/[3]/[2]. Assuming a straight copy silently corrupts
per-wheel features.

Struct ADDRESSES change every launch - always `locate()`. `locate()` seeds from
one telemetry frame then confirms across further frames; a single frame yields
~900 false positives, confirmation cuts it to ~12 (all genuine aliases).

**The car must be MOVING when you locate.** With velocity ~0 the pose object is
indistinguishable from the physics body - that mistake cost a whole pass here
(0x305cbf748 was identified as the vehicle struct while parked; it is a
pose/scene transform holding orientation and nothing dynamic, and it reported a
"velocity" of (0.98, 0.01, -0.22) - the `left` unit vector - until the car moved
and eliminated it).

### Orientation

On the pose object, not the vehicle struct: `left` at +356, `dir` at +368.
Two objects, two reads.

### Still missing

- `slip[4]`, `ground_dist`: vary, but correlate nowhere within +/-16 KB of the
  vehicle struct. Separate object, not yet located.
- `gear`, `in_steer`, `in_gas`, `adherence`, `wear[4]`, `icing[4]`: constant in
  every run so far, so correlation cannot place them. Per FLEET_V2 these are
  genuinely 0/inactive on a plain road map. Identifying them needs a drive
  script that actually varies throttle and steering.
- `lidar` (16 dims), `fx_ahead` (8 dims): computed by OUR plugin from surveyed
  map geometry. Not vehicle state - these can NEVER come from RAM.

So RAM covers pose, velocity, rpm, suspension and derived speed. That is a real
subset of the 106-dim obs, not a full replacement for the plugin yet - but it
needs no Openplanet, no Developer mode and no paid account, which is the
FLEET_V2 constraint that forced the degraded 33-float adapter.

### Method traps (each of these produced a WRONG answer here)

1. **Exact matching fails.** The plugin frame is tens of ms stale against an
   instantaneous memory read; at 30 m/s even the correct address misses. Use
   Pearson correlation across samples - it survives lag and unit differences.
2. **Scan and verify must sample telemetry and memory TOGETHER.** Comparing a
   pre-scan position against post-scan memory fails silently: at 38 m/s the car
   travels ~19 m during a 10 s scan.
3. **Freeze the sim to scan.** `requestedGameSpeed` accepts fractional values
   (0.05 works), so the car keeps a genuine non-zero velocity while a full
   ~3 GB scan stays coherent.
4. **A clip offset does not predict a RAM offset.** The clip is a serialized
   snapshot; TICK gathers fields from wherever they live.
5. **Vary the inputs.** A constant-throttle script leaves in_gas/in_steer
   constant, and a constant field cannot be located by correlation at all.

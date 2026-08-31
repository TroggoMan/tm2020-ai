Reads the car and the track, and hands it to a program outside the game over a
local socket. It doesn't draw anything and it doesn't drive anything — it's a
data feed, and the program on the other end decides what to do with it.

I wrote it for a reinforcement learning project, but there's nothing
RL-specific in here. Anything that wants to read Trackmania from outside can
use it: a custom overlay on a second monitor, a logger, a driving analyser, a
stream widget.

## What comes out

One JSON object per line on TCP 8766, at up to your frame rate.

**The car** — position, velocity, direction, up and left vectors, speed, rpm,
gear, distance, race time, checkpoint count, respawns, adherence, side speed,
air brake, and the inputs the game actually received.

**Each wheel** — slip coefficient, ground contact material, icing, dirt, tyre
wear, brake coefficient. Plus wetness and water immersion for the car.

**Effects** — turbo level and remaining time, reactor level, type and timer,
cruise speed, the slow-motion coefficient, and vehicle type.

**Every seat in splitscreen.** A `players` array with one entry per local
player, each with its own position, race time, checkpoint count, inputs and UI
state. As far as I know nothing else exposes this.

**Every vehicle in the scene**, optionally, which is how you capture a ghost
racing alongside you.

## Commands, on the same socket

- `landmarks` — world positions of every checkpoint, the finish and the spawn
- `dumpmap [occupancy]` — block model names, and optionally the grid cells each
  block occupies
- `restart`, `goto <uid>`, `playmap <url>`, `menu` — map control
- `rate <hz>` — change the send rate while it's running
- `perms` — whether this account may start a local map
- `ping`

`landmarks` and `dumpmap` are the reason this exists in the shape it does. With
those you can work out where a track goes before driving it, which is a very
different problem from reading a speedometer.

## How it compares to what's already here

**TrackmaniaRL Connect** covers the same ground for the single-player car and
sends packed binary, which is lighter on the wire than my JSON. If that's all
you need, it's the leaner choice and it's been around longer.

Three things here that aren't there:

- per-seat splitscreen data, rather than terminal 0 only
- static map geometry (`landmarks`, `dumpmap`) as well as live car state
- one dependency (VehicleState) instead of a chain through PlayerState,
  MLHook and MLFeedRaceData

And one thing that's worse: JSON lines are bigger than packed floats. At 100Hz
with all the optional blocks on it's a few hundred KB/s over loopback. That has
never been the bottleneck for me — the frame rate is — but it's a real
difference and worth knowing before you pick.

**MLFeed: Race Data** is a different job. It's about race and player state
across everyone in a server; this is about one machine's cars in detail.

## It does not press anything

Worth saying plainly, because "telemetry for an AI" invites the question.

This plugin reads `InputSteer`, `InputGasPedal` and `InputIsBraking` the same
way any dashboard does, and that's the extent of its involvement with input.
There is no key synthesis, no input injection, nothing that steers a car.

The map commands (`PlayMap`, `RequestRestartMap`, `RequestGotoMap`) are the
documented script API, and the same calls the map-rotation and ManiaExchange
plugins already make.

In my own setup the driving is done by a virtual gamepad created at the
operating system level, entirely outside the game and entirely outside this
plugin. The game sees an ordinary controller. Nothing here is involved.

## Settings

- **Listen port** — 8766 by default. One port per running game if you have
  several.
- **Max send rate (Hz)** — 0 for every frame. The plugin emits from a render
  callback, so this is a ceiling, not a promise: it can't go faster than the
  game draws.
- **Include per-wheel detail** — on by default.
- **Include every vehicle in the scene** — for ghost capture.
- **Emit one record per local player** — splitscreen; harmless and empty in
  single player.

## Notes

Only one client can connect at a time. If you need several readers, put a
fan-out proxy in front of it rather than reconnecting.

The map uid is included in every record, including the heartbeat sent when no
car is in view. That matters in splitscreen, where the camera is on one seat
at a time and most records are heartbeats.

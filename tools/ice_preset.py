#!/usr/bin/env python3
"""Set up an ICE explore run's tuning, for whatever map is loaded right now.

    tools/ice_preset.py                 # ask the game which map, then write it
    tools/ice_preset.py --uid <uid>     # a map you already know
    tools/ice_preset.py --dry-run       # print the changes, write nothing
    tools/ice_preset.py --stage race    # the race-stage variant

Writes `configs/<uid>.explore.json`, which is exactly the file the web panel
edits - so this IS "set it up in the GUI", it just does not need the map to
have been through the panel first. Anything already in that file is kept unless
this preset has an opinion about it, and every change is printed with the
reason, so nothing arrives silently.

WHY ICE NEEDS ITS OWN PRESET
----------------------------
Ice is ~0.20 adherence against asphalt's 1.0 (env/surfaces.py). Three of the
defaults are priors that are simply false there:

1. **"Throttle is good."** `reward.w_gas` pays for holding the pedal down. On
   ice that pays for wheelspin. `w_accel` - which pays for the m/s actually
   gained - is the honest half of that pair and stays on, so the preset drops
   `w_gas` to 0 and keeps `w_accel`. (The README makes this exact distinction
   for a car pinned against a wall; ice is the same problem with no wall.)
2. **"Sliding is a technique to add later."** On tarmac it is - `speedslide`
   ships at w=0 and needs 400km/h. On ice, controlling the slide *is* how you
   make progress at all, so `iceslide` is the difference between a dense signal
   and a car that spins in place while progress stays flat. It goes on here,
   during explore, rather than being held back for the racer.
3. **"The game's own steering ramp is enough."** `action.steer_rate` defaults
   to 0 for good reasons on tarmac. But the measured failure on 2026-08-31 was
   the policy sawing the wheel 8.3 times a second - sign flips on 41.3% of
   consecutive steps - and on ice a sawed wheel does not merely fail to turn,
   it spins the car. Notches + a slew limit cut that to 0.8% and 0.2/s.

What this preset deliberately does NOT set: `surfaces` penalties and
`terminate_on_surface`. Which materials are off-track is a property of the
individual map, the env writes `maps/<uid>.materials.json` as it drives, and
the panel's "on this map" tier fills in from that. Guessing here is how you get
a -500 term that teaches the policy the reward function is broken.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (section, key, value, force, why).
#
# `force` is the important column. A preset that overwrites everything would
# stomp values somebody tuned against this particular track - and silently, on
# a file the panel owns. So only the keys that ARE the ice preset are forced;
# the rest are defaults, filled in when a fresh map has nothing there and left
# alone otherwise. Running this twice on a tuned map is then a no-op.
EXPLORE = [
    ("enabled", "iceslide", True, True,
     "on ice the drift IS the driving, not a flourish to add later"),
    ("enabled", "speedslide", False, True,
     "SDHelper's bands need 400km/h on road and grey out on ice - inert here"),

    ("iceslide", "w", 0.12, True,
     "per-step pay at the balanced slip angle. Sized against progress: at "
     "~100km/h progress earns ~0.69/step, and a held green drift tops out at "
     "w*streak_cap = 0.48, so it shapes without out-earning getting there"),
    ("iceslide", "w_any", 0.03, True,
     "a floor for yellow/orange, so there is a gradient to climb from outside "
     "the band - a step function gives the policy no direction"),
    ("iceslide", "w_blue", 0.0, True,
     "NO penalty for under/overangle yet. Charging for a mistake before the "
     "car can drive at all is the '-500 for grass' trap; turn this on once it "
     "is finishing"),

    ("reward", "w_gas", 0.0, True,
     "the 'just hold throttle' prior is false at 0.2 adherence - it pays for "
     "wheelspin. w_accel keeps paying for speed actually gained"),
    ("reward", "w_accel", 0.05, False, "the honest half of that pair"),

    ("action", "steer_levels", 10, True,
     "ten notches per side, i.e. how a keyboard drives - and keyboard players "
     "are competitive, so it is known to be enough control"),
    ("action", "steer_rate", 8.0, True,
     "full lock in 125ms. Explore only: the racer sets 0.0 because some TM "
     "technique needs lock-to-lock in one frame"),

    ("line", "w_soft", 0.0, True,
     "explore's provisional line cuts through scenery - charging per metre off "
     "it punishes driving on the only surface there is"),
    ("line", "max_offset", 250.0, False,
     "explore opens this anyway; stated here so the panel shows it"),

    ("stuck", "speed", 0.7, False, "a car sliding gently on ice is not stuck"),
    ("stuck", "seconds", 8.0, False,
     "8s not 5s: an ice recovery is slow and legitimate"),

    ("episode", "max_episode_s", 120.0, False,
     "a car that has never reached a checkpoint gets 2 minutes"),
    ("episode", "grant_per_cp", 30.0, False,
     "+30s per checkpoint ever reached, so the cap tracks the frontier"),
    ("episode", "episode_ceiling", 210.0, False,
     "hard ceiling on that growth"),
]

# The racer inherits the explorer's weights, so the steering regime has to
# change deliberately rather than by leaving the explore file in place.
RACE_DELTA = [
    ("action", "steer_rate", 8.0, True,
     "KEPT, not dropped to 0 - the tarmac racer drops the slew limit because "
     "some TM technique needs lock-to-lock in one frame, but ice is a BALANCE "
     "problem: Snacky's ICE BASICS (which env/iceslide.py is built on) is "
     "about holding a sustained counter-steer, not flicking. And the racer "
     "inherits a policy trained WITH the ramp, so removing it means oversteer "
     "while the buffer is empty - which on ice is a spin, not a lost tenth"),
    ("line", "w_soft", 0.002, True,
     "a real line exists now, so leaning on it is fair"),
    ("line", "max_offset", 60.0, True,
     "and leaving it should end the episode"),
]


def live_uid(port: int = 8767, timeout: float = 8.0) -> str | None:
    """Ask the game which map is loaded.

    Via the `landmarks` reply rather than the telemetry record's `map` field:
    in splitscreen the top-level record is frequently a heartbeat with no map,
    and a null uid there is what once had the env caching geometry under
    'unknown'. The landmarks reply answers authoritatively.
    """
    from env.tm_env import TelemetryLink
    link = TelemetryLink(addr=("127.0.0.1", port))
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            reply = link.command("landmarks", wait=3.0)
            # `map`, not `uid` - that is the field name tm_env reads too.
            if reply and reply.get("ok") and reply.get("map"):
                return reply["map"]
            time.sleep(1.0)
    except Exception as ex:                                    # noqa: BLE001
        print(f"  could not reach the game on :{port} ({ex})", flush=True)
    finally:
        try:
            link.close()
        except Exception:                                      # noqa: BLE001
            pass
    return None


def apply(data: dict, changes: list) -> tuple[list, list]:
    """Merge changes in. Returns (applied, kept) for printing.

    `kept` is the unforced keys the file already had an opinion about - worth
    showing, because "the preset did not change your stuck threshold" is
    information, and a silent skip looks identical to a silent overwrite.
    """
    applied, kept = [], []
    for section, key, value, force, why in changes:
        target = data.setdefault(section, {}) if section else data
        old = target.get(key, _UNSET)
        if not force and old is not _UNSET:
            if old != value:
                kept.append((section, key, old))
            continue
        if old != value:
            applied.append((section, key,
                            "<unset>" if old is _UNSET else old, value, why))
        target[key] = value
    return applied, kept


_UNSET = object()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uid", help="map uid; default: ask the running game")
    ap.add_argument("--stage", choices=("explore", "race"), default="explore")
    ap.add_argument("--port", type=int, default=8767,
                    help="broker port for the instance to ask (default 8767)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    uid = args.uid
    if not uid:
        print(f"asking the game on :{args.port} which map is loaded…",
              flush=True)
        uid = live_uid(args.port)
        if not uid:
            print("\nNo map uid. Load the track in the game first (and make "
                  "sure the plugin is connected), or pass --uid.\n"
                  "The uid also appears in the panel once a map is loaded.",
                  flush=True)
            return 1
        print(f"  map: {uid}", flush=True)

    path = os.path.join(ROOT, "configs", f"{uid}.{args.stage}.json")
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"editing existing {os.path.relpath(path, ROOT)}", flush=True)
    except (OSError, json.JSONDecodeError):
        data = {}
        print(f"creating {os.path.relpath(path, ROOT)}", flush=True)

    changes = EXPLORE if args.stage == "explore" else EXPLORE + RACE_DELTA
    applied, kept = apply(data, changes)

    if not applied:
        print("  already set up - nothing to change", flush=True)
    for section, key, old, new, why in applied:
        print(f"  {section}.{key}: {old} -> {new}\n      {why}", flush=True)
    if kept:
        print("\n  left alone (already tuned for this map; the preset only "
              "suggests these):", flush=True)
        for section, key, old in kept:
            print(f"    {section}.{key} = {old}", flush=True)

    if args.dry_run:
        print("\n--dry-run: nothing written", flush=True)
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nwrote {os.path.relpath(path, ROOT)} - the panel owns it from "
          f"here, and the env re-reads it within a second while training runs.",
          flush=True)

    if args.stage == "explore":
        print("\nBefore the first run, cache the map's geometry (lidar needs "
              "it; without it every beam reads 1.0 and the car is blind):\n"
              "  .venv/bin/python tools/dump_map.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

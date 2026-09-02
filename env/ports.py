"""Which ports belong to which game instance.

Stdlib only, and deliberately its own module rather than living in tm_env:
the fleet supervisor and the web panel both need this mapping and neither can
import tm_env, which pulls in numpy and gymnasium. The pad server in
particular has to run on the SYSTEM python (it needs evdev and uinput
permission), so anything it shares with the trainer has to be importable
without the venv.

    instance 0    pad 8765   plugin 8766   broker 8767
    instance 1    pad 8775   plugin 8776   broker 8777
    instance 2    pad 8785   plugin 8786   broker 8787

The stride is ten, not one. The three base ports are adjacent, so stepping
each of them by the instance index walks them into each other immediately -
instance 1's pad would be 8766, which is instance 0's plugin, and the symptom
would be a pad server and a game silently fighting over one socket. Ten leaves
room in between, and instance 0 keeps exactly the ports every existing script,
config and habit already uses.

The plugin port is the one this cannot set: it lives in each game's own
Openplanet config inside that game's Wine prefix, and has to be set there by
hand to match.
"""
from __future__ import annotations

import json
import os

# Written by tools/calibrate_seats.py; pad index -> splitscreen seat.
_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "seat_calibration.json")

# Which machine an instance's pad and broker live on.
#
# Defaults to this machine, so nothing changes for a single-box setup. Set
# TMAI_HOSTS to spread the games across several machines and feed ONE learner:
#
#     TMAI_HOSTS="127.0.0.1,192.168.0.203"
#
# The list is indexed by instance, and the last entry repeats - so two hosts
# and four instances puts 0 on the first and 1,2,3 on the second unless you
# spell all four out. The learner is not the thing worth distributing: it is a
# 256x256 MLP that costs ~3ms a step. The GAMES are, because they run in real
# time and no amount of compute makes a lap happen faster.
HOST = os.environ.get("TMAI_HOST", "127.0.0.1")
HOSTS = [h.strip() for h in os.environ.get("TMAI_HOSTS", "").split(",")
         if h.strip()] or [HOST]


def host_for(instance: int) -> str:
    """Which machine instance `i` runs on."""
    i = int(instance)
    return HOSTS[i] if i < len(HOSTS) else HOSTS[-1]
PAD_BASE = 8765
PLUGIN_BASE = 8766
BROKER_BASE = 8767
INSTANCE_STRIDE = 10


def instance_ports(i: int) -> dict[str, int]:
    off = INSTANCE_STRIDE * int(i)
    return {"pad": PAD_BASE + off,
            "plugin": PLUGIN_BASE + off,
            "broker": BROKER_BASE + off}


def pad_addr(i: int = 0) -> tuple[str, int]:
    return (host_for(i), instance_ports(i)["pad"])


def broker_addr(i: int = 0) -> tuple[str, int]:
    return (host_for(i), instance_ports(i)["broker"])


# --- splitscreen -----------------------------------------------------------
#
# A different shape entirely. Above, an instance is a whole game with its own
# plugin and broker. In splitscreen there is ONE game - one plugin, one broker
# - and up to four seats inside it, each needing its own uinput pad so the game
# can tell them apart.
#
# So the pads still step by ten (four distinct devices), but every seat shares
# instance 0's broker and reads its own entry out of the telemetry's `players`
# array. That is why the env takes `slot` separately from `instance`: they
# answer different questions.
#
#     seat 0    pad 8765  ->  broker 8767, players[0]
#     seat 1    pad 8775  ->  broker 8767, players[1]
#     seat 2    pad 8785  ->  broker 8767, players[2]
#     seat 3    pad 8795  ->  broker 8767, players[3]
#
# Local multiplayer is also the only way a Starter Access account can start a
# map off local disk, so this is not merely a throughput trick.

MAX_SEATS = 4

# Seats of the SECOND and later games get their own range, well clear of the
# per-instance block above.
#
# The obvious formula - PAD_BASE + 10*(game*MAX_SEATS + seat) - collides:
# game 1 seat 0 lands on 8805, which is instance 4's pad, and game 1 seat 1
# lands on 8815, which is instance 5's broker. With six accounts in play that
# is not hypothetical. Game 0 keeps the ports everything already uses, so a
# running single-game setup is never disturbed by adding a second.
SEAT_PAD_BASE = 8900


def _calibrated_pad_index(seat: int) -> int:
    """Which PAD INDEX actually drives this seat, per the last calibration.

    The formula below assumes pad N drives seat N. The game does not promise
    that: it binds seats to controllers in whatever order it enumerated them,
    and that order changes when the pad servers or the game restart (there is
    a standing warning about this in the project notes).

    This was not hypothetical. A measured run had pad1 driving car3 and pad3
    driving car1 - so env slot 1 was steering a car it could not see while
    observing a car it could not steer, and likewise for slot 3. Those two
    seats reached the first checkpoint in 0.7% and 0.3% of episodes against
    88% and 79% for the two correctly-wired ones, and because they failed fast
    they produced ~76% of all episodes: three quarters of the replay buffer
    was transitions whose action did not cause the observed outcome.

    `tools/calibrate_seats.py` has always measured this correctly and written
    it to logs/seat_calibration.json - but only the web panel ever read it, to
    draw a status dot. Nothing applied it. Now the trainer does.

    Falls back to the identity (pad N -> seat N) when there is no calibration,
    which is the old behaviour.
    """
    try:
        with open(_CALIBRATION_PATH) as fh:
            mapping = json.load(fh).get("mapping", {})
    except (OSError, ValueError):
        return int(seat)
    # The file is pad -> seat; we need seat -> pad.
    for pad_index, bound_seat in mapping.items():
        if bound_seat is not None and int(bound_seat) == int(seat):
            return int(pad_index)
    return int(seat)


def seat_ports(seat: int, game: int = 0, raw: bool = False) -> dict[str, int]:
    """Ports for one splitscreen seat of one game.

    The seat has its own pad - the game tells seats apart by controller - but
    shares its game's single plugin and broker, reading its own entry out of
    the telemetry's `players` array.

    The pad is chosen by the last calibration, not by assuming pad N drives
    seat N - see _calibrated_pad_index.
    """
    game_ports = instance_ports(game)
    # raw=True bypasses the calibration and returns the plain formula.
    #
    # tools/calibrate_seats.py MUST use it. It drives "pad N" and watches which
    # car moves; if it asks this function for the pad while this function is
    # already applying the last calibration, it measures through the
    # correction, finds every seat driving its own car, and writes the
    # IDENTITY - silently undoing the fix. That happened once: a crossed
    # mapping was corrected, the tool was re-run to confirm it, and the confirm
    # step reset it to crossed.
    idx = int(seat) if raw else _calibrated_pad_index(seat)
    if game == 0:
        pad = PAD_BASE + INSTANCE_STRIDE * idx
    else:
        pad = SEAT_PAD_BASE + INSTANCE_STRIDE * (
            (int(game) - 1) * MAX_SEATS + idx)
    return {"pad": pad,
            "plugin": game_ports["plugin"],
            "broker": game_ports["broker"]}


def seat_pad_addr(seat: int, raw: bool = False) -> tuple[str, int]:
    return (HOST, seat_ports(seat, raw=raw)["pad"])

"""Speed-slide quality, using SDHelper's own numbers.

A speed slide ("SD") is the fast line above ~400km/h on road: the car is held
at a *specific* sideways speed, and both too little and too much is slower.
Getting that band right by guesswork is hopeless, so this table is lifted
straight from the SDHelper plugin (Tobirousch/sdhelper, installed in this
prefix), which is what human SD players actually steer by.

The important thing about SDHelper is what it does *not* do. It never reads
skidmarks - it can't, nothing exposes them. It reads exactly two numbers,

    VehicleState::GetSideSpeed(vis)   sideways speed
    vis.FrontSpeed                    forward speed, sign included

plus the material under the front-left wheel, and then it *swaps the skidmark
texture on disk* so the marks the game draws come out green / yellow / orange /
blue. The colour is an output, not an input. So we do not need to see
skidmarks to learn what the helper teaches - we need the same two numbers,
and the plugin has streamed both all along.

Which is also why this is a reward and not an observation: `side` and `speed`
are already in the observation vector, so the policy can see everything the
band is computed from. It just had no reason to care until now.

Bands are in km/h of |side speed|. Each row is

    (orange_lo, yellow_lo, green_lo, green_hi, yellow_hi, orange_hi)

read as: green between green_lo and green_hi, yellow out to the yellow edges,
orange out to the orange edges, nothing outside that. Grass and dirt slide at
much lower side speeds than road, and reversing shifts every band up.
"""
from __future__ import annotations

# Forward. Below `limit` km/h a slide is not an SD, it is just scrubbing speed
# off, and SDHelper greys itself out - so do we.
FORWARD = {
    "road":  {"limit": 400.0, "band": (7.0, 13.0, 19.0, 22.0, 28.0, 34.0)},
    "grass": {"limit": 200.0, "band": (1.0, 1.0, 7.0, 10.0, 13.0, 22.0)},
    "dirt":  {"limit": 200.0, "band": (1.0, 3.0, 9.0, 12.0, 18.0, 24.0)},
    # PLASTIC - not from SDHelper (no plastic case) but anchored on the wiki:
    # SD works from 200 km/h on dirt/grass/plastic, the angle is MUCH LOWER
    # than road, and plastic's grip is about grass's. So it sits near the grass
    # row above, not the road one. See _PLASTIC.
    "plastic": {"limit": 200.0, "band": (2.0, 5.0, 8.0, 12.0, 16.0, 21.0)},
}

# Reversing. SDHelper applies no speed floor at all going backwards.
BACKWARD = {
    "road":  {"limit": 0.0, "band": (17.0, 23.0, 29.0, 32.0, 38.0, 44.0)},
    "grass": {"limit": 0.0, "band": (1.0, 6.0, 12.0, 15.0, 21.0, 27.0)},
    "dirt":  {"limit": 0.0, "band": (2.0, 8.0, 14.0, 17.0, 23.0, 29.0)},
    "plastic": {"limit": 0.0, "band": (2.0, 7.0, 13.0, 16.0, 21.0, 27.0)},
}

# SDHelper switches on the raw material name under the front-left wheel, and
# only distinguishes three cases. "Green" is the name TM2020's grass blocks
# actually report - not "Grass", which is why matching on the obvious string
# silently never fires.
_GRASS = ("Green", "Grass", "WetGrass", "Wheat")
_DIRT = ("Dirt", "DirtRoad", "WetDirtRoad", "Sand", "Gravel")

# PLASTIC - added here, and its band is the one thing in this file that is NOT
# lifted from SDHelper.
#
# WHY SDHELPER HAS NOTHING TO LIFT: the helper's entire output is swapping the
# skidmark texture on disk so the marks come out green/yellow/orange/blue.
# Plastic draws NO SKIDMARKS AT ALL, so there was never anything for it to
# paint and it has no plastic case - Plastic fell through its `else` into the
# ROAD row, which demands 400 km/h. On a 30%-grip surface that floor is never
# cleared, so the term was silently dead on every plastic track. Same shape of
# gap as ice, which is why env/iceslide.py had to be written from scratch.
#
# WHERE THESE NUMBERS COME FROM (community consensus, not measurement):
#   * slides start once you clear 200-220 km/h, best in 4th gear and up
#     -> limit 200, matching grass/dirt rather than road's 400;
#   * "slide the absolute minimum possible", and sliding wide "dramatically
#     bogs down your speed" -> a NARROW, LOW green with the upper shoulders
#     pulled in tighter than dirt's (16/21 against dirt's 18/24), because
#     over-sliding is the expensive error here;
#   * anchor for the green itself: road's green is 19-22 km/h of side speed at
#     >=400 km/h, i.e. atan(20/400) ~= 2.9 deg of slip. Holding that same
#     shallow angle at 200 km/h works out at ~10 km/h of side speed, hence
#     green 8-12.
#
# CONFIRMED 2026-09-07 against community sources, and it validates the shape:
# speedsliding on DIRT, GRASS AND PLASTIC works at 200 km/h or above, and the
# angle has to be MUCH LOWER than on road for maximal speed gain - gentler
# steering, or more infrequent and lighter tapping. So the 200 floor and the
# shallow green below are right, and `speed_floor_kmh` should NOT be lowered to
# "make the term fire" on a slow car. Tried that (120) and reverted: below 200
# there is no speed-gain mechanic to reward, so paying for a slide there trains
# a habit that has to be untrained later. The route to making this term fire is
# reward.par_speed making slowness expensive until the car reaches 200+.
#
# A SPEEDSLIDE IS NOT A DRIFT, and this module only models the former:
#   speedslide (SD / speeddrift) - a technique to GAIN speed. Hold a specific
#     shallow angle at high speed and the car accelerates beyond normal; the
#     community keys it visually off ~50% skidmark overlap. Road wants 400+
#     (autoslide past ~598); dirt/grass/plastic from 200 at a shallower angle.
#   drift (brake drift, "s4d" = press (s)brake (4)for (d)rift) - a technique to
#     CORNER. Lift, steer into the corner, tap/hold brake so the rear steps out,
#     rotate, then accelerate out; tap the brake again to tighten. Needs ~180+.
#     BRAKING IS PART OF IT, not a failure to be penalised.
# Nothing in this repo rewards a drift. env/iceslide.py is the closest shape
# (banded on slip ANGLE rather than side speed) but it is gated to the `ice`
# grip group in tm_env, and its 33-46 deg green is an ice number, not a plastic
# one. A plastic drift term would be a new thing, keyed on the brake-then-
# rotate sequence, and env/hints.py is the mechanism for teaching that sequence.
#
# THE BAND WAS WIDENED TO 45-75 AND PUT BACK. Recorded so it is not done a
# third time. A 14.0s reference ghost on the plastic training map holds a
# median 57 km/h of side speed (13-15 deg) while cornering, which made the
# 8-12 green look 5x too tight - it scores that driver as "not sliding
# usefully" 77% of the time. But that is CORNERING, not SD:
#   * the wiki puts plastic SD at 200+ km/h with an angle MUCH LOWER than
#     road's, and plastic's grip at about grass's - and grass's green here is
#     7-10 km/h, right next to plastic's 8-12. The band is consistent with
#     every source once it is read as an SD band.
#   * the ghost at 200-280 km/h is making corners, which on this track is what
#     the four seconds are actually made of. That is a different technique with
#     no band in this repo (see the speedslide-is-not-a-drift note above), and
#     writing its numbers into the SD row just conflates the two again.
# So: do not calibrate this band off cornering telemetry. If the SD band is
# ever to be measured, measure it where the car is ACCELERATING through a
# shallow slide above 200 km/h, not where it is rotating through a corner.
#
# Also corrected: the Speed Drift Trainer plugin (SilasDo) ships calibration
# for road/grass/dirt/plastic/metal, and its plastic data spans 308-997 km/h.
# That is where its author collected data on fullspeed maps, NOT where the
# technique begins - the wiki is explicit that dirt SD starts around 220 and
# runs to ~900. Do not read that lower bound as a floor.
#
# UNITS TRAP, do not "fix" this by typing 35 into the band: the widely quoted
# "~35% angle" for plastic is a STEERING INPUT percentage, not a slip angle,
# and this table is in km/h of side speed. The three are different quantities.
# Treat the band as a seed estimate exactly as env/iceslide.py says of its own:
# set w > 0, watch the WHY log's grade/score against lap time, move the edges.
#
# Because plastic is a genuine speedslide (shallow angle, carry speed) rather
# than ice's balance-a-big-angle problem, it belongs in this module with the
# streak / accel / stall machinery, not in iceslide.
_PLASTIC = ("Plastic",)

MS_TO_KMH = 3.6

# What the helper paints, and what each colour is worth. Green is the target;
# blue is SDHelper's "you are not sliding usefully" default.
GRADES = ("none", "blue", "orange", "yellow", "green")
_SCORE_AT = (0.0, 0.35, 1.0, 1.0, 0.35, 0.0)


def sd_surface(material_name: str) -> str:
    """Which surface case a material falls into.

    SDHelper's own three, plus plastic - which the helper cannot distinguish
    (no skidmarks to paint) and so silently dropped into `road` and its
    unreachable 400 km/h floor. See _PLASTIC.
    """
    if material_name in _GRASS:
        return "grass"
    if material_name in _DIRT:
        return "dirt"
    if material_name in _PLASTIC:
        return "plastic"
    return "road"


def evaluate(side_speed_ms: float, front_speed_ms: float,
             material_name: str = "Asphalt",
             floor_kmh: float = 0.0) -> tuple[str, float, dict]:
    """Grade the current slide.

    Returns (grade, score, detail). `score` is 0..1, peaking across the whole
    green band and falling off through yellow and orange to nothing - a ramp
    rather than four steps, because a step function gives the policy no
    direction to move in. `grade` is the discrete colour SDHelper would be
    showing, for the logs and the panel.

    Speeds go in as m/s, which is what the plugin streams; the bands are in
    km/h, which is what SD players talk in.

    `floor_kmh` replaces SDHelper's per-surface speed floor outright, in both
    directions. It has to be able to *lower* the floor as well as raise it -
    that is the whole point of having it, since checking a slide reward
    against a slower car is otherwise impossible.
    """
    front = float(front_speed_ms) * MS_TO_KMH
    side = abs(float(side_speed_ms)) * MS_TO_KMH
    surf = sd_surface(material_name)
    table = BACKWARD if front < 0 else FORWARD
    row = table[surf]
    limit = float(floor_kmh) if floor_kmh else row["limit"]
    detail = {"surface": surf, "side_kmh": side, "front_kmh": front,
              "reversing": front < 0, "band": row["band"], "limit": limit}

    if abs(front) < limit:
        detail["reason"] = "below the SD speed floor"
        return "none", 0.0, detail

    lo, hi = row["band"][0], row["band"][-1]
    if side <= lo or side >= hi:
        return "blue", 0.0, detail

    b = row["band"]
    # Piecewise-linear through the band edges: 0 at the orange edges, 0.35 at
    # the yellow edges, 1.0 across green.
    score = 0.0
    for i in range(len(b) - 1):
        if b[i] <= side <= b[i + 1]:
            span = b[i + 1] - b[i]
            t = 0.0 if span <= 0 else (side - b[i]) / span
            score = _SCORE_AT[i] + t * (_SCORE_AT[i + 1] - _SCORE_AT[i])
            break

    if b[2] < side < b[3]:
        grade = "green"
    elif b[1] < side < b[4]:
        grade = "yellow"
    else:
        grade = "orange"
    return grade, float(score), detail

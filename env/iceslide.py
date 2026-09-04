"""Ice-drift quality - the ice counterpart to env/speedslide.py.

WHAT THE RESEARCH TURNED UP
---------------------------
1. SDHelper (Tobirousch/sdhelper, in this prefix) - the reference for slide
   bands - has NO ice model. Its source switches on three cases only
   (`GroundContactMaterial` == "Green" / "Dirt" / else -> road) and bands
   `|side speed|` km/h against a forward *speed floor* of 400 (road) / 200
   (grass, dirt). Ice / RoadIce / Snow fall into "road", and you never reach
   400 km/h on ~0.1-adherence ice, so SDHelper just greys out the whole time.
   Nothing to lift, and it uses neither gear nor angle.

2. Snacky_TM's "ICE BASICS 1/2 & 2/2" blueprints describe ice sliding as a
   CONTROL problem around a target angle, not a number:

     * entry rotation into the corner builds rotational momentum;
     * counter-steer + accelerate balances it into a "sideways slide while
       maintaining grip";
     * the slide has three states -
         UNDERANGLE  - widens your radius   (the safe error: you run wide)
         BALANCED    - the safezone         (hold here)
         OVERANGLE   - tightens your radius  (the dangerous error)
     * left alone the angle DECAYS toward underangle ("TIME -> UNDERANGLE");
       a brake tap PUSHES it toward overangle ("BRAKE -> OVERANGLE"), and too
       far past overangle is "!! SLIDEOUT !!".

So the model here is a band with the BALANCED angle as green, underangle and
overangle as the yellow/orange shoulders, and the overangle shoulder tighter
than the underangle one because sliding out is worse than running wide. The
policy is not told to tap the brake - being in-band is rewarded and the brake
taps that hold it there fall out of that.

The quantity is the SLIP ANGLE - atan2(side_speed, front_speed) in degrees -
not raw side speed, because a balanced drift is about the same angle across the
useful speed range (~90-260 km/h) while its side speed is not. Once
front_speed goes negative the car is tail-first (slip > 90 deg); that gets its
own higher band with no speed floor, as SDHelper does for reversing.

Both inputs (side speed, forward speed) are already in the observation, so -
like speedslide - this is a REWARD, not a new thing to see. The degree numbers
are seed estimates: set `w` > 0, watch the WHY log's grade/score against lap
time, move the edges. `w` ships at 0.
"""
from __future__ import annotations

import math

MS_TO_KMH = 3.6

# Slip-angle bands, in DEGREES. Row layout matches speedslide.py:
#   (orange_lo, yellow_lo, green_lo, green_hi, yellow_hi, orange_hi)
# green == Snacky's BALANCED safezone; below green_lo is UNDERANGLE, above
# green_hi is OVERANGLE. The overangle shoulder (green_hi..orange_hi) is
# deliberately narrower than the underangle one (orange_lo..green_lo): running
# wide costs time, sliding out ends the run.
FORWARD = {"limit": 70.0, "band": (12.0, 24.0, 33.0, 46.0, 53.0, 61.0)}
# Tail-first. No speed floor - a backward drift at any speed is a real state.
BACKWARD = {"limit": 0.0, "band": (95.0, 108.0, 118.0, 133.0, 143.0, 155.0)}

GRADES = ("none", "blue", "orange", "yellow", "green")
_SCORE_AT = (0.0, 0.35, 1.0, 1.0, 0.35, 0.0)


def slip_angle_deg(side_speed_ms: float, front_speed_ms: float) -> float:
    """Angle between heading and travel, 0..180 deg. 0 = straight ahead,
    90 = fully sideways, >90 = travelling tail-first."""
    return math.degrees(math.atan2(abs(float(side_speed_ms)),
                                   float(front_speed_ms)))


def evaluate(side_speed_ms: float, front_speed_ms: float,
             gear: int | None = None, floor_kmh: float = 0.0,
             gear_band: tuple[int, int] | None = None,
             gear_penalty: float = 0.5) -> tuple[str, float, dict]:
    """Grade the current ice drift.

    Returns (grade, score, detail). `score` is 0..1, flat across green
    (BALANCED) and ramping down through the underangle / overangle shoulders
    to 0 - a slope, not steps, so the policy has a direction to move in.
    Speeds go in as m/s (what the plugin streams); bands are km/h / degrees.

    `floor_kmh` replaces the forward speed floor outright when non-zero.
    `gear_band` (lo, hi) inclusive: when set and `gear` is outside it the score
    is multiplied by `gear_penalty`. Off by default - SDHelper uses no gear and
    there is no defensible gear->angle table; this is only a soft nudge.
    """
    front_kmh = float(front_speed_ms) * MS_TO_KMH
    beta = slip_angle_deg(side_speed_ms, front_speed_ms)
    row = BACKWARD if front_kmh < 0 else FORWARD
    limit = float(floor_kmh) if floor_kmh else row["limit"]
    b = row["band"]
    detail = {"slip_deg": beta, "front_kmh": front_kmh,
              "reversing": front_kmh < 0, "band": b, "limit": limit,
              "gear": gear}

    if abs(front_kmh) < limit:
        detail["reason"] = "below the drift speed floor"
        return "none", 0.0, detail

    if beta <= b[0] or beta >= b[-1]:
        detail["reason"] = ("overangle - slideout" if beta >= b[-1]
                            else "underangle - barely sliding")
        return "blue", 0.0, detail

    score = 0.0
    for i in range(len(b) - 1):
        if b[i] <= beta <= b[i + 1]:
            span = b[i + 1] - b[i]
            t = 0.0 if span <= 0 else (beta - b[i]) / span
            score = _SCORE_AT[i] + t * (_SCORE_AT[i + 1] - _SCORE_AT[i])
            break

    if b[2] < beta < b[3]:
        grade = "green"                       # BALANCED
    elif b[1] < beta < b[4]:
        grade = "yellow"
    else:
        grade = "orange"

    if gear_band is not None and gear is not None:
        lo, hi = gear_band
        if gear < lo or gear > hi:
            score *= float(gear_penalty)

    return grade, float(score), detail

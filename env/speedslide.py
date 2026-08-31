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
}

# Reversing. SDHelper applies no speed floor at all going backwards.
BACKWARD = {
    "road":  {"limit": 0.0, "band": (17.0, 23.0, 29.0, 32.0, 38.0, 44.0)},
    "grass": {"limit": 0.0, "band": (1.0, 6.0, 12.0, 15.0, 21.0, 27.0)},
    "dirt":  {"limit": 0.0, "band": (2.0, 8.0, 14.0, 17.0, 23.0, 29.0)},
}

# SDHelper switches on the raw material name under the front-left wheel, and
# only distinguishes three cases. "Green" is the name TM2020's grass blocks
# actually report - not "Grass", which is why matching on the obvious string
# silently never fires.
_GRASS = ("Green", "Grass", "WetGrass", "Wheat")
_DIRT = ("Dirt", "DirtRoad", "WetDirtRoad", "Sand", "Gravel")

MS_TO_KMH = 3.6

# What the helper paints, and what each colour is worth. Green is the target;
# blue is SDHelper's "you are not sliding usefully" default.
GRADES = ("none", "blue", "orange", "yellow", "green")
_SCORE_AT = (0.0, 0.35, 1.0, 1.0, 0.35, 0.0)


def sd_surface(material_name: str) -> str:
    """Which of SDHelper's three surface cases a material falls into."""
    if material_name in _GRASS:
        return "grass"
    if material_name in _DIRT:
        return "dirt"
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

"""Surface materials and map effects, as the policy sees them.

The plugin has always streamed `mat` - the EPlugSurfaceMaterialId under each
wheel - and the environment has always thrown it away. That meant the policy
could not tell tarmac from grass except indirectly, through a slip coefficient
that arrives after the mistake has already been made.

Two tables live here:

  MATERIALS   id -> name, straight out of Openplanet.h. Used for the tuning UI
              and for the per-surface reward weights, which are keyed by name
              because "Grass" is something you can reason about and 2 is not.

  GROUPS      name -> grip class. The observation gets the *group*, not the raw
              id: there are 81 materials and a track uses maybe six, so a raw
              one-hot would be mostly dead inputs and would not transfer to a
              map with a different palette. Grip class does transfer - ice is
              ice whatever it is called.
"""
from __future__ import annotations

# Index is the material id. Order is EPlugSurfaceMaterialId's, verbatim.
MATERIALS = [
    "Concrete", "Pavement", "Grass", "Ice", "Metal", "Sand",
    "Dirt", "Turbo_Deprecated", "DirtRoad", "Rubber", "SlidingRubber", "Test",
    "Rock", "Water", "Wood", "Danger", "Asphalt", "WetDirtRoad",
    "WetAsphalt", "WetPavement", "WetGrass", "Snow", "ResonantMetal", "GolfBall",
    "GolfWall", "GolfGround", "Turbo2_Deprecated", "Bumper_Deprecated",
    "NotCollidable", "FreeWheeling_Deprecated",
    "TurboRoulette_Deprecated", "WallJump", "MetalTrans", "Stone", "Player", "Trunk",
    "TechLaser", "SlidingWood", "PlayerOnly", "Tech", "TechArmor", "TechSafe",
    "OffZone", "Bullet", "TechHook", "TechGround", "TechWall", "TechArrow",
    "TechHook2", "Forest", "Wheat", "TechTarget", "PavementStair", "TechTeleport",
    "Energy", "TechMagnetic", "TurboTechMagnetic_Deprecated",
    "Turbo2TechMagnetic_Deprecated", "TurboWood_Deprecated", "Turbo2Wood_Deprecated",
    "FreeWheelingTechMagnetic_Deprecated", "FreeWheelingWood_Deprecated",
    "TechSuperMagnetic", "TechNucleus", "TechMagneticAccel", "MetalFence",
    "TechGravityChange", "TechGravityReset", "RubberBand", "Gravel",
    "Hack_NoGrip_Deprecated", "Bumper2_Deprecated",
    "NoSteering_Deprecated", "NoBrakes_Deprecated", "RoadIce", "RoadSynthetic",
    "Green", "Plastic", "DevDebug", "Free3", "XXX_Null",
]

# Grip classes that go into the observation, in one-hot order. Split out from
# the old 6-class version after r/TrackMania "Need help with surfaces"
# (u/_ar_op) put approximate grip figures on each: wood ~100% (+ more steering,
# stronger braking), road/metal ~80%, dirt ~50%, grass ~40%, plastic ~30%,
# ice ~20%. Plastic and wood were BOTH in "road" before, which is wrong on
# every count - they are the two extremes of the grip range, not the middle.
GROUP_NAMES = ("wood", "road", "metal", "plastic",
               "dirt", "grass", "ice", "wet", "other")
N_GROUPS = len(GROUP_NAMES)

_BY_GROUP = {
    # ~100% grip, wider steering, stronger brakes. Its own class, and the only
    # surface where 1-2% wheel wetness already changes everything.
    "wood": ("Wood", "TurboWood_Deprecated", "Turbo2Wood_Deprecated"),
    "road": ("Concrete", "Asphalt", "Pavement", "PavementStair", "RoadSynthetic",
             "Rubber", "Stone", "Tech", "TechGround", "TechArmor", "TechSafe",
             "TechArrow", "Energy"),
    # ~road grip, but the wet-wheels effect is NOT wiped on contact, and magnet
    # blocks add downforce (bigger drift radius). Close enough to share a class.
    "metal": ("Metal", "MetalTrans", "MetalFence", "ResonantMetal", "WallJump",
              "TechMagnetic", "TechSuperMagnetic", "TechMagneticAccel",
              "TechNucleus", "RubberBand"),
    # ~30% grip, springy, "plastic bounce" off side/roof contact. Not road.
    "plastic": ("Plastic",),
    "dirt": ("Dirt", "DirtRoad", "Sand", "Gravel", "Rock", "Forest"),
    "grass": ("Grass", "Wheat", "Green"),
    # ~20% grip; the icy-wheels effect makes any surface slide out. Water AS A
    # SURFACE (skimming across it) sits here - post-2022 "straight is fastest",
    # same as flat ice. Being SUBMERGED is a separate signal (`submerged`).
    "ice": ("Ice", "RoadIce", "Snow", "Water"),
    # Permanently-wet blocks a mapper placed. Kept distinct rather than folded
    # into their dry base: we have not confirmed they raise WetnessValue01, so
    # calling a wet-asphalt block "dry road" could hand the policy a grip
    # figure that is 4x too high. The dynamic `wet` scalar covers wet-from-
    # splashing on a normal surface.
    "wet": ("WetAsphalt", "WetDirtRoad", "WetPavement", "WetGrass",
            "SlidingRubber", "SlidingWood"),
}

GROUPS: dict[str, str] = {}
for _g, _names in _BY_GROUP.items():
    for _n in _names:
        GROUPS[_n] = _g

# id -> group index, precomputed so the hot path is a list lookup.
GROUP_OF_ID = [
    GROUP_NAMES.index(GROUPS.get(name, "other")) for name in MATERIALS
]

# Materials worth showing in the tuning UI. The full list is 81 entries, most
# of which are ShootMania leftovers you will never drive on.
TUNABLE = ("Concrete", "Asphalt", "Pavement", "Plastic", "RoadSynthetic",
           # BOTH grass ids matter. TM2020 reports the ordinary green stuff
           # beside a Stadium road as `Green` (id 76), not `Grass` (id 2) -
           # confirmed by driving a surface test track and watching the wheel
           # materials. Both map to the `grass` group so the physics side was
           # always right, but `Green` was missing here, so the one surface you
           # actually slide onto could not be tuned from the panel at all.
           "Grass", "Green", "WetGrass",
           "Dirt", "DirtRoad", "Sand", "Gravel", "Rock",
           "Ice", "RoadIce", "Snow", "Water", "WetAsphalt",
           "Metal", "MetalFence", "Wood", "Rubber", "NotCollidable")


def material_name(mat_id) -> str:
    try:
        return MATERIALS[int(mat_id)]
    except (ValueError, TypeError, IndexError):
        return "XXX_Null"


def group_index(mat_id) -> int:
    try:
        return GROUP_OF_ID[int(mat_id)]
    except (ValueError, TypeError, IndexError):
        return GROUP_NAMES.index("other")


# Materials that mean "this wheel is on the EDGE of the track", not "this wheel
# is on a driving surface".
#
# Established by driving a surface test track and reading the wheel materials
# against the block names: the raised lip of a road piece reports `Rubber`, and
# the user's rule is that the border is the thing you generally do not want the
# car to touch. Kept deliberately narrow - one material, confirmed by
# measurement - because a false positive here tells the policy it is falling
# off the track when it is not.
#
# NB Rubber stays in the `road` GRIP group as well. Grip and meaning are
# different questions: a kerb is grippy AND it is the edge, and the policy
# wants both answers. This set only feeds the separate `border` observation.
BORDER_MATERIALS = ("Rubber",)
BORDER_IDS = frozenset(
    i for i, name in enumerate(MATERIALS) if name in BORDER_MATERIALS)


def is_border(mat_id) -> bool:
    """Is this wheel on a track border (kerb) rather than a driving surface?"""
    try:
        return int(mat_id) in BORDER_IDS
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Map effects, from the plugin's `dumpmap`.
#
# Four classes, because what the policy needs to know about a thing 40m ahead
# is what it will do to the car, not which of 22 block models it is.
# ---------------------------------------------------------------------------

EFFECT_CLASSES = ("boost", "handicap", "danger", "bumper")

EFFECT_CLASS = {
    "turbo": "boost", "turbo2": "boost", "turbo_roulette": "boost",
    "reactor": "boost", "reactor2": "boost", "forced_accel": "boost",
    "cruise": "boost",

    "no_grip": "handicap", "slow_motion": "handicap", "no_steer": "handicap",
    "no_brakes": "handicap", "no_engine": "handicap",

    "fragile": "danger", "reset": "danger",

    "bumper": "bumper", "bumper2": "bumper", "bumper_barrelroll": "bumper",
}


def effect_class(kind: str) -> str | None:
    """Vehicle-switch gates are deliberately unclassified: changing car changes
    the whole physics model, which is not something a distance-ahead scalar can
    usefully warn about."""
    return EFFECT_CLASS.get(kind)


def cell_to_world(cell, base_height: int,
                  block=(32.0, 8.0, 32.0)) -> tuple[float, float, float]:
    """Grid cell -> world position of its centre.

    Matches the game's own coordToPosition: X and Z are block-sized and
    zero-based, Y is offset by the map's DecoBaseHeightOffset. Half a block is
    added in X and Z so the result is the cell's middle rather than its corner.
    """
    cx, cy, cz = cell
    return ((cx + 0.5) * block[0],
            (cy - base_height) * block[1],
            (cz + 0.5) * block[2])

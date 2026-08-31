"""Ground-sensing rays against the map's own block grid.

There is no raycast API in TM2020 - checked, it does not exist - and screen
capture would need a GPU budget and a trained encoder to learn what the game
already knows. But every block in a map exposes the exact grid cells it
occupies (`CGameCtnBlock.BlockUnitsE` -> `AbsoluteOffset`), so the plugin can
export a complete occupancy grid once per map and the rays can be marched
against that, in numpy, for microseconds a step.

## What these rays actually measure

Not wall distance. **Distance until the ground runs out.**

A block unit marks a cell the block occupies in the editor, not where the
drivable surface sits inside it. A road block's cell reads "solid" even though
you drive straight through it, so casting for the first solid cell would report
a hit immediately in every direction and tell you nothing. Inverting the test
fixes it: march until the cell is *empty*, and what comes back is how far you
can go that way before there is no track under you.

## What it is good for

The grid is 32 x 8 x 32 metres per cell, so a ray resolves to whole blocks.
That is enough to answer "which way does the track continue" and "how many
blocks wide is it here", which is what stage-one exploration needs to find the
finish on a map nobody has driven. It is **not** precise enough to place a car
against a barrier - the reference line and the live per-wheel surface readings
do that job, and this sits alongside them rather than replacing them.

Being honest about that resolution matters: a policy handed a 32m-quantised
"distance to edge" and told it is precise would learn to trust a number that
cannot support the decision.
"""
from __future__ import annotations

import json
import os

import numpy as np

# Metres. Matches the game's own block grid; the plugin reports it too and the
# reported value wins if they ever disagree.
BLOCK = (32.0, 8.0, 32.0)

# How far a ray looks, in cells. Eight cells is 256m - well past anything a car
# can act on, and the cost is linear in this so there is no reason to go wider.
MAX_CELLS = 8

# Beam directions in the car's horizontal plane, as angles from straight ahead.
# Forward-weighted: where the track goes matters far more than what is behind,
# and evenly spacing 16 beams around a full circle spends half of them on
# information the car cannot use.
BEAM_ANGLES_DEG = (0, 10, -10, 22, -22, 38, -38, 58, -58, 80, -80,
                   110, -110, 145, -145, 180)


# How a block's footprint sits relative to its stored Coord, per cardinal
# direction. Nadeo's convention is not documented and the obvious guess is
# wrong often enough to matter, so all four plausible ones are implemented and
# `pick_convention` chooses by testing against positions known to be on-track.
#
# For each direction, (which axis the footprint's X extends along, sign) etc.
# 0 = footprint grows +x/+z from Coord and swaps axes on odd rotations.
N_CONVENTIONS = 4

# Which block models count as ground you can drive on.
#
# This is not a nicety. A real map came back as 2350 blocks of which **2304
# were "Grass"** - TM2020 paves the entire 48x48 map with a base plane - and
# 46 were road. Counting all of them made every cell of the map solid, so the
# rays only ever found the map boundary and reported "ground everywhere" in all
# 16 directions. The lidar was live and telling the policy nothing.
DRIVABLE_HINTS = (
    "road", "platform", "track", "slope", "loop", "tilt", "bump", "ring",
    "pipe", "wall", "start", "finish", "checkpoint", "multilap", "turbo",
    "boost", "gate", "circuit", "speed", "dirt", "ice", "plastic", "grass"
    "core", "base",
)
# Checked first, so "RoadTechStraight" stays drivable while a bare "Grass"
# base tile does not.
NOT_DRIVABLE = (
    "grass", "water", "deco", "trees", "tree", "scenery", "structure",
    "fabric", "cliff", "rock", "building", "sign", "light", "pole", "fence",
)


def is_drivable(name: str) -> bool:
    """Unknown blocks are treated as NOT drivable, deliberately.

    The two errors are not symmetric. Calling scenery drivable tells the policy
    there is road where there is a building - it will drive at it. Calling road
    scenery just makes the track read narrower than it is, which keeps the car
    central. When in doubt, be wrong in the direction that does not crash.
    """
    n = (name or "").lower()
    if any(k in n for k in NOT_DRIVABLE):
        return False
    return any(k in n for k in DRIVABLE_HINTS)


def expand_boxes(boxes: np.ndarray, convention: int = 0,
                 names: list[str] | None = None) -> np.ndarray:
    """Per-block (x, y, z, dir, sx, sy, sz[, name index]) -> occupied cells.

    With `names`, non-drivable models are dropped entirely.
    """
    keep = None
    if names is not None and boxes.shape[1] >= 8:
        keep = np.asarray([is_drivable(n) for n in names], dtype=bool)

    out: list[tuple[int, int, int]] = []
    for row in boxes.tolist():
        x, y, z, d, sx, sy, sz = row[:7]
        if keep is not None and len(row) > 7:
            ni = row[7]
            if 0 <= ni < len(keep) and not keep[ni]:
                continue
        d &= 3
        sx, sy, sz = max(sx, 1), max(sy, 1), max(sz, 1)
        # On a 90/270 degree rotation the footprint's X and Z extents swap.
        ex, ez = (sz, sx) if d % 2 else (sx, sz)
        if convention == 0:          # Coord is the min corner, always
            ox, oz = 0, 0
        elif convention == 1:        # Coord is the pre-rotation min corner
            ox = -(ex - 1) if d in (1, 2) else 0
            oz = -(ez - 1) if d in (2, 3) else 0
        elif convention == 2:        # mirror of 1
            ox = -(ex - 1) if d in (2, 3) else 0
            oz = -(ez - 1) if d in (1, 2) else 0
        else:                        # Coord is the centre
            ox, oz = -(ex // 2), -(ez // 2)
        for i in range(ex):
            for j in range(sy):
                for k in range(ez):
                    out.append((x + ox + i, y + j, z + oz + k))
    if not out:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(out, dtype=np.int32)


def pick_convention(dump: dict, on_track, verbose: bool = True) -> int:
    """Choose the footprint convention by testing, not by guessing.

    `on_track` is a list of world positions the car is known to have occupied
    or must occupy - the spawn, the checkpoints, the finish, or every sample of
    a recorded lap. Whichever convention puts the most of them inside an
    occupied cell is the right one, and if none does noticeably better than the
    others that is worth knowing too.
    """
    best, best_score, scores = 0, -1.0, []
    for c in range(N_CONVENTIONS):
        grid = OccupancyGrid.from_dump(dump, convention=c)
        if grid is None or not len(grid):
            scores.append(0.0)
            continue
        hit = 0
        for p in on_track:
            cx, cy, cz = grid.world_to_cell(p)
            if any(grid.is_solid(cx, cy + dy, cz) for dy in (-1, 0, 1)):
                hit += 1
        score = hit / max(len(on_track), 1)
        scores.append(score)
        if score > best_score:
            best, best_score = c, score
    if verbose:
        print("  footprint convention scores: "
              + ", ".join(f"#{i} {s:.0%}" for i, s in enumerate(scores))
              + f"  -> using #{best}", flush=True)
    return best


class OccupancyGrid:
    """Which 32x8x32m cells of the map contain a block."""

    def __init__(self, cells: np.ndarray, base_height: int = 8,
                 block=BLOCK, map_uid: str | None = None):
        self.cells = np.asarray(cells, dtype=np.int32).reshape(-1, 3)
        self.base_height = int(base_height)
        self.block = tuple(float(b) for b in block)
        self.map_uid = map_uid
        # A set of packed ints beats an ndarray membership test by orders of
        # magnitude here, and the grid is small enough that packing is exact.
        self._solid = set(self._pack(self.cells))

    @staticmethod
    def _pack(c: np.ndarray) -> np.ndarray:
        # Coordinates are non-negative uints from the game; 12 bits each is
        # 4096 cells per axis, far beyond any real map.
        return (c[:, 0].astype(np.int64) << 24) \
             | (c[:, 1].astype(np.int64) << 12) \
             | c[:, 2].astype(np.int64)

    def __len__(self) -> int:
        return len(self._solid)

    def is_solid(self, cx: int, cy: int, cz: int) -> bool:
        if cx < 0 or cy < 0 or cz < 0:
            return False
        return ((int(cx) << 24) | (int(cy) << 12) | int(cz)) in self._solid

    def world_to_cell(self, pos) -> tuple[int, int, int]:
        """Inverse of the game's coordToPosition."""
        x, y, z = pos
        return (int(np.floor(x / self.block[0])),
                int(np.floor(y / self.block[1])) + self.base_height,
                int(np.floor(z / self.block[2])))

    # -- persistence ------------------------------------------------------

    @classmethod
    def from_dump(cls, dump: dict, convention: int = 0) -> "OccupancyGrid | None":
        """Build from the plugin's `dumpmap occupancy`.

        The plugin sends per-block boxes - coord, cardinal direction, footprint
        size - rather than finished cells, because how a footprint rotates
        about its anchor is the one genuinely uncertain part and resolving it
        here means iterating without a game reload per attempt. See
        `expand_boxes` and `pick_convention`.

        The older "cells" form is still accepted so a cached map from before
        the change still loads.
        """
        boxes = dump.get("boxes")
        if boxes:
            arr = np.asarray(boxes, dtype=np.int32)
            # 8 wide once block names arrived; 7 for a dump from before that.
            width = 8 if arr.size % 8 == 0 and dump.get("names") else 7
            if arr.size % width:
                return None
            cells = expand_boxes(arr.reshape(-1, width), convention,
                                 dump.get("names"))
        else:
            flat = dump.get("cells")
            if not flat:
                return None
            arr = np.asarray(flat, dtype=np.int32)
            if arr.size % 3:
                return None
            cells = arr.reshape(-1, 3)
        grid = cls(cells,
                   base_height=dump.get("base_height", 8),
                   block=dump.get("block_size", BLOCK),
                   map_uid=dump.get("map"))
        # Keep the per-block boxes + names so env.roadtrace can trace the road
        # ribbon later without another dump; None on an old cells-only cache.
        grid.boxes = dump.get("boxes")
        grid.names = dump.get("names")
        return grid

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        doc = {"map": self.map_uid, "base_height": self.base_height,
               "block_size": list(self.block),
               "cells": self.cells.reshape(-1).tolist()}
        # Round-trip the raw block list so a cached map can still be traced by
        # env.roadtrace / env.routemodel.
        if getattr(self, "boxes", None):
            doc["boxes"] = self.boxes
        if getattr(self, "names", None):
            doc["names"] = self.names
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "OccupancyGrid | None":
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        return cls.from_dump(d)


class Lidar:
    """Fixed set of ground rays, cast in the car's frame."""

    def __init__(self, grid: OccupancyGrid, angles=BEAM_ANGLES_DEG,
                 max_cells: int = MAX_CELLS):
        self.grid = grid
        self.max_cells = max_cells
        rad = np.radians(np.asarray(angles, dtype=np.float64))
        # Unit vectors in the (forward, left) plane, applied to the car's own
        # basis at query time so nothing depends on world heading.
        self.fwd = np.cos(rad)
        self.left = np.sin(rad)
        self.n = len(rad)
        self.max_range = max_cells * max(grid.block[0], grid.block[2])

    def cast(self, pos, dir_, left_) -> np.ndarray:
        """Distance along each beam until the ground runs out, in metres.

        Marched a half-cell at a time so a beam cannot skip diagonally between
        two solid cells through the gap at their shared corner.
        """
        pos = np.asarray(pos, dtype=np.float64)
        dir_ = np.asarray(dir_, dtype=np.float64)
        left_ = np.asarray(left_, dtype=np.float64)

        step = min(self.grid.block[0], self.grid.block[2]) * 0.5
        steps = int(self.max_cells * 2)
        out = np.full(self.n, self.max_range, dtype=np.float64)

        for b in range(self.n):
            ray = dir_ * self.fwd[b] + left_ * self.left[b]
            norm = np.linalg.norm(ray)
            if norm < 1e-9:
                continue
            ray = ray / norm
            for k in range(1, steps + 1):
                d = k * step
                p = pos + ray * d
                cx, cy, cz = self.grid.world_to_cell(p)
                # Check a three-cell vertical window, not just the car's own.
                # Cells are 8m tall and the car's reported Position sits
                # somewhere in the middle of the block it is driving on, so it
                # lands one cell either side depending on ride height, camber
                # and whether the road is climbing. Looking only at the car's
                # own layer reports "no ground" for a road that is plainly
                # there - which reads as an edge in every direction at once.
                if not (self.grid.is_solid(cx, cy, cz)
                        or self.grid.is_solid(cx, cy - 1, cz)
                        or self.grid.is_solid(cx, cy + 1, cz)):
                    out[b] = d
                    break
        return out

    def normalised(self, pos, dir_, left_) -> np.ndarray:
        return (self.cast(pos, dir_, left_) / self.max_range).astype(np.float64)


def load_or_fetch(root: str, map_uid: str | None, fetch,
                  on_track=None) -> OccupancyGrid | None:
    """Cached per map. The dump is a few hundred KB over the socket and the map
    does not change while you are driving it, so it is fetched once, resolved,
    and read from disk on every later run.

    `on_track` are world positions the car must be able to occupy - the spawn,
    checkpoints and finish. They are used once, when the map is first scanned,
    to pick the footprint convention.
    """
    if not map_uid:
        return None
    path = os.path.join(root, "maps", f"{map_uid}.json")
    grid = OccupancyGrid.load(path)
    if grid is not None and len(grid):
        return grid
    dump = fetch()
    if not dump or not dump.get("ok"):
        return None
    conv = pick_convention(dump, list(on_track or [])) if on_track else 0
    grid = OccupancyGrid.from_dump(dump, convention=conv)
    if grid is None:
        return None
    try:
        # Saved as finished cells, so a later run does not have to re-resolve.
        grid.save(path)
    except OSError:
        pass
    return grid

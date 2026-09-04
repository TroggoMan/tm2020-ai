"""A drivable centerline traced from road-block connectivity, in 3D.

The occupancy grid (``env.lidar``) rasterises every block to flat 32 m cells
and throws the sequence away. On a compact campaign track the road stacks over
itself - bridges, road-over-road - so the flattened blob self-intersects and
any shortest path through it recovers a different checkpoint order every run.

This keeps the blocks as blocks. The plugin's ``dumpmap occupancy`` gives, per
block: integer grid ``coord`` (x, y, z), a cardinal ``dir``, footprint
``size`` in cells, and the model name. From that:

  1. each road block contributes its DECK footprint - the top cell layer, in
     (cx, cz), at a single deck height ``cy = y + sy - 1``;
  2. two blocks are neighbours when their deck footprints are 4-adjacent in XZ
     *and* within ~1 level in Y - so a deck at y=8 never links to the road at
     y=6 beneath it;
  3. the ribbon is then a near-simple path: start and finish blocks come out
     degree 1, the interior mostly degree 2. Walk it start -> finish, and at
     the handful of junctions take the neighbour that best continues the
     current heading (roads do not kink 135 degrees between pieces);
  4. the checkpoint ORDER is the order the ``RoadTechCheckpoint`` blocks are
     walked - read off, not guessed;
  5. the deck-centre sequence is splined into a Centerline; width comes from
     the block family.

No per-block spline or clip data is needed - only coord + dir + size + name,
all of which the existing dump already carries.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

from .centerline import Centerline

# Support structure, not track surface: pillars, bases, deadends. These carry
# road-ish names but you do not drive on them, and their tall footprints bridge
# unrelated levels.
_NOT_SURFACE = re.compile(
    r"Pillar|Deadend|DeadEnd|Support|Structure|"
    r"BasePillar|BaseClip|Podium", re.I)

# Blocks whose extra cell layers are an ARCH OVER the road, not fill under it.
# A checkpoint is a road piece with a gantry on top, so its bounding box is 2-3
# cells tall while the surface you drive on is the BOTTOM one. Measured against
# real driven positions, the top-cell rule put the line a full block (8m) above
# the road at every checkpoint - which is also why every gate read ~9m off the
# line it was supposed to sit on.
_GANTRY = re.compile(r"Checkpoint|^RoadTechStart|^RoadTechFinish", re.I)

# How far above the deck cell's floor the car's reported position sits. Taken
# from telemetry, not assumed: on every single-layer block the driven y is
# exactly (deck - base_height) * block_y + 2.
RIDE_M = 2.0

# Does this piece have BARRIERS along its edges, or can you drive straight off?
#
# The user's rule, checked against both surveyed maps: Platform* pieces have no
# sides, Road* pieces do - with one exception they spotted themselves,
# `OpenTechRoadStraight`, which is road-named but open with penalty grass
# verges. So "Open" has to be tested BEFORE "Road", and it cannot be a plain
# prefix check.
#
# This matters more than it looks. On the training track every single edgeless
# block sits between checkpoint 4 and checkpoint 5 - seven consecutive Platform
# pieces, one of them named BaseWithHole24m - and that is precisely the section
# the policy has never once completed.
_NO_SIDES = re.compile(r"^Open|Platform", re.I)

# ...and a third case, found by building a bobsleigh section and dumping it:
# the walled pieces are named `RoadIce*WithWall*` - `WithWall` is an explicit
# token in the block name, and there is no "bobsleigh" string anywhere. A
# boolean would file `RoadIceStraight` and `RoadIceWithWallCurve5` as the same
# thing, when one is an ordinary barriered road and the other is a banked wall
# you can deliberately ride. So this is a DEGREE of containment, not a flag.
_WALLED = re.compile(r"WithWall|Bobsleigh|Tube|Pipe(?!line)", re.I)

NO_SIDES, BARRIERED, WALLED = 0.0, 0.5, 1.0


def _containment(name: str) -> float:
    """How much the track holds the car in, here.

    0.0  drive straight off  - Platform*, Open*
    0.5  ordinary barriers   - most Road*
    1.0  walled / rideable   - Road*WithWall*, bobsleigh, tubes

    Kept on one scale rather than split into more inputs because it is a single
    physical quantity and the policy only ever needs to know "how much room for
    error is there ahead".
    """
    n = name or ""
    if _NO_SIDES.search(n):
        return NO_SIDES
    if _WALLED.search(n):
        return WALLED
    return BARRIERED


def _has_sides(name: str) -> bool:
    """Kept for callers that only want the boolean question."""
    return _containment(name) > NO_SIDES


# family -> (is_cp, is_start, is_finish, half_width_m)
#
# Start/finish are matched by a loose "<track family> ... Start|Finish" rule,
# not just "^RoadTechStart". The platform sets name their end pieces
# PlatformIceStart / PlatformDirtFinish / etc., and without these the walk
# found no start block, returned nothing, and the whole map got "barriered
# everywhere" with no roadtrace - which is exactly the Ice-platform failure.
_TRACK_ROOT = r"(?:RoadTech|RoadDirt|RoadIce|RoadBump|RoadWater|Road\w*|" \
              r"Platform\w*|Track\w*|DirtRoad\w*)"
_FAM = [
    (re.compile(r"Checkpoint", re.I),                       (True,  False, False, 6.0)),
    (re.compile(_TRACK_ROOT + r".*Start", re.I),            (False, True,  False, 8.0)),
    (re.compile(_TRACK_ROOT + r".*Finish", re.I),           (False, False, True,  8.0)),
    (re.compile(r"^RoadTechStart", re.I),                   (False, True,  False, 6.0)),
    (re.compile(r"^RoadTechFinish", re.I),                  (False, False, True,  6.0)),
    (re.compile(r"Platform", re.I),                         (False, False, False, 8.0)),
    (re.compile(r"Road", re.I),                             (False, False, False, 6.0)),
]


def _family(name):
    for rx, tup in _FAM:
        if rx.search(name):
            return tup
    return (False, False, False, 6.0)


class _Block:
    __slots__ = ("name", "x", "y", "z", "dir", "sx", "sy", "sz",
                 "is_cp", "is_start", "is_finish", "hw", "cells", "cy", "c_xz",
                 "anchor")

    def __init__(self, name, coord, direction, size):
        self.name = name
        self.x, self.y, self.z = (int(v) for v in coord)
        self.dir = int(direction) % 4
        self.sx, self.sy, self.sz = (int(max(s, 1)) for s in size)
        self.is_cp, self.is_start, self.is_finish, self.hw = _family(name)
        ex, ez = (self.sz, self.sx) if self.dir % 2 else (self.sx, self.sz)
        self.cells = {(self.x + i, self.z + j)
                      for i in range(ex) for j in range(ez)}
        # The deck level this block CERTAINLY sits at, or None when it spans
        # levels and has to be inferred from its neighbours.
        #
        #   * a gantry block's extra cells are above the road  -> deck = y
        #   * a single-layer block has nowhere else to be      -> deck = y
        #   * anything else (ramps, and road-on-terrain whose extra cells are
        #     hill fill below) is ambiguous from the box alone. Resolved in
        #     _deck_levels() by interpolating between the certain ones along
        #     the chain, which gets ramps right for free: a slope between a
        #     deck-6 block and a deck-7 block simply climbs from 6 to 7.
        if _GANTRY.search(name) or self.sy == 1:
            self.anchor = float(self.y)
        else:
            self.anchor = None
        # Adjacency only needs a level that is within a cell or two of the
        # truth, so it uses the certain value when there is one and the top
        # layer otherwise. Measured: identical coverage either way.
        self.cy = self.y if self.anchor is not None else self.y + self.sy - 1
        self.c_xz = np.array([np.mean([c[0] for c in self.cells]) + 0.5,
                              np.mean([c[1] for c in self.cells]) + 0.5])

    def world_centre(self, block_size, base_height, level=None):
        bx, by, bz = block_size
        lv = self.cy if level is None else level
        return np.array([self.c_xz[0] * bx,
                         (lv - base_height) * by + RIDE_M,
                         self.c_xz[1] * bz])


def _load_blocks(boxes, names, drivable):
    arr = np.asarray(boxes, dtype=int).reshape(-1, 8)
    out = []
    for x, y, z, d, sx, sy, sz, ni in arr.tolist():
        nm = names[ni]
        if not drivable(nm) or _NOT_SURFACE.search(nm):
            continue
        out.append(_Block(nm, (x, y, z), d, (sx, sy, sz)))
    return out


def _adjacent(a: _Block, b: _Block) -> bool:
    if abs(a.cy - b.cy) > 2:
        return False
    for (cx, cz) in a.cells:
        if ((cx + 1, cz) in b.cells or (cx - 1, cz) in b.cells
                or (cx, cz + 1) in b.cells or (cx, cz - 1) in b.cells
                or (cx, cz) in b.cells):
            return True
    return False


def _walk(blocks):
    """Maximum-coverage start->finish path over the adjacency graph.

    The road is one physical ribbon, but the graph carries a few extra edges
    where it passes over/under itself within the Y tolerance. A *shortest* path
    hops those and skips real sections; the path that visits the MOST blocks is
    the true lap (a full Hamiltonian start->finish path when one exists). The
    graph is nearly linear so this is cheap. Heading continuity only orders the
    search so a good path is found first. Returns the ordered block-index list.
    """
    n = len(blocks)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if _adjacent(blocks[i], blocks[j]):
                adj[i].append(j)
                adj[j].append(i)

    start = next((i for i, b in enumerate(blocks) if b.is_start), None)
    finish = next((i for i, b in enumerate(blocks) if b.is_finish), None)
    if start is None or finish is None:
        return None, adj

    best = []
    budget = [400_000]      # DFS step cap; the graph is nearly a path so this
                            # is never approached in practice

    def heading(prev, cur):
        if prev is None:
            return None
        v = blocks[cur].c_xz - blocks[prev].c_xz
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-6 else None

    def dfs(cur, prev, seen, path):
        nonlocal best
        if budget[0] <= 0:
            return
        budget[0] -= 1
        if cur == finish:
            # keep the LONGEST start->finish path, not the first one found
            if len(path) > len(best):
                best = path[:]
            return
        h = heading(prev, cur)

        def score(x):
            if h is None:
                return 0.0
            v = blocks[x].c_xz - blocks[cur].c_xz
            nv = np.linalg.norm(v)
            return float(h @ (v / nv)) if nv > 1e-6 else -1.0

        nbrs = [x for x in adj[cur] if x not in seen]
        # explore best-heading first so a good path is found early, but do not
        # prune - the far loop is reached by briefly turning away from finish
        for x in sorted(nbrs, key=score, reverse=True):
            if h is not None and score(x) < -0.75:   # a near-U-turn is spurious
                continue
            seen.add(x)
            path.append(x)
            dfs(x, cur, seen, path)
            path.pop()
            seen.discard(x)
            if len(best) >= n:            # full cover - nothing longer possible
                return

    dfs(start, None, {start}, [start])
    return (best or None), adj


def _deck_levels(chain):
    """A deck level per block along the walked chain.

    Blocks that know their own level (gantries, single-layer pieces) anchor the
    sequence; runs of ambiguous blocks between two anchors are interpolated.
    That is what makes ramps come out right without having to recognise a ramp:
    a two-cell slope sitting between a deck-6 and a deck-7 neighbour is simply
    the half-way point, and a four-cell hill straight between two deck-9 blocks
    is flat at 9.
    """
    n = len(chain)
    lvl = [b.anchor for b in chain]
    known = [i for i, v in enumerate(lvl) if v is not None]
    if not known:                       # nothing certain: fall back to the box
        return [float(b.y + b.sy - 1) for b in chain]
    for a, b in zip(known, known[1:]):
        if b - a > 1:
            for m in range(a + 1, b):
                t = (m - a) / (b - a)
                lvl[m] = lvl[a] + (lvl[b] - lvl[a]) * t
    for m in range(known[0]):
        lvl[m] = lvl[known[0]]
    for m in range(known[-1] + 1, n):
        lvl[m] = lvl[known[-1]]
    return lvl


def _joint_xz(a: _Block, b: _Block, block_size):
    """World (x, z) of the middle of the face two consecutive blocks share.

    The road crosses that face; it does not generally pass through either
    block's centre. Using centres put the line through the middle of a 32 m
    cell, which on a curve is off the tarmac entirely - the face midpoints are
    on it by construction, at both the entry and the exit of every piece.
    """
    bx, _, bz = block_size
    pairs = [((cx, cz), (cx + dx, cz + dz))
             for (cx, cz) in a.cells
             for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
             if (cx + dx, cz + dz) in b.cells]
    if not pairs:                       # overlapping footprints (stacked pieces)
        return np.array([(a.c_xz[0] + b.c_xz[0]) / 2.0 * bx,
                         (a.c_xz[1] + b.c_xz[1]) / 2.0 * bz])
    pts = [(((ca[0] + cb[0]) / 2.0 + 0.5) * bx,
            ((ca[1] + cb[1]) / 2.0 + 0.5) * bz) for ca, cb in pairs]
    return np.asarray(pts, float).mean(axis=0)


def _catmull(points, per_seg=8):
    p = np.asarray(points, float)
    if len(p) < 4:
        return p
    ext = np.vstack([p[0], p, p[-1]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in np.linspace(0, 1, per_seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-1])
    return np.asarray(out)


def _cluster_xz(points, radius):
    """Single-linkage cluster of world (x, y, z) points by XZ distance;
    returns one centroid per cluster."""
    pts = [np.asarray(p, float) for p in points]
    clusters: list[list] = []
    for p in pts:
        hit = [k for k, c in enumerate(clusters)
               if any(math.hypot(p[0] - q[0], p[2] - q[2]) <= radius
                      for q in c)]
        if not hit:
            clusters.append([p])
            continue
        merged = [p]
        for k in sorted(hit, reverse=True):
            merged += clusters.pop(k)
        clusters.append(merged)
    return [np.mean(np.asarray(c, float), axis=0) for c in clusters]


def _edge_dist(covered, cxw, czw, px, pz, bx, bz, max_m=48.0):
    """Metres from world (cxw, czw) to the platform edge along +/-(px, pz)."""
    step = min(bx, bz) * 0.5
    reach = []
    for sgn in (1.0, -1.0):
        d = 0.0
        while d < max_m:
            d += step
            cx = int((cxw + px * sgn * d) // bx)
            cz = int((czw + pz * sgn * d) // bz)
            if (cx, cz) not in covered:
                break
        reach.append(d)
    return max(2.0, min(reach))


def _platform_centerline(blocks, base_height, block_size, spawn, checkpoints,
                         finish, spacing, verbose):
    """A flat platform field has no defined path - any line across it is on the
    road - so `_walk` treats the tiles as a maze and boustrophedons through
    every one (out-and-back, ~1.9 km for a 0.5 km map). Instead: run the line
    straight from spawn through the checkpoint(s) to the finish, snapped
    laterally to the middle of whatever platform is under it, with the
    half-width measured out to the real edge.
    """
    bx, by, bz = block_size
    covered = {}                         # (cx, cz) -> deck level
    for b in blocks:
        for cell in b.cells:
            covered[cell] = b.cy
    if not covered:
        return None

    s, f = np.asarray(spawn, float), np.asarray(finish, float)
    cps = _cluster_xz(checkpoints, max(bx, bz) * 1.5) if checkpoints else []
    axis = np.array([f[0] - s[0], f[2] - s[2]])
    axis = axis / (np.linalg.norm(axis) or 1.0)
    cps.sort(key=lambda c: np.dot([c[0] - s[0], c[2] - s[2]], axis))
    anchors = [s] + cps + [f]

    route = []                           # dense world-XZ polyline through anchors
    for a, b in zip(anchors, anchors[1:]):
        d = math.hypot(b[0] - a[0], b[2] - a[2])
        n = max(2, int(d / (min(bx, bz) * 0.5)))
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            route.append((a[0] + (b[0] - a[0]) * t, a[2] + (b[2] - a[2]) * t))
    route.append((anchors[-1][0], anchors[-1][2]))

    R = 2
    pts, hw, on = [], [], 0
    for i, (wx, wz) in enumerate(route):
        cx0, cz0 = int(round(wx / bx)), int(round(wz / bz))
        near = [(cx, cz) for cx in range(cx0 - R, cx0 + R + 1)
                for cz in range(cz0 - R, cz0 + R + 1) if (cx, cz) in covered]
        j = min(i + 1, len(route) - 1)
        tvx, tvz = route[j][0] - wx, route[j][1] - wz
        tn = math.hypot(tvx, tvz) or 1.0
        px, pz = -tvz / tn, tvx / tn      # unit perpendicular, world XZ
        if near:
            on += 1
            ncx = np.mean([c[0] for c in near]) + 0.5
            ncz = np.mean([c[1] for c in near]) + 0.5
            cxw, czw = ncx * bx, ncz * bz
            lvl = float(np.mean([covered[c] for c in near]))
            hwm = _edge_dist(covered, cxw, czw, px, pz, bx, bz)
        else:
            cxw, czw, lvl, hwm = wx, wz, float(base_height), 8.0
        pts.append([cxw, (lvl - base_height) * by + RIDE_M, czw])
        hw.append(hwm)

    pts = np.asarray(pts, float)
    line = Centerline(_catmull(pts), spacing=spacing)
    rp, lp = pts[:, [0, 2]], np.asarray(line.points, float)[:, [0, 2]]
    idx = np.argmin(((lp[:, None, 0] - rp[None, :, 0]) ** 2 +
                     (lp[:, None, 1] - rp[None, :, 1]) ** 2), axis=1)
    hw_line = np.asarray(hw, float)[idx]
    sides_line = np.full(len(line.points), _containment("Platform"))
    cov = on / max(len(route), 1)
    if verbose:
        print(f"  roadtrace: platform field, {len(covered)} cells, straight "
              f"spawn -> {len(cps)} cp -> finish, {line.length:.0f} m, "
              f"{cov * 100:.0f}% over platform")
    return {"order": list(range(len(cps))), "line": line,
            "half_width": hw_line, "sides": sides_line,
            "blocks": [b.name for b in blocks], "coverage": cov,
            "complete": cov > 0.8}


def build_road_trace(boxes, names, base_height, block_size,
                     spawn, checkpoints, finish, drivable=None,
                     spacing: float = 2.0, verbose: bool = True):
    """boxes / names: straight from dumpmap occupancy. spawn / finish: world
    (x, y, z). checkpoints: world (x, y, z) list, any order. Returns
    {order, line, half_width, blocks, coverage} or None."""
    if drivable is None:
        from .lidar import is_drivable as drivable
    blocks = _load_blocks(boxes, names, drivable)
    if not blocks:
        if verbose:
            print("  roadtrace: no road blocks in the dump")
        return None

    # A platform field is not a ribbon - walking it block-by-block just snakes.
    # Route straight through the checkpoints instead. Road-block maps (RoadTech*
    # etc.) are untouched: they have no Platform* tiles.
    plat = sum(1 for b in blocks if re.search(r"Platform", b.name, re.I))
    if spawn is not None and finish is not None and plat >= 0.6 * len(blocks):
        pc = _platform_centerline(blocks, base_height, block_size, spawn,
                                  checkpoints or [], finish, spacing, verbose)
        if pc is not None:
            return pc

    order_idx, adj = _walk(blocks)
    if not order_idx:
        if verbose:
            print("  roadtrace: no start/finish block, or graph disconnected")
        return None

    chain = [blocks[i] for i in order_idx]
    cov = len(chain) / len(blocks)
    reached = chain[-1].is_finish
    complete = cov >= 0.7 and reached
    if not complete and verbose:
        print(f"  roadtrace: partial - {len(chain)}/{len(blocks)} blocks, "
              f"reached_finish={reached}. Returning a best-effort estimate; "
              f"edit the gaps in the panel.")

    bx, by, bz = block_size
    cps_w = [np.asarray(c, float) for c in checkpoints]

    # Match each checkpoint block to the landmark the game reports for it, in
    # XZ (a block is 32m square, checkpoints are far further apart than that,
    # and height is the thing we are trying to establish so it cannot be part
    # of the test). That fixes the walk order AND pins the deck height: the
    # landmark is the game's own statement of where the checkpoint is, which
    # beats anything inferred from the bounding box - and it is the only way to
    # get a banked checkpoint right, where the surface is not on a cell floor
    # at all.
    seq = []
    for i, b in enumerate(chain):
        if not b.is_cp:
            continue
        here = np.array([b.c_xz[0] * bx, b.c_xz[1] * bz])
        dists = [float(np.linalg.norm(here - np.array([c[0], c[2]])))
                 for c in cps_w]
        j = int(np.argmin(dists))
        if dists[j] < 45.0 and j not in seq:
            seq.append(j)
            b.anchor = (float(cps_w[j][1]) - RIDE_M) / by + base_height
    for j in range(len(cps_w)):
        if j not in seq:
            seq.append(j)

    levels = _deck_levels(chain)

    # The line runs spawn -> every block-to-block joint -> finish. Joints, not
    # block centres: the road crosses the shared face, so those points are on
    # the tarmac at the entry and exit of every piece, and their height is the
    # deck level the two neighbours agree on.
    bx, by, bz = block_size
    pts = [np.asarray(spawn, float)]
    anchors = [chain[0]]
    for i in range(len(chain) - 1):
        xz = _joint_xz(chain[i], chain[i + 1], block_size)
        y = ((levels[i] + levels[i + 1]) / 2.0 - base_height) * by + RIDE_M
        pts.append(np.array([xz[0], y, xz[1]]))
        anchors.append(chain[i + 1])
    pts.append(np.asarray(finish, float))
    anchors.append(chain[-1])
    pts = np.asarray(pts, float)
    line = Centerline(_catmull(pts), spacing=spacing)

    hw = np.empty(len(line.points))
    sides = np.empty(len(line.points))
    for k, q in enumerate(line.points):
        j = int(np.argmin(np.linalg.norm(pts - q, axis=1)))
        blk = anchors[min(j, len(anchors) - 1)]
        hw[k] = blk.hw
        sides[k] = _containment(blk.name)

    if verbose:
        print(f"  roadtrace: {len(chain)}/{len(blocks)} blocks "
              f"({cov*100:.0f}% cover), length {line.length:.0f} m, "
              f"{len(line.points)} pts")
        print(f"  roadtrace: checkpoint order {seq}")
    return {"order": seq, "line": line, "half_width": hw, "sides": sides,
            "blocks": [b.name for b in chain], "coverage": cov,
            "complete": complete}


# --- cache ----------------------------------------------------------------

def cache_path(root, uid):
    return os.path.join(root, "maps", f"{uid}.roadtrace.json")


def save(root, uid, model):
    line = model["line"]
    doc = {"map": uid, "order": model["order"],
           "coverage": model.get("coverage"),
           "blocks": model.get("blocks", []),
           "points": line.points.tolist(),
           "half_width": [round(float(x), 2) for x in model["half_width"]],
           # round(), NOT int(): containment is 0.0 / 0.5 / 1.0 and int()
           # would silently collapse every barriered road (0.5) to "no sides".
           "sides": [round(float(x), 2) for x in model.get("sides", [])]}
    os.makedirs(os.path.dirname(cache_path(root, uid)), exist_ok=True)
    with open(cache_path(root, uid), "w") as f:
        json.dump(doc, f)


def load(root, uid, spacing: float = 2.0):
    p = cache_path(root, uid)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        doc = json.load(f)
    line = Centerline(np.asarray(doc["points"], float), spacing=spacing)
    return {"order": doc["order"], "line": line,
            "half_width": np.asarray(doc["half_width"]),
            "sides": np.asarray(doc.get("sides")
                                or [1.0] * len(doc["half_width"]), float),
            "blocks": doc.get("blocks", []),
            "coverage": doc.get("coverage")}

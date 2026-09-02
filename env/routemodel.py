"""Assemble the reference line the explorer actually drives, from layers.

    hand-drawn edits   >   roadtrace   >   learned-from-runs   >   provisional

`provisional` (straight spawn->gates->finish) is always available. `roadtrace`
follows the road ribbon and recovers the real checkpoint order when the track
is connected road. `learned` (env.learnedmap) is the fallback for a track
roadtrace cannot follow - jumps, platforms, open ground - and needs the
explorer to have driven it first. `edits` are polyline patches a human drew in
the panel, and they win wherever they are placed.

NOTE the order: roadtrace outranks learned, which is the opposite of what an
earlier version of this docstring claimed. It is deliberate. Roadtrace is
built from the map's own block geometry and is measured accurate to within a
few centimetres vertically and unbiased laterally; `learned` is the median of
a handful of laps by a policy that is still learning to steer, so it is only
better when there is no geometry to trace.

`merged_line()` is what `tm_env._load_map` calls in explore mode.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .centerline import Centerline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def edits_path(root: str, uid: str) -> str:
    return os.path.join(root, "maps", f"{uid}.route_edits.json")


def load_edits(root: str, uid: str) -> list[dict]:
    p = edits_path(root, uid)
    if not os.path.isfile(p):
        return []
    try:
        with open(p) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for pat in doc.get("patches", []):
        try:
            a = [float(pat["a"][0]), float(pat["a"][1])]
            b = [float(pat["b"][0]), float(pat["b"][1])]
            pts = [[float(x), float(z)] for x, z in pat["points"]]
        except (KeyError, TypeError, ValueError):
            continue
        if len(pts) < 1:
            continue
        out.append({"a": a, "b": b, "points": pts,
                    "kind": pat.get("kind", "road")})
    return out


def save_edits(root: str, uid: str, patches: list[dict]) -> None:
    os.makedirs(os.path.join(root, "maps"), exist_ok=True)
    with open(edits_path(root, uid), "w") as f:
        json.dump({"map": uid, "patches": patches}, f, indent=1)


def _nearest_idx(pts_xz: np.ndarray, p) -> int:
    d = pts_xz - np.asarray(p, float)
    return int(np.argmin(np.einsum("ij,ij->i", d, d)))


def _dekink(pts: np.ndarray, cos_thresh: float = -0.25, max_passes: int = 5):
    """Drop points where the line folds back on itself.

    Splicing several hand-drawn patches whose ends do not quite line up can
    leave a point whose incoming and outgoing segments point in near-opposite
    directions - the line goes forward, snaps ~20m back, then forward again.
    The policy's lookahead sits on that backward leg and it steers to follow
    it, which looks like the car deliberately turning round.

    A reversal spike is one point, or a short run of them, between two
    stretches heading the same way. Smooth curves never reverse and a real
    90-ish corner has cos ~ 0, so a threshold of -0.25 (past ~105 deg) only
    ever catches folds. Iterated, because removing one spike can expose the
    next. Returns (points, n_removed).
    """
    pts = np.asarray(pts, float)
    removed = 0
    for _ in range(max_passes):
        if len(pts) < 4:
            break
        keep = [0]
        i = 1
        dropped_this_pass = 0
        while i < len(pts) - 1:
            a = pts[i] - pts[keep[-1]]
            b = pts[i + 1] - pts[i]
            na = float(np.hypot(a[0], a[2]))
            nb = float(np.hypot(b[0], b[2]))
            if na > 1e-6 and nb > 1e-6 and \
                    (a[0] * b[0] + a[2] * b[2]) / (na * nb) < cos_thresh:
                dropped_this_pass += 1      # skip pts[i]
                i += 1
                continue
            keep.append(i)
            i += 1
        keep.append(len(pts) - 1)
        if not dropped_this_pass:
            break
        removed += dropped_this_pass
        pts = pts[keep]
    return pts, removed


_RIDE_M = 2.0          # car sits this far above the deck cell floor
_GROUND_EPS = 4.0      # world-y at or below this is the void / ground plane


def _snap_to_deck(pts: np.ndarray, grid, verbose: bool = False):
    """Pull each line point's Y onto the real road deck from the occupancy grid.

    Hand-drawn patches are 2D; their height is lerped straight between the two
    anchors, so a steep ramp comes out flattened and the line ends up buried in
    the hillside 15-40m below the road it is meant to follow. Here every point
    looks up the solid column at its own XZ and takes the deck nearest the
    height it already has - with continuity, so the line does not hop onto a
    bridge stacked overhead. Points over a genuine gap (no solid column) keep
    their interpolated Y and are re-interpolated from the snapped neighbours.
    """
    if grid is None or not len(grid):
        return pts, 0
    bx, by, bz = grid.block
    base_h = grid.base_height
    ys = pts[:, 1].copy()
    snapped = np.full(len(pts), np.nan)
    prev = None
    n_moved = 0
    for i, p in enumerate(pts):
        cx, _, cz = grid.world_to_cell(p)
        decks = []
        for cy in range(base_h - 1, base_h + 20):
            if grid.is_solid(int(cx), cy, int(cz)):
                wy = (cy - base_h) * by
                if wy > _GROUND_EPS:
                    decks.append(wy + _RIDE_M)
        if not decks:
            continue
        # Target the drawn (interpolated) height: it is a sane top-down guess
        # and is normally BELOW every real deck on a ramp, so "nearest deck"
        # picks the road surface rather than a bridge stacked above it. Only
        # fall back to continuity to kill a lone spike.
        best = min(decks, key=lambda d: abs(d - ys[i]))
        if prev is not None and abs(best - prev) > 30.0:
            near = [d for d in decks if abs(d - prev) <= 30.0]
            if near:
                best = min(near, key=lambda d: abs(d - ys[i]))
        snapped[i] = best
        prev = best
        if abs(best - ys[i]) > 0.5:
            n_moved += 1
    # fill gaps (no solid column) by interpolating between snapped neighbours
    have = np.where(~np.isnan(snapped))[0]
    if len(have) >= 2:
        snapped_filled = np.interp(np.arange(len(pts)), have, snapped[have])
        # leading / trailing runs with no anchor: keep the original Y
        snapped_filled[:have[0]] = ys[:have[0]]
        snapped_filled[have[-1] + 1:] = ys[have[-1] + 1:]
        out = pts.copy()
        out[:, 1] = snapped_filled
        # light smoothing so a cell-boundary step does not read as a bump
        k = np.array([0.25, 0.5, 0.25])
        out[1:-1, 1] = np.convolve(out[:, 1], k, mode="valid")
        return out, n_moved
    return pts, 0


def _apply_patches(pts: np.ndarray, patches: list[dict]):
    """pts: (N,3). Each patch replaces the span between its two anchors (matched
    to the nearest line points) with its own polyline; y is lerped from the two
    anchor points so a top-down drawing keeps a sane height. Returns (pts, jump
    spans as (start,end) indices in the NEW array)."""
    if not patches:
        return pts, []
    xz = pts[:, [0, 2]]
    # apply in order of anchor position so index maths stays valid
    spans = []
    for pat in patches:
        ia = _nearest_idx(xz, pat["a"])
        ib = _nearest_idx(xz, pat["b"])
        lo, hi = sorted((ia, ib))
        y0, y1 = pts[lo, 1], pts[hi, 1]
        pp = pat["points"]
        if (ia > ib):                        # patch runs against the line
            pp = pp[::-1]
        m = len(pp)
        seg = np.empty((m, 3))
        for k, (x, z) in enumerate(pp):
            t = k / max(m - 1, 1)
            seg[k] = (x, y0 + (y1 - y0) * t, z)
        new = np.vstack([pts[:lo + 1], seg, pts[hi:]])
        if pat["kind"] == "jump":
            spans.append((lo + 1, lo + 1 + m))
        pts = new
        xz = pts[:, [0, 2]]
    return pts, spans


def merged_line(root: str, uid: str, gates=None, dump=None,
                spacing: float = 2.0, use_learned: bool = True,
                verbose: bool = True):
    """Returns {source, line, half_width, order, jumps, layers} or None.

    `gates` is an env.mapdata.Gates (for provisional / order). `dump` is the raw
    ``dumpmap occupancy`` dict for roadtrace. Either may be None."""
    layers = []
    base = None
    order = None
    hw = None
    sides = None

    # -- roadtrace: best-effort estimate of the whole route -----------
    if (dump and dump.get("boxes") and gates is not None
            and gates.checkpoints and gates.finish):
        try:
            from .roadtrace import build_road_trace
            fin = np.mean(np.asarray(gates.finish[0]), axis=0)
            rt = build_road_trace(
                dump["boxes"], dump.get("names", []),
                dump.get("base_height", 0), dump.get("block_size", (32, 8, 32)),
                gates.spawn, [np.asarray(c) for c in gates.centres], fin,
                verbose=verbose)
        except Exception as ex:                       # noqa: BLE001
            rt = None
            if verbose:
                print(f"  routemodel: roadtrace failed ({ex})")
        if rt is not None:
            base = rt["line"].points.copy()
            order = rt["order"]
            hw = rt["half_width"]
            sides = rt.get("sides")
            layers.append(f"roadtrace({rt.get('coverage', 1) * 100:.0f}%)")
            # refresh the plain-JSON cache the stdlib panel reads
            try:
                from .roadtrace import save as _rt_save
                _rt_save(root, uid, rt)
            except Exception:                             # noqa: BLE001
                pass

    # -- learned: the fallback when there is no traceable road ribbon --
    if base is None and use_learned:
        try:
            from .learnedmap import load as load_learned
            lm = load_learned(root, uid, spacing=spacing)
        except Exception:                             # noqa: BLE001
            lm = None
        if lm is not None:
            base = lm["line"].points.copy()
            order = lm["order"]
            hw = lm["half_width"]
            sides = lm.get("sides")
            layers.append("learned")

    # -- provisional (fallback) --------------------------------------
    if base is None:
        if gates is None:
            return None
        from .mapdata import provisional_line
        pl = provisional_line(gates, spacing=spacing)
        base = pl.points.copy()
        order = list(range(len(gates.centres))) if gates.centres else []
        hw = np.full(len(base), 8.0)
        # No geometry to judge from: assume barriers. Wrong in the safe
        # direction - a car told there IS a wall drives more carefully
        # than one told there is not.
        sides = np.ones(len(base))
        layers.append("provisional")

    # -- hand edits (always win where placed) -----------------------
    patches = load_edits(root, uid)
    jumps_world = []
    if patches:
        base, jspans = _apply_patches(base, patches)
        # hw needs to match the new length: nearest-carry
        old_hw = hw
        hw = np.interp(np.linspace(0, 1, len(base)),
                       np.linspace(0, 1, len(old_hw)), old_hw)
        if sides is not None:
            old_sides = sides
            sides = np.interp(np.linspace(0, 1, len(base)),
                              np.linspace(0, 1, len(old_sides)), old_sides)
        for lo, hi in jspans:
            lo = max(0, min(lo, len(base) - 1))
            hi = max(0, min(hi, len(base)))
            if hi > lo:
                jumps_world.append([float(base[lo, 0]), float(base[lo, 2]),
                                    float(base[hi - 1, 0]), float(base[hi - 1, 2])])
        layers.append(f"edits({len(patches)})")

    base, kinks = _dekink(base)
    if kinks and verbose:
        print(f"  routemodel: removed {kinks} kink(s) (line folded back on "
              f"itself - patch splice artefact)")

    # Ride the real road deck: 2D patches lerp height between anchors, which
    # buries the line in a hillside wherever the road ramps up steeply.
    if patches:
        try:
            from .lidar import OccupancyGrid
            _grid = OccupancyGrid.load(os.path.join(root, "maps", f"{uid}.json"))
            base, moved = _snap_to_deck(base, _grid, verbose=verbose)
            if moved and verbose:
                print(f"  routemodel: snapped {moved} point(s) onto the "
                      f"occupancy deck (patch height was off the road)")
        except Exception as ex:                            # noqa: BLE001
            if verbose:
                print(f"  routemodel: deck-snap skipped ({ex})")

    line = Centerline(base, spacing=spacing)
    hw = np.interp(line.s / max(line.length, 1e-6),
                   np.linspace(0, 1, len(hw)), hw)
    if sides is None:
        sides = np.ones(len(line.points))
    else:
        sides = np.interp(line.s / max(line.length, 1e-6),
                          np.linspace(0, 1, len(sides)), sides)
    src = "+".join(layers)
    if verbose:
        print(f"  routemodel: {src}  {line.length:.0f} m  "
              f"{len(line.points)} pts  order {order}")
    return {"source": src, "line": line, "half_width": hw, "sides": sides,
            "order": order, "jumps": jumps_world, "layers": layers}

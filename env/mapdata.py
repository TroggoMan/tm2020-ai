"""Static map knowledge: where the gates are and where the effects are.

Both come from the plugin - `landmarks` and `dumpmap` - and both are constant
for a map, so everything here is computed once when the map loads and then only
queried.

Two things this fixes that the first cut got wrong:

  * A gate is not one landmark. Querying a real map showed a finish line made
    of six separate landmarks 32m apart, all tagged "Goal" with order 0 - one
    per block of the finish's width. Counting them individually would have
    counted a single wide checkpoint six times. Gates are therefore clustered
    by position, and the whole cluster counts once.

  * Landmark `Order` is not a reliable sequence. Mappers only set it when they
    link checkpoints, so it is very often 0 for every gate on the map. When a
    reference line is available the gates are ordered by arc length along it
    instead, which is exact.
"""
from __future__ import annotations

import numpy as np

from .centerline import Centerline
from .surfaces import EFFECT_CLASSES, cell_to_world, effect_class

# Two landmarks closer than this are the same gate. A block is 32m, so this is
# "adjacent or overlapping blocks" with a little slack, and it is well under
# the spacing between distinct checkpoints on any sane track.
GATE_CLUSTER_M = 48.0

# What "no effect of this class anywhere ahead" reports as. Far enough that the
# normalised value saturates, so the policy reads it as "nothing coming".
FAR = 400.0


def _cluster(points: list[np.ndarray], radius: float) -> list[list[np.ndarray]]:
    """Single-linkage clustering. N is a handful of landmarks, so the naive
    O(n^2) walk is not worth improving."""
    clusters: list[list[np.ndarray]] = []
    for p in points:
        joined = None
        for c in clusters:
            if any(np.linalg.norm(p - q) <= radius for q in c):
                if joined is None:
                    c.append(p)
                    joined = c
                else:
                    # p bridges two clusters; merge them.
                    joined.extend(c)
                    c.clear()
        if joined is None:
            clusters.append([p])
    return [c for c in clusters if c]


class Gates:
    """Checkpoint and finish gates for one map."""

    def __init__(self, items: list[dict], line: Centerline | None = None,
                 cluster_m: float = GATE_CLUSTER_M):
        cps = [np.asarray(i["pos"], dtype=np.float64)
               for i in items if i.get("kind") == "checkpoint"]
        fins = [np.asarray(i["pos"], dtype=np.float64)
                for i in items if i.get("kind") == "finish"]
        self.spawn = next((np.asarray(i["pos"], dtype=np.float64)
                           for i in items if i.get("kind") == "spawn"), None)

        self.checkpoints = _cluster(cps, cluster_m)
        self.finish = _cluster(fins, cluster_m)

        # Order the gates the way the car will meet them.
        #
        # This is not optional bookkeeping: the checkpoint counter and the
        # provisional line both walk the list in order, and the game reports
        # landmarks in an arbitrary one. A real map came back with its far
        # checkpoint listed before its near one, which would have run stage
        # one's line backwards through the first turn.
        #
        # `Order` is not a usable key - mappers leave it at 0 unless they have
        # explicitly linked checkpoints, and it was 0 on both there.
        if line is not None and self.checkpoints:
            # Exact: arc length along a line that is known to follow the track.
            self.checkpoints.sort(
                key=lambda g: line.project(np.mean(np.asarray(g), axis=0))[1])
            self.order_source = "reference line"
        elif len(self.checkpoints) > 1:
            self.checkpoints = self._chain_from_spawn()
            self.order_source = "nearest-neighbour from spawn (heuristic)"
        else:
            self.order_source = "trivial"

        self.centres = [np.mean(np.asarray(g), axis=0) for g in self.checkpoints]
        self.normals = self._gate_normals()
        # Per-gate size, filled in by set_sizes(). None means "use whatever
        # the caller passes", which is the old global behaviour.
        self.widths: list[float | None] = [None] * len(self.checkpoints)
        self.heights: list[float | None] = [None] * len(self.checkpoints)

    # -- gate size --------------------------------------------------------

    def spans(self) -> list[dict]:
        """How big each gate physically is, from its own landmarks.

        Checkpoints are not all one size - a 1-block gate and a five-lane gate
        are both "a checkpoint" - and the game gives us no dimensions at all:
        `CGameScriptMapLandmark` carries a position, a tag and an order, and
        `CGameScriptMapWaypoint` carries two booleans. Nothing else.

        What we do get is one landmark per block of the gate, clustered
        together, so the cluster's own extent measures the gate. `lateral` is
        the spread across the gate's plane and `vertical` the spread up it;
        both are 0 for a single-block gate, which is correct - a 1x1 gate has
        no extent, and its size is entirely the margin around it.
        """
        out = []
        for i, cluster in enumerate(self.checkpoints):
            pts = np.asarray(cluster, dtype=np.float64)
            normal = self.normals[i] if i < len(self.normals) else None
            if normal is None or len(pts) < 2:
                out.append({"blocks": len(pts), "lateral": 0.0,
                            "vertical": float(np.ptp(pts[:, 1])) if len(pts) else 0.0})
                continue
            rel = pts - self.centres[i]
            flat = rel.copy()
            flat[:, 1] = 0.0
            along = flat - np.outer(flat @ normal, normal)
            lateral = float(np.linalg.norm(along, axis=1).max() * 2.0)
            out.append({"blocks": len(pts), "lateral": lateral,
                        "vertical": float(np.ptp(rel[:, 1]))})
        return out

    def set_sizes(self, overrides: dict | None = None) -> None:
        """Per-gate half-width and height, from an index -> dict mapping.

        Only what you name is overridden; everything else keeps falling back
        to the global setting. Indices are the *ordered* gate indices - the
        same numbers the checkpoint counter and the split times use - so
        "checkpoint 2 is a wide one" is one line.
        """
        self.widths = [None] * len(self.checkpoints)
        self.heights = [None] * len(self.checkpoints)
        for key, val in (overrides or {}).items():
            try:
                i = int(key)
            except (TypeError, ValueError):
                continue
            if not (0 <= i < len(self.checkpoints)) or not isinstance(val, dict):
                continue
            if val.get("half_width") is not None:
                self.widths[i] = float(val["half_width"])
            if val.get("height") is not None:
                self.heights[i] = float(val["height"])

    def size_of(self, index: int, half_width: float,
                height: float) -> tuple[float, float]:
        w = self.widths[index] if index < len(self.widths) else None
        h = self.heights[index] if index < len(self.heights) else None
        return (half_width if w is None else w, height if h is None else h)

    # -- gate planes ------------------------------------------------------

    def _gate_normals(self) -> list[np.ndarray]:
        """A facing direction per gate, so "went through it" is answerable.

        The landmark gives us a position and nothing else - no rotation - so
        the plane the car has to cross has to be inferred. The route supplies
        it: a checkpoint faces along the track, and the track at that
        checkpoint runs from whatever comes before it to whatever comes after.
        Horizontal only, because a gate is vertical however steep the road is.

        With one gate and no spawn there is nothing to infer from and the
        normal comes back as None, which the crossing test reads as "fall back
        to proximity" rather than as "never counts".
        """
        pts: list[np.ndarray | None] = [self.spawn]
        pts += list(self.centres)
        pts.append(np.mean(np.asarray(self.finish[0]), axis=0)
                   if self.finish else None)

        out = []
        for i in range(len(self.centres)):
            before, after = pts[i], pts[i + 2]
            here = self.centres[i]
            if before is not None and after is not None:
                v = np.asarray(after) - np.asarray(before)
            elif after is not None:
                v = np.asarray(after) - here
            elif before is not None:
                v = here - np.asarray(before)
            else:
                out.append(None)
                continue
            v = np.asarray([v[0], 0.0, v[2]], dtype=np.float64)
            n = np.linalg.norm(v)
            out.append(v / n if n > 1e-6 else None)
        return out

    # NB: deriving the gate planes from the reference line's tangent instead of
    # the checkpoint-to-checkpoint chord was tried and is WORSE, despite the
    # chord being 22-40 degrees off the road direction at every gate here.
    # Replayed against 207 real traces: the chord reproduces the game's own
    # checkpoint count exactly (207/207, no false positives, zero steps of
    # lag); the tangent gave 206/207, one false positive and up to 8 steps of
    # lag, because the game triggers a checkpoint on entering the block's
    # volume, slightly before the plane through the landmark. Do not "fix" it.

    def crossed(self, prev: np.ndarray, pos: np.ndarray, index: int,
                half_width: float, height: float) -> bool:
        """Did the car pass THROUGH gate `index` between these two samples?

        Three conditions, all of which the game itself imposes and proximity
        imposes none of:

          * the step crosses the gate's plane - the signed distance along the
            gate normal changes sign, in either direction, because TM credits
            a checkpoint however you go through it;
          * at the crossing point the car is inside the gate's width, measured
            sideways from the gate line rather than from its middle, so a wide
            multi-block gate counts at its edge;
          * and within `height` vertically, so a road stacked over the
            checkpoint does not collect it from above.

        The crossing point is interpolated rather than tested at either
        endpoint: at 40Hz and 400km/h the car moves ~2.8m per step, and a gate
        is a plane with no thickness.
        """
        if index >= len(self.checkpoints):
            return False
        half_width, height = self.size_of(index, half_width, height)
        normal = self.normals[index] if index < len(self.normals) else None
        if normal is None:
            return self.hit(pos, index, half_width)

        centre = self.centres[index]
        d0 = float(np.dot(np.asarray(prev) - centre, normal))
        d1 = float(np.dot(np.asarray(pos) - centre, normal))
        if d0 == d1 or (d0 > 0) == (d1 > 0):
            return False

        t = d0 / (d0 - d1)
        hit = np.asarray(prev) + (np.asarray(pos) - np.asarray(prev)) * t

        # Distance from the nearest landmark in the cluster, in the plane. The
        # cluster is one landmark per block of the gate's width, so measuring
        # from the nearest one is measuring from the gate itself.
        for q in self.checkpoints[index]:
            v = hit - q
            if abs(v[1]) > height:
                continue
            flat = np.asarray([v[0], 0.0, v[2]])
            lateral = flat - np.dot(flat, normal) * normal
            if float(np.linalg.norm(lateral)) <= half_width:
                return True
        return False

    def crossed_any(self, prev: np.ndarray, pos: np.ndarray, taken: set[int],
                    half_width: float, height: float) -> int | None:
        """Index of an untaken gate this step went through, in any order."""
        for i in range(len(self.checkpoints)):
            if i in taken:
                continue
            if self.crossed(prev, pos, i, half_width, height):
                return i
        return None

    def _chain_from_spawn(self) -> list[list[np.ndarray]]:
        """Greedy nearest-neighbour walk: start at the spawn, repeatedly hop to
        the closest gate not yet visited.

        A heuristic, and it says so. It is right for the ordinary case of a
        track that does not double back on itself, and wrong is recoverable -
        stage one simply fails to make progress, and the order it chose is
        printed so you can override it. Once any lap is recorded the reference
        line replaces this with the exact answer.
        """
        start = self.spawn
        if start is None:
            start = np.mean(np.asarray(self.checkpoints[0]), axis=0)
        # Track indices, not the gates themselves: a gate is a list of numpy
        # arrays, so list.remove() would compare them element-wise and raise.
        remaining = list(range(len(self.checkpoints)))
        centres = [np.mean(np.asarray(g), axis=0) for g in self.checkpoints]
        out, here = [], np.asarray(start, dtype=np.float64)
        while remaining:
            i = min(remaining, key=lambda k: np.linalg.norm(centres[k] - here))
            remaining.remove(i)
            out.append(self.checkpoints[i])
            here = centres[i]
        return out

    def reorder(self, indices) -> None:
        """Override the guessed order, e.g. from --gate-order on the trainer."""
        if sorted(indices) != list(range(len(self.checkpoints))):
            raise ValueError(
                f"gate order must be a permutation of 0..{len(self.checkpoints)-1}, "
                f"got {list(indices)}")
        self.checkpoints = [self.checkpoints[i] for i in indices]
        self.centres = [np.mean(np.asarray(g), axis=0) for g in self.checkpoints]
        # The normals are derived FROM the order, so a reorder invalidates
        # them; recompute or every gate faces the way the old route ran.
        self.normals = self._gate_normals()
        self.order_source = "manual override"

    def __len__(self) -> int:
        return len(self.checkpoints)

    def hit(self, pos: np.ndarray, index: int, radius: float) -> bool:
        """Is the car at gate `index`? Any landmark in the cluster counts, so a
        wide gate is as easy to trigger at its edge as at its middle."""
        if index >= len(self.checkpoints):
            return False
        return any(np.linalg.norm(pos - q) < radius
                   for q in self.checkpoints[index])

    def hit_any(self, pos: np.ndarray, taken: set[int], radius: float) -> int | None:
        """Index of an as-yet-untaken gate the car is at, in any order.

        TM2020 does not enforce checkpoint order unless the mapper has
        explicitly linked them - you must collect them all before the finish
        counts, but the sequence is yours. Counting strictly in sequence would
        therefore refuse to credit a checkpoint the game itself had credited,
        and a policy that found a faster ordering would be punished for it.

        The ordering in `self.checkpoints` stays useful as *guidance* - it is
        what the provisional line follows - but it is not a rule.
        """
        for i, gate in enumerate(self.checkpoints):
            if i in taken:
                continue
            if any(np.linalg.norm(pos - q) < radius for q in gate):
                return i
        return None

    def at_finish(self, pos: np.ndarray, radius: float) -> bool:
        return any(np.linalg.norm(pos - q) < radius
                   for g in self.finish for q in g)

    def describe(self) -> str:
        n = len(self.checkpoints)
        raw = sum(len(g) for g in self.checkpoints)
        extra = f" (from {raw} landmarks)" if raw != n else ""
        order = f", order by {self.order_source}" if n > 1 else ""
        return (f"{n} checkpoint gate{'s' if n != 1 else ''}{extra}"
                f", {len(self.finish)} finish{order}")

    def route(self) -> str:
        """The chosen order, spelled out. Worth printing whenever the order is
        a guess - it is the one thing that silently wrecks stage one."""
        pts = []
        if self.spawn is not None:
            pts.append(("spawn", self.spawn))
        pts += [(f"cp{i+1}", c) for i, c in enumerate(self.centres)]
        if self.finish:
            pts.append(("finish", np.mean(np.asarray(self.finish[0]), axis=0)))
        legs = []
        for (na, a), (nb, b) in zip(pts, pts[1:]):
            legs.append(f"{na} -> {nb} {np.linalg.norm(np.asarray(b)-np.asarray(a)):.0f}m")
        return "  ".join(legs)


def provisional_line(gates: "Gates", spacing: float = 2.0) -> Centerline:
    """A straight-line path through spawn, the checkpoints and the finish.

    This is the whole trick behind attacking a map nobody has driven. The
    positions are static map data, so the route *order* is known before a wheel
    turns - what is unknown is where the road physically goes between them.

    Feeding those points through the normal `Centerline` means stage one needs
    no new environment: progress along this line is progress toward the finish,
    and every lookahead, projection and reward term already written keeps
    working. It just happens to be a terrible racing line - it cuts through
    scenery and ignores every corner - so the off-line limit has to be opened
    right up and the actual road-finding left to the lidar.

    Stage two then replaces it with the best real trajectory, which is a
    Centerline of exactly the same shape, so nothing downstream changes.
    """
    pts: list[np.ndarray] = []
    if gates.spawn is not None:
        pts.append(np.asarray(gates.spawn, dtype=np.float64))
    pts.extend(np.asarray(c, dtype=np.float64) for c in gates.centres)
    if gates.finish:
        pts.append(np.mean(np.asarray(gates.finish[0]), axis=0))
    if len(pts) < 2:
        raise ValueError(
            "need at least a spawn and a finish to build a provisional line; "
            "the map reported neither")

    # Interpolate along each leg so the resampler has something to work with -
    # two points 1500m apart give a line with no intermediate samples, and
    # every lookahead query would return the same far-away point.
    dense: list[np.ndarray] = []
    for a, b in zip(pts, pts[1:]):
        n = max(int(np.linalg.norm(b - a) / spacing), 1)
        for k in range(n):
            dense.append(a + (b - a) * (k / n))
    dense.append(pts[-1])
    return Centerline(np.asarray(dense), spacing=spacing)


class EffectMap:
    """Where the boosters, no-grip patches and bumpers are, along the line.

    Positions are projected onto the reference line once, so the per-step query
    is "what is the next boost after arc length s" - a binary search, not a
    distance check against every effect on the map.
    """

    def __init__(self, dump: dict, line: Centerline):
        self.base_height = int(dump.get("base_height", 8))
        self.counts: dict[str, int] = {c: 0 for c in EFFECT_CLASSES}
        self.raw_kinds: dict[str, int] = {}
        # class -> (arc lengths ascending, lateral offset at that point)
        self._s: dict[str, np.ndarray] = {}
        self._off: dict[str, np.ndarray] = {}

        buckets: dict[str, list[tuple[float, float]]] = {
            c: [] for c in EFFECT_CLASSES}
        for e in dump.get("effects", []):
            kind = e.get("type", "")
            self.raw_kinds[kind] = self.raw_kinds.get(kind, 0) + 1
            cls = effect_class(kind)
            if cls is None:
                continue
            for pos in self._positions(e):
                _, s, off = line.project(np.asarray(pos, dtype=np.float64))
                # An effect 60m off the line is on some other part of the track
                # that happens to run alongside; warning about it would be a
                # lie. Blocks are 32m, so one block of slack.
                if off > 40.0:
                    continue
                buckets[cls].append((s, off))
                self.counts[cls] += 1

        for cls, vals in buckets.items():
            vals.sort()
            self._s[cls] = np.asarray([v[0] for v in vals], dtype=np.float64)
            self._off[cls] = np.asarray([v[1] for v in vals], dtype=np.float64)

    def _positions(self, e: dict):
        if e.get("kind") == "item":
            yield e["pos"]
            return
        for cell in e.get("cells", []):
            yield cell_to_world(cell, self.base_height)

    def ahead(self, s: float, cls: str) -> tuple[float, float]:
        """Distance and lateral offset of the next effect of this class.

        Returns (FAR, 0.0) when there is nothing left ahead, which is also what
        a map with no effects at all reports for every query.
        """
        arr = self._s.get(cls)
        if arr is None or arr.size == 0:
            return FAR, 0.0
        i = int(np.searchsorted(arr, s, side="left"))
        if i >= arr.size:
            return FAR, 0.0
        return min(float(arr[i] - s), FAR), float(self._off[cls][i])

    def describe(self) -> str:
        used = ", ".join(f"{c}={self.counts[c]}" for c in EFFECT_CLASSES
                         if self.counts[c])
        return used or "no effects on the line"

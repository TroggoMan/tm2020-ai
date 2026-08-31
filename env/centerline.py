"""A reference line through a track, and the geometry queries the policy needs.

There is no raycast/LIDAR API in TM2020 (see README), and the offline Gbx block
geometry isn't parsed yet. So the interim spatial representation is a *reference
line*: drive the track once, keep the positions, resample them to even spacing,
and describe the car's situation relative to that line.

That gives the three things a racing policy actually needs - how far along am I,
how far off the line am I, and where does the track go next - without any new
plugin work. When the Gbx geometry lands, real wall distances get added
alongside this rather than replacing it.
"""
from __future__ import annotations

import json

import numpy as np


class Centerline:
    def __init__(self, points: np.ndarray, spacing: float = 2.0):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"expected an (N,3) array of positions, got {pts.shape}")
        self.map_uid: str | None = None
        self.points = self._resample(pts, spacing)
        # Arc length at each sample, and the unit tangent along the line.
        deltas = np.diff(self.points, axis=0)
        seg = np.linalg.norm(deltas, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        tangents = np.zeros_like(self.points)
        tangents[:-1] = deltas
        tangents[-1] = deltas[-1]
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        self.tangents = tangents / np.maximum(norms, 1e-9)

    @staticmethod
    def _resample(pts: np.ndarray, spacing: float) -> np.ndarray:
        """Even arc-length spacing, so lookahead distances mean the same thing
        everywhere. A raw recording is dense in slow corners and sparse on fast
        straights, which would otherwise skew every lookahead query."""
        deltas = np.diff(pts, axis=0)
        seg = np.linalg.norm(deltas, axis=1)
        keep = seg > 1e-6
        pts = np.concatenate([pts[:1], pts[1:][keep]])
        seg = seg[keep]
        if len(pts) < 2:
            raise ValueError("reference line needs at least two distinct points")
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        n = max(int(total / spacing) + 1, 2)
        target = np.linspace(0.0, total, n)
        out = np.empty((n, 3))
        for axis in range(3):
            out[:, axis] = np.interp(target, s, pts[:, axis])
        return out

    def project(self, pos: np.ndarray) -> tuple[int, float, float]:
        """Nearest sample to `pos`. Returns (index, arc length, lateral offset).

        Lateral offset is unsigned distance to the line - which side we're on
        doesn't matter for the reward, and a sign would need a consistent
        surface normal we don't have.
        """
        d = self.points - pos
        i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        return i, float(self.s[i]), float(np.linalg.norm(d[i]))

    def project_near(self, pos: np.ndarray, last_index: int | None,
                     moved_m: float | None = None,
                     window_m: float = 12.0, lost_m: float = 40.0):
        """Like `project`, but searching only near where we were last step.

        A real road folds back on itself - hairpins, road stacked over road -
        and a global nearest-point search jumps between the branches whenever
        the car passes near a fold. Measured on this track that was a **32 m
        jump in arc length for 0.8 m of movement**, at five places on the lap:
        a free chunk of progress reward, and an 18-dimensional lookahead that
        silently teleports to a different part of the circuit at exactly the
        corners where the policy most needs it to be right.

        The car cannot move further than a few metres per control step, so the
        honest answer is the nearest point *near the last one*. Pass `moved_m`
        - how far the car actually travelled since the last call - and the
        window follows it, which is what finally closes the gap: a fixed 12 m
        window still let an 11.9 m arc jump through, and 11.9 m in a 50 ms step
        is 857 km/h. Three times the distance travelled leaves room for the
        line being longer than the chord through a corner, and nothing like
        enough room to reach the far side of a hairpin.

        Falls back to a global search when there is no previous index, or when
        the local answer is more than `lost_m` away - which is the car having
        genuinely left the line (a respawn, a big off) rather than a fold.
        """
        if last_index is None:
            return self.project(pos)
        if moved_m is not None:
            window_m = min(window_m, max(3.0, 3.0 * float(moved_m)))
        lo = int(np.searchsorted(self.s, self.s[last_index] - window_m))
        hi = int(np.searchsorted(self.s, self.s[last_index] + window_m))
        hi = max(hi, lo + 1)
        d = self.points[lo:hi] - pos
        j = int(np.argmin(np.einsum("ij,ij->i", d, d))) + lo
        off = float(np.linalg.norm(self.points[j] - pos))
        if off > lost_m:
            return self.project(pos)
        return j, float(self.s[j]), off

    def lookahead(self, index: int, distances) -> np.ndarray:
        """World-space points on the line at the given distances ahead."""
        s_target = self.s[index] + np.asarray(distances, dtype=np.float64)
        out = np.empty((len(s_target), 3))
        for axis in range(3):
            out[:, axis] = np.interp(s_target, self.s, self.points[:, axis])
        return out

    def save(self, path: str, map_uid: str | None = None) -> None:
        """The map uid is recorded so a line can never be silently used on the
        wrong track. Doing that produced 2140 one-step episodes before anyone
        noticed, because a car hundreds of metres from the line looks exactly
        like a car that cannot drive."""
        with open(path, "w") as f:
            json.dump({"spacing_resampled": True,
                       "map": map_uid or self.map_uid,
                       "points": self.points.tolist()}, f)

    @classmethod
    def load(cls, path: str, spacing: float = 2.0) -> "Centerline":
        with open(path) as f:
            data = json.load(f)
        line = cls(np.asarray(data["points"]), spacing=spacing)
        line.map_uid = data.get("map")
        return line

    @staticmethod
    def peek_map(path: str) -> str | None:
        """Which map a line file belongs to, without parsing its points.
        Returns None for lines recorded before the uid was stored."""
        try:
            with open(path) as f:
                return json.load(f).get("map")
        except (OSError, json.JSONDecodeError):
            return None


def car_frame(vec: np.ndarray, dir_: np.ndarray, up: np.ndarray,
              left: np.ndarray) -> np.ndarray:
    """World vector -> car-local (forward, left, up).

    The policy must not depend on absolute world position or heading, or it
    learns the one track's coordinates instead of how to drive. Everything
    spatial goes through here first.
    """
    return np.array([float(np.dot(vec, dir_)),
                     float(np.dot(vec, left)),
                     float(np.dot(vec, up))])

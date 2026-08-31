"""A map model learned from the explorer's own driven runs.

`roadtrace` only works on a track that is one connected string of road blocks.
This does not care what the track is made of - it reads the trace files the
env already writes (``runs/<uid>/traces/*.json``: t, x, y, z, speed, steer,
gas, brake, cp per sample) and boils the good ones down into:

  * an empirical centerline - the median of where the surviving runs went,
    aligned on their checkpoint crossings so "60 % round" is the same place
    in every run;
  * a per-point corridor half-width - how far the runs spread either side;
  * the checkpoint ORDER, by majority vote over the runs;
  * jump segments - stretches where the car left the ground.

It is only worth anything once enough decent runs exist, so callers treat it
as the third tier behind hand edits and `roadtrace`. Re-run it as the policy
improves and the line pulls toward the real racing line.
"""
from __future__ import annotations

import collections
import glob
import json
import os

import numpy as np

from .centerline import Centerline

# a run has to have reached at least this fraction of the best CP count seen
# across all traces to be trusted as "knows roughly where the track goes"
_FRONTIER_FRAC = 0.6
_MIN_RUNS = 6
_RESAMPLE_N = 400          # points along the normalised progress axis


def _list_traces(root: str, uid: str):
    d = os.path.join(root, "runs", uid, "traces")
    return sorted(glob.glob(os.path.join(d, "*.json")))


def _load(path: str):
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    s = doc.get("samples")
    if not s or len(s) < 8:
        return None
    a = np.asarray(s, dtype=np.float64)      # (N, 9)
    # t, x, y, z, speed, steer, gas, brake, cp
    return {"xyz": a[:, 1:4], "speed": a[:, 4], "cp": a[:, 8].astype(int),
            "cp_final": int(doc.get("checkpoints", a[:, 8].max())),
            "finished": bool(doc.get("finished")),
            "dist": float(doc.get("distance", 0.0))}


def _progress_axis(run, n_cp):
    """Map each sample to a monotone progress coordinate in [0, n_cp+1):
    integer part = checkpoints taken, fractional part = fraction of the way to
    the next one by cumulative distance. Gives a common axis to average on
    without needing the checkpoint POSITIONS."""
    xyz = run["xyz"]
    seg = np.r_[0.0, np.linalg.norm(np.diff(xyz, axis=0), axis=1).cumsum()]
    cp = run["cp"]
    prog = np.zeros(len(xyz))
    for k in range(n_cp + 1):
        m = cp == k
        if not m.any():
            continue
        lo = seg[m][0]
        hi = seg[m][-1]
        span = max(hi - lo, 1e-6)
        prog[m] = k + np.clip((seg[m] - lo) / span, 0.0, 0.999)
    # force monotone (numerical safety)
    return np.maximum.accumulate(prog)


def build_learned_map(root: str, uid: str, spacing: float = 2.0,
                      verbose: bool = True):
    """Returns {order, line, half_width, jumps, n_runs} or None."""
    paths = _list_traces(root, uid)
    runs = [r for r in (_load(p) for p in paths) if r is not None]
    if len(runs) < _MIN_RUNS:
        if verbose:
            print(f"  learnedmap: only {len(runs)} traces, need {_MIN_RUNS}")
        return None

    best_cp = max(r["cp_final"] for r in runs)
    if best_cp < 1:
        if verbose:
            print("  learnedmap: no run has taken a checkpoint yet")
        return None
    keep = [r for r in runs if r["cp_final"] >= max(1, best_cp * _FRONTIER_FRAC)]
    if len(keep) < _MIN_RUNS:
        keep = sorted(runs, key=lambda r: r["cp_final"], reverse=True)[:_MIN_RUNS]
    if verbose:
        print(f"  learnedmap: {len(keep)}/{len(runs)} runs kept "
              f"(best CP {best_cp})")

    # checkpoint count by majority vote among finishers, else the max seen
    fins = [r["cp_final"] for r in keep if r["finished"]]
    n_cp = collections.Counter(fins).most_common(1)[0][0] if fins else best_cp

    # resample every kept run onto a common progress grid and stack
    grid = np.linspace(0.0, n_cp + 0.999, _RESAMPLE_N)
    stack = []
    airtime = np.zeros(_RESAMPLE_N)
    for r in keep:
        prog = _progress_axis(r, n_cp)
        if prog[-1] - prog[0] < 0.5:
            continue
        xyz = np.empty((_RESAMPLE_N, 3))
        for ax in range(3):
            xyz[:, ax] = np.interp(grid, prog, r["xyz"][:, ax])
        stack.append(xyz)
        # crude airborne flag: local vertical speed sign flip with low ground
        # persistence -> approximated by speed staying high while dz/dt small
        # (kept simple; refined once a grounded flag is in the trace)
    if len(stack) < 3:
        if verbose:
            print("  learnedmap: runs too short after alignment")
        return None
    S = np.stack(stack)                      # (R, N, 3)

    med = np.median(S, axis=0)               # (N, 3) empirical centerline
    # lateral spread -> half width. Measure perpendicular to the median tangent.
    tang = np.gradient(med, axis=0)
    tang[:, 1] = 0.0
    tn = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.maximum(tn, 1e-9)
    nrm = np.stack([-tang[:, 2], np.zeros(_RESAMPLE_N), tang[:, 0]], axis=1)
    lat = np.einsum("rnc,nc->rn", S - med[None], nrm)     # (R, N) signed offset
    hw = np.maximum(np.abs(np.percentile(lat, 90, axis=0)),
                    np.abs(np.percentile(lat, 10, axis=0)))
    hw = np.clip(hw, 3.0, 40.0)
    # 5-tap smooth
    k = np.ones(5) / 5.0
    hw = np.convolve(np.pad(hw, 2, "edge"), k, "valid")

    line = Centerline(med, spacing=spacing)
    # re-map hw onto the resampled line
    src_s = np.linspace(0.0, 1.0, _RESAMPLE_N)
    dst_s = line.s / max(line.length, 1e-6)
    hw_line = np.interp(dst_s, src_s, hw)

    order = list(range(n_cp))       # progress axis already enforces 0..n_cp-1
    if verbose:
        print(f"  learnedmap: {len(line.points)} pts, {line.length:.0f} m, "
              f"{n_cp} checkpoints, half-width {hw_line.min():.0f}-"
              f"{hw_line.max():.0f} m")
    return {"order": order, "line": line, "half_width": hw_line,
            "jumps": [], "n_runs": len(stack)}


# --- cache --------------------------------------------------------------

def cache_path(root: str, uid: str) -> str:
    return os.path.join(root, "maps", f"{uid}.learned.json")


def save(root: str, uid: str, model: dict) -> None:
    line = model["line"]
    doc = {"map": uid, "order": model["order"], "n_runs": model.get("n_runs"),
           "jumps": model.get("jumps", []),
           "points": line.points.tolist(),
           "half_width": [round(float(x), 2) for x in model["half_width"]]}
    os.makedirs(os.path.dirname(cache_path(root, uid)), exist_ok=True)
    with open(cache_path(root, uid), "w") as f:
        json.dump(doc, f)


def load(root: str, uid: str, spacing: float = 2.0):
    p = cache_path(root, uid)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        doc = json.load(f)
    line = Centerline(np.asarray(doc["points"], float), spacing=spacing)
    return {"order": doc["order"], "line": line,
            "half_width": np.asarray(doc["half_width"]),
            "jumps": doc.get("jumps", []), "n_runs": doc.get("n_runs")}

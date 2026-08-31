#!/usr/bin/env python3
"""Section times: where the lap is actually being lost.

A whole-lap time tells you the run was slow. It does not tell you *which part*
was slow, and on a track with four checkpoints there are five sections that
each need working on separately. This reads `runs/<map>/splits.jsonl` - one
line per episode, written by the environment - and reports, per section:

    best        the fastest that section has ever been driven
    median      what it usually is
    gap         median - best, i.e. how much is sitting on the table there
    reached     how often the car even got that far

The section with the biggest `gap` is the one to point a hint at. The section
with a low `reached` is not a slow section, it is where the car keeps dying,
and that is a different problem.

    tools/splits.py                          the current map, best guess
    tools/splits.py --map <uid>
    tools/splits.py --last 200               only the most recent 200 episodes
    tools/splits.py --theoretical            also print the sum of the bests

"Theoretical best" is the sum of every section's best - a lap nobody has
driven, made of pieces everyone has. It is the number worth chasing, and the
gap between it and your actual best is the total on the table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def load(map_uid: str, last: int = 0) -> list[dict]:
    path = os.path.join(RUNS, map_uid, "splits.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"no split log at {path}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-last:] if last else rows


def maps_with_splits() -> list[str]:
    if not os.path.isdir(RUNS):
        return []
    return sorted(d for d in os.listdir(RUNS)
                  if os.path.isfile(os.path.join(RUNS, d, "splits.jsonl")))


def sections(rows: list[dict]) -> list[list[int]]:
    """Section durations per episode: cp0 from the start, then each gap."""
    width = max((len(r.get("splits") or []) for r in rows), default=0)
    out: list[list[int]] = [[] for _ in range(width)]
    for r in rows:
        sp = r.get("splits") or []
        prev = 0
        for i, t in enumerate(sp):
            if t is None:
                break
            d = int(t) - prev
            prev = int(t)
            # A negative section means the clock reset mid-episode - a give-up
            # that was counted late. Dropping it is right; keeping it would
            # make that section's "best" an impossible number.
            if d >= 0:
                out[i].append(d)
    return out


def fmt(ms: float) -> str:
    return f"{ms / 1000.0:7.3f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", help="map uid (default: the only one with splits)")
    ap.add_argument("--last", type=int, default=0,
                    help="only the most recent N episodes")
    ap.add_argument("--theoretical", action="store_true")
    a = ap.parse_args()

    uid = a.map
    if not uid:
        found = maps_with_splits()
        if not found:
            raise SystemExit(f"no split logs under {RUNS} yet - "
                             "they are written once a run passes a checkpoint")
        if len(found) > 1:
            print("maps with split logs: " + ", ".join(found))
            print("pass --map <uid>")
            return 1
        uid = found[0]

    rows = load(uid, a.last)
    secs = sections(rows)
    if not secs:
        raise SystemExit("no checkpoints were reached in any logged episode")

    total = len(rows)
    print(f"{uid}: {total} episode(s), {len(secs)} timed section(s)\n")
    print(f"{'section':>10}  {'best':>9}  {'median':>9}  {'gap':>9}  reached")
    print("-" * 60)
    best_sum = 0.0
    complete = True
    for i, vals in enumerate(secs):
        label = f"start->cp{i}" if i == 0 else f"cp{i - 1}->cp{i}"
        if not vals:
            print(f"{label:>10}  {'-':>9}  {'-':>9}  {'-':>9}  0/{total}")
            complete = False
            continue
        ordered = sorted(vals)
        best = ordered[0]
        median = ordered[len(ordered) // 2]
        best_sum += best
        print(f"{label:>10}  {fmt(best)}  {fmt(median)}  {fmt(median - best)}  "
              f"{len(vals)}/{total}")

    finished = [r for r in rows if r.get("finished") and r.get("race_time")]
    if finished:
        pb = min(int(r["race_time"]) for r in finished)
        print(f"\nbest finished lap: {fmt(pb)}  ({len(finished)} finish(es))")
    else:
        print("\nno finished lap yet")

    if a.theoretical and complete:
        print(f"theoretical best:  {fmt(best_sum)}  "
              f"(every section's own best, in one lap)")
        if finished:
            print(f"on the table:      {fmt(pb - best_sum)}")
    elif a.theoretical:
        print("theoretical best needs every section driven at least once")

    worst = max(range(len(secs)),
                key=lambda i: (sorted(secs[i])[len(secs[i]) // 2]
                               - sorted(secs[i])[0]) if secs[i] else -1)
    if secs[worst]:
        label = f"start->cp{worst}" if worst == 0 else f"cp{worst - 1}->cp{worst}"
        print(f"\nbiggest spread is {label} - that is the section to aim a "
              f"hint at (cp_from/cp_to in the tuning config).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

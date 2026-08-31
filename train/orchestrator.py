#!/usr/bin/env python3
"""Per-track lifecycle orchestrator - SKELETON / DRY-RUN ONLY.

The plan (see FLEET_V2.md for the full reasoning):

    SURVEY    privileged account (Developer mode, our plugin).
              Offline Gbx parse + one scripted lap. Produces
              maps/<uid>.{occupancy,materials,effects} and appends decay
              constants to calibration.json. Idempotent - skipped if fresh.

    EXPLORE   privileged account. Full 106-dim obs, explore stage, provisional
              line from the map's own landmarks. Runs to handover
              (N finishes, no improvement in M) -> lines/<uid>-explored.json.

    RACE_BULK free fleet (School mode, SAC_GetData + adapter). N game
              instances, race stage on the explored line. Throughput lives
              here. On plain maps the adapter's ~62-dim obs is complete;
              maps with real effects also need RECONSTRUCT (not built).

    REFINE    privileged account. When a materially-new best line appears,
              ONE scripted survey pass along it, refresh the effect map for
              the changed sections, resume. Per-line event, not a live loop.

    DONE      split times plateau -> freeze, advance to the next track.

Nothing here launches anything yet. It models the state machine, prints the
commands each stage WOULD run, and checks which artifacts already exist. Wire
the `_run_*` methods to real subprocess calls once a School-mode game + the
adapter have been tested end to end.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.mapnames import name_for, uid_for            # noqa: E402

VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")

STAGES = ("SURVEY", "EXPLORE", "RACE_BULK", "REFINE", "DONE")


def _exists(*parts) -> bool:
    return os.path.isfile(os.path.join(ROOT, *parts))


def _fresh(path: str, max_age_days: float = 30.0) -> bool:
    try:
        return (time.time() - os.path.getmtime(path)) < max_age_days * 86400
    except OSError:
        return False


@dataclass
class TrackState:
    uid: str
    stage: str = "SURVEY"
    best_time: float | None = None
    last_best_line_len: int = 0
    finishes: int = 0
    note: str = ""

    @property
    def name(self) -> str:
        return name_for(self.uid, short=True)


class Orchestrator:
    def __init__(self, uids: list[str], seats: int = 4, instances: int = 2,
                 dry_run: bool = True):
        self.tracks = [TrackState(u) for u in uids]
        self.seats = seats
        self.instances = instances
        self.dry_run = dry_run
        self.state_path = os.path.join(ROOT, "runs", "orchestrator_state.json")
        self._load()

    def _load(self) -> None:
        try:
            with open(self.state_path) as f:
                saved = {t["uid"]: t for t in json.load(f).get("tracks", [])}
        except (OSError, ValueError):
            return
        for t in self.tracks:
            s = saved.get(t.uid)
            if s:
                t.stage = s.get("stage", t.stage)
                t.best_time = s.get("best_time")
                t.last_best_line_len = s.get("last_best_line_len", 0)
                t.finishes = s.get("finishes", 0)

    # -- artifact checks ------------------------------------------------------

    def survey_done(self, uid: str) -> bool:
        occ = os.path.join(ROOT, "maps", f"{uid}.occupancy.json")
        mat = os.path.join(ROOT, "maps", f"{uid}.materials.json")
        # effects file is optional (plain maps have none), so it is not required
        return _exists("maps", f"{uid}.materials.json") and _fresh(mat) \
            and (_exists("maps", f"{uid}.occupancy.json") and _fresh(occ)
                 or True)   # occupancy not yet produced by any tool; don't block

    def explore_line(self, uid: str) -> str | None:
        cand = os.path.join(ROOT, "lines", f"{uid}-explored.json")
        return cand if _exists("lines", f"{uid}-explored.json") else None

    # -- stage runners (STUBS) ---------------------------------------------

    def _cmd_survey(self, uid: str) -> list[str]:
        return [VENV_PY, "tools/maps.py", "--dump", uid,
                "--occupancy", "--materials", "--effects"]

    def _cmd_explore(self, uid: str) -> list[str]:
        return [VENV_PY, "train/train_sac.py", "--stage", "explore",
                "--seats", str(self.seats),
                "--handover", "20", "--handover-patience", "25", "--then-race",
                "--bootstrap", "straight",
                "--name", f"explore_{name_for(uid, short=True)}"]

    def _cmd_race_bulk(self, uid: str, line: str) -> list[str]:
        return [VENV_PY, "train/train_sac.py", "--stage", "race",
                "--line", line, "--seats", str(self.seats),
                "--instances", str(self.instances),
                "--auto-rollback",
                "--name", f"race_{name_for(uid, short=True)}"]

    def _cmd_refine(self, uid: str, line: str) -> list[str]:
        return [VENV_PY, "tools/maps.py", "--survey-along", line,
                "--refresh-effects", uid]

    def _run(self, cmd: list[str], label: str) -> int:
        if self.dry_run:
            print(f"    [dry-run] {label}: {' '.join(cmd)}")
            return 0
        raise NotImplementedError(
            "orchestrator stage execution is not wired yet - run the printed "
            "command by hand, or implement subprocess launch + monitoring here")

    # -- the machine -----------------------------------------------------

    def step_track(self, t: TrackState) -> None:
        print(f"  {t.name} [{t.uid[:6]}…]  stage={t.stage}")

        if t.stage == "SURVEY":
            if self.survey_done(t.uid):
                t.note = "survey artifacts present"
                t.stage = "EXPLORE"
            else:
                self._run(self._cmd_survey(t.uid), "SURVEY")
                t.note = "survey needed (dump not found or stale)"
            return

        if t.stage == "EXPLORE":
            line = self.explore_line(t.uid)
            if line:
                t.last_best_line_len = self._line_len(line)
                t.note = f"explored line ready ({t.last_best_line_len} pts)"
                t.stage = "RACE_BULK"
            else:
                self._run(self._cmd_explore(t.uid), "EXPLORE")
                t.note = "explore running to handover"
            return

        if t.stage == "RACE_BULK":
            line = self.explore_line(t.uid)
            if not line:
                t.stage = "EXPLORE"
                t.note = "explored line vanished - back to EXPLORE"
                return
            self._run(self._cmd_race_bulk(t.uid, line), "RACE_BULK")
            t.note = "bulk racing; watch splits for plateau"
            # REFINE / DONE transitions are driven by split-time telemetry,
            # which this skeleton does not read yet.
            return

        if t.stage == "REFINE":
            line = self.explore_line(t.uid)
            self._run(self._cmd_refine(t.uid, line or "?"), "REFINE")
            t.stage = "RACE_BULK"
            return

    def _line_len(self, path: str) -> int:
        try:
            with open(path) as f:
                d = json.load(f)
            return len(d.get("samples") or d.get("points") or [])
        except (OSError, ValueError):
            return 0

    def run_once(self) -> None:
        print(f"orchestrator: {len(self.tracks)} track(s), "
              f"{self.seats} seats, {self.instances} instances, "
              f"dry_run={self.dry_run}")
        for t in self.tracks:
            before = t.stage
            self.step_track(t)
            if t.note:
                print(f"      -> {t.note}"
                      + (f"  ({before} -> {t.stage})" if before != t.stage else ""))
        self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump({"saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "tracks": [asdict(t) for t in self.tracks]}, f, indent=2)
        print(f"  state -> {os.path.relpath(self.state_path, ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps", nargs="+",
                    help="map uids or name aliases (see maps/names.json)")
    ap.add_argument("--seats", type=int, default=4)
    ap.add_argument("--instances", type=int, default=2)
    ap.add_argument("--execute", action="store_true",
                    help="actually launch stages (NOT IMPLEMENTED - raises)")
    args = ap.parse_args()

    uids = []
    for m in args.maps:
        u = uid_for(m) or m
        uids.append(u)
    Orchestrator(uids, args.seats, args.instances,
                 dry_run=not args.execute).run_once()
    return 0


if __name__ == "__main__":
    sys.exit(main())

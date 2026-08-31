#!/usr/bin/env python3
"""Captures the game's own replay files as the policy improves.

The finish screen offers Improve / Save Replay / Exit, but we never touch it:
Trackmania already autosaves every personal best to

    Documents/Trackmania/Replays/Autosaves/
        <login>_<map>_PersonalBest_TimeAttack.Replay.Gbx

One file per map, with no time in the name - so the *next* PB overwrites it.
That is the whole problem this module solves: watch the directory, and the
moment a file's mtime moves, copy it out under a name that records which
episode produced it, before the next improvement clobbers it.

Doing it this way rather than driving the finish menu means nothing here breaks
when Nadeo moves a button, and no blind D-pad navigation is involved.

Standalone use (works without the trainer, names files by timestamp):

    python3 tools/replay_watcher.py runs/manual
"""
from __future__ import annotations

import os
import shutil
import threading
import time

AUTOSAVES = os.environ.get("TMAI_AUTOSAVES") or (
    "/mnt/4TB/SteamLibrary/steamapps/compatdata/2225070/pfx/drive_c/"
    "users/steamuser/Documents/Trackmania/Replays/Autosaves"
)

# How long a "this episode just finished" note stays valid. The game writes the
# replay a moment after the finish screen appears, so the note has to outlive
# that gap - but not so long that it mislabels an unrelated later write.
NOTE_TTL = 45.0


class ReplayWatcher:
    """Polls the autosave directory and copies out anything that changes."""

    def __init__(self, out_dir: str, watch_dir: str = AUTOSAVES, poll: float = 1.0):
        self.out_dir = out_dir
        self.watch_dir = watch_dir
        self.poll = poll
        self._seen: dict[str, float] = {}
        self._note: tuple[int, int, float | None, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.copied = 0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> bool:
        if not os.path.isdir(self.watch_dir):
            print(f"replay watcher: {self.watch_dir} not found - disabled",
                  flush=True)
            return False
        # Snapshot first, so the replays already sitting there from your own
        # driving are not all copied in on the first tick.
        self._seen = self._scan()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"replay watcher: watching {self.watch_dir} "
              f"({len(self._seen)} existing)", flush=True)
        return True

    def stop(self) -> None:
        self._stop.set()

    # -- api --------------------------------------------------------------

    def note(self, episode: int, step: int, race_time_ms: float | None) -> None:
        """Tell the watcher which episode is about to produce a replay.

        Called when an episode ends with finished=True. The write lands a
        second or two later, and whatever appears within NOTE_TTL is attributed
        to this episode.
        """
        with self._lock:
            self._note = (episode, step, race_time_ms, time.time())

    # -- internals --------------------------------------------------------

    def _scan(self) -> dict[str, float]:
        out = {}
        try:
            for fn in os.listdir(self.watch_dir):
                if not fn.lower().endswith(".gbx"):
                    continue
                full = os.path.join(self.watch_dir, fn)
                try:
                    out[fn] = os.stat(full).st_mtime
                except OSError:
                    continue
        except OSError:
            pass
        return out

    def _name_for(self, src: str) -> str:
        with self._lock:
            note = self._note
            if note and time.time() - note[3] > NOTE_TTL:
                note = None
        if note:
            ep, step, rt, _ = note
            stamp = f"t{rt / 1000:.3f}s" if rt else "tunknown"
            base = f"ep{ep:05d}_step{step // 1000:05d}k_{stamp}"
        else:
            # No episode claimed it - still worth keeping, just less useful.
            base = f"unmatched_{int(os.stat(src).st_mtime)}"
        return base + ".Replay.Gbx"

    def _capture(self, fn: str) -> None:
        src = os.path.join(self.watch_dir, fn)
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            dst = os.path.join(self.out_dir, self._name_for(src))
            # copy2 keeps mtime, so the copies stay in the same order as the
            # runs that produced them even if the names are ever wrong.
            shutil.copy2(src, dst)
            self.copied += 1
            print(f"  replay saved -> {os.path.relpath(dst)}", flush=True)
        except OSError as ex:
            print(f"  replay copy failed: {ex}", flush=True)

    def _run(self) -> None:
        while not self._stop.wait(self.poll):
            current = self._scan()
            for fn, mtime in current.items():
                if self._seen.get(fn) == mtime:
                    continue
                # A file being written is not a file finished being written.
                # Wait for its size to settle before copying a truncated Gbx.
                if self._settled(fn):
                    self._capture(fn)
                    self._seen[fn] = mtime
                else:
                    # Leave it out of _seen so the next tick retries it.
                    pass
            # Drop entries for files that vanished, so a delete-then-rewrite
            # cycle still registers as a change.
            for fn in list(self._seen):
                if fn not in current:
                    del self._seen[fn]

    def _settled(self, fn: str, tries: int = 5) -> bool:
        full = os.path.join(self.watch_dir, fn)
        last = -1
        for _ in range(tries):
            try:
                size = os.stat(full).st_size
            except OSError:
                return False
            if size == last and size > 0:
                return True
            last = size
            time.sleep(0.2)
        return False


def main() -> int:
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/manual"
    w = ReplayWatcher(os.path.abspath(out))
    if not w.start():
        return 1
    print("watching - drive a PB and it will be copied out. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        w.stop()
        print(f"\n{w.copied} replay(s) captured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

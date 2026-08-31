"""Cycle the map during training, so one model sees many tracks.

The single thing standing between "drives this track" and "drives Trackmania".
A policy trained on one map memorises that map: its corners, its straights,
where the finish is. Nothing about the network prevents generalisation - the
observation is already almost entirely map-agnostic, and the explore stage
builds its reference line from whatever map is loaded - but a training
distribution of one map can only ever teach one map.

So: every N episodes, load the next map. The environment notices the uid
change, re-asks for landmarks, rebuilds its line and reloads the occupancy
grid, and carries on. The replay buffer keeps everything, which is the point -
off-policy learning means transitions from map 3 are still teaching material
while the car is driving map 17.

    --maps "Downloaded/*.Map.Gbx" --map-every 40

Two things to be deliberate about:

**One reward for all of them.** Tuning config is per map by default, so a
rotation would quietly score map 2 differently from map 1 and the buffer would
mix the two. `--shared-config` pins one config for the whole run. Use it.

**Explore stage only.** A race line is a recorded lap of one specific track;
it cannot be regenerated for a map the car has never seen. Rotation is how you
train the explorer to handle anything, and the explorer is what makes an
unseen map tractable in the first place.
"""
from __future__ import annotations

import glob
import os

from stable_baselines3.common.callbacks import BaseCallback


class MapRotator(BaseCallback):
    """Move to the next map when this one is done with.

    Two gates, and which you want depends on what you are training.

    **mastery** (default) - move on when the track is actually learned: it has
    been finished `finishes` times AND has then gone `patience` finishes
    without a new best time. This is "get this one perfect first", and it is
    the right gate for a curriculum. A track that never gets finished never
    advances, which is correct: moving on from a track the car cannot complete
    teaches nothing and loses the one it was making progress on.

    **every** - a fixed number of episodes per map, regardless of how it went.
    Right for building a general driver out of many maps, wrong for a
    curriculum, because it moves on from a track mid-learning.

    Either way the switch itself is a `playmap` command through the game's own
    API. No menu navigation, no simulated clicks.

    Sends the command through the environment\'s own telemetry link rather
    than opening another connection, so it goes to the game this run is
    actually driving.
    """

    def __init__(self, maps: list[str], every: int = 0,
                 finishes: int = 5, patience: int = 25, verbose: int = 0):
        super().__init__(verbose)
        self.maps = list(maps)
        self.every = max(0, int(every))          # 0 = mastery gate
        self.finishes_needed = max(1, int(finishes))
        self.patience = max(1, int(patience))
        self.index = 0
        self.episodes = 0
        self.finishes = 0
        self.best_ms: int | None = None
        self.since_best = 0

    def _load(self, path: str) -> None:
        # get_attr reaches into the worker processes; env 0 owns the link the
        # game is on, and in splitscreen every seat shares that one game.
        try:
            links = self.training_env.get_attr("telem", indices=[0])
        except Exception as ex:
            print(f"  map rotation: cannot reach the telemetry link ({ex})",
                  flush=True)
            return
        if not links or links[0] is None:
            return
        try:
            links[0].command(f"playmap {path}", wait=5.0)
            print(f"  map rotation -> {os.path.basename(path)} "
                  f"({self.index + 1}/{len(self.maps)})", flush=True)
        except Exception as ex:
            print(f"  map rotation failed: {ex}", flush=True)

    def _advance(self) -> None:
        self.index = (self.index + 1) % len(self.maps)
        self.finishes = 0
        self.best_ms = None
        self.since_best = 0
        self._load(self.maps[self.index])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if dones is None:
            return True
        infos = [] if infos is None else infos
        for info, d in zip(list(infos) + [{}] * len(dones), dones):
            if not d:
                continue
            self.episodes += 1

            if self.every:
                if self.episodes % self.every == 0:
                    self._advance()
                continue

            # Mastery gate.
            if not (info or {}).get("finished"):
                continue
            self.finishes += 1
            t = (info or {}).get("race_time")
            if t is not None and (self.best_ms is None or int(t) < self.best_ms):
                self.best_ms = int(t)
                self.since_best = 0
                print(f"  {os.path.basename(self.maps[self.index])}: "
                      f"finish #{self.finishes} in {self.best_ms / 1000:.3f}s "
                      f"(new best)", flush=True)
            else:
                self.since_best += 1
            if (self.finishes >= self.finishes_needed
                    and self.since_best >= self.patience):
                print(f"\n  {os.path.basename(self.maps[self.index])} learned: "
                      f"{self.finishes} finishes, best "
                      f"{self.best_ms / 1000:.3f}s, no improvement in "
                      f"{self.patience}. Moving on.\n", flush=True)
                self._advance()
        return True


def resolve_maps(spec: str, prefix_docs: str) -> list[str]:
    """Turn a glob into the paths the GAME will understand.

    PlayMap takes a path as the game sees it, not as the filesystem does, so
    what goes over the wire is the part under Maps/ - `Downloaded/foo.Map.Gbx`
    - regardless of which prefix the file physically lives in.
    """
    root = os.path.join(prefix_docs, "Maps")
    found = sorted(glob.glob(os.path.join(root, spec))) or sorted(
        glob.glob(os.path.join(root, "**", spec), recursive=True))
    return [os.path.relpath(f, root) for f in found]

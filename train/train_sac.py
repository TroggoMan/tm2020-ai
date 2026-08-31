#!/usr/bin/env python3
"""SAC training against the live game.

Why SAC rather than NEAT: the game runs at 1x wall-clock and cannot be
fast-forwarded, so real seconds are the scarce resource. NEAT extracts one
fitness scalar from a ~30s run and throws away the ~2000 samples; SAC keeps
every transition in a replay buffer and reuses it. See the README.

    python3 train/train_sac.py --line lines/spring2026-03.json

Run the pad server and the broker first. Openplanet must be in School mode.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env.centerline import Centerline
from env.tm_env import TrackmaniaEnv, instance_ports
from env.ports import host_for, seat_ports
from train.bootstrap import BootstrapSAC, HintRelay
from train.handover import HandoverWatch, RACE_PROFILE, build_race_line
from train.regression import RegressionGuard
from train.rotate import MapRotator, resolve_maps
from train.nn_probe import NNProbe
from train.why import WhyLog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")


def explore_line(order: str | None = None):
    """Build the stage-one line before the env exists.

    The env needs a Centerline to construct, and in explore mode that line
    comes from the map's landmarks - so this opens its own short-lived link to
    the broker, asks, and closes it again.
    """
    from env.mapdata import Gates, provisional_line
    from env.tm_env import TelemetryLink

    link = TelemetryLink()
    try:
        # The plugin can take a moment to have a playground; a bare retry beats
        # failing the run because the map was still loading.
        reply = None
        for _ in range(10):
            reply = link.command("landmarks", wait=3.0)
            if reply and reply.get("ok"):
                break
            time.sleep(1.0)
        if not reply or not reply.get("ok"):
            raise SystemExit(
                "explore mode needs the map's landmarks, and the plugin did "
                f"not provide them ({(reply or {}).get('err', 'no reply')}). "
                "Is a map loaded and the plugin reloaded?")
        gates = Gates(reply.get("items", []))
        if order:
            gates.reorder([int(i) for i in order.split(",")])
        print(f"landmarks: {gates.describe()}", flush=True)
        print(f"route:     {gates.route()}", flush=True)
        if gates.order_source.endswith("(heuristic)"):
            print("           ^ if that order looks wrong for this track, "
                  "pass --gate-order (e.g. --gate-order 1,0)", flush=True)
        return provisional_line(gates)
    finally:
        link.close()


def write_meta(path: str, env, args, grad: int) -> None:
    """A sidecar describing what this model was trained AGAINST.

    The observation shape is checked by SB3 on load, so a mismatch there
    fails loudly. Nothing checks the rest, and the rest matters just as much:
    if the throttle was continuous when these weights were learned and is
    binary now, then an actor output of 0.3 used to mean 65% throttle and now
    means fully on. The model loads perfectly and drives differently.

    Same for the checkpoint test and the time charge - they change what the
    critic's numbers mean without changing any shape. Recording them means a
    resume can say so instead of quietly starting from a policy that was
    solving a different problem.
    """
    # Read the config rather than the env: `env` here is a VecEnv wrapping N
    # TrackmaniaEnvs in other processes, and it has no .cfg to ask.
    from env.config import TuningConfig, config_path
    cfg = TuningConfig(config_path(ROOT, None, ""))
    data = {"obs_dim": int(env.observation_space.shape[0]),
            "control_hz": args.control_hz, "stage": args.stage,
            "instances": args.instances, "seats": args.seats,
            "gradient_steps": grad,
            "action": dict(cfg.data.get("action", {})),
            "cp_mode": cfg.get("line", "cp_mode", "gate"),
            "par_speed": cfg.get("reward", "par_speed", 0.0),
            "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with open(path + ".meta.json", "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
    return data


def check_meta(path: str, args) -> None:
    """Say so when a resumed model was trained under different rules."""
    try:
        with open(path + ".meta.json") as f:
            old = json.load(f)
    except (OSError, json.JSONDecodeError):
        print("  no metadata for this model - it predates the sidecar, so "
              "what it was trained against is unknown", flush=True)
        return
    from env.config import TuningConfig, config_path
    now = TuningConfig(config_path(ROOT, None, ""))
    notes = []
    was = (old.get("action") or {}).get("binary_gas")
    is_ = now.data.get("action", {}).get("binary_gas")
    if was is not None and was != is_:
        notes.append(
            f"throttle was {'on/off' if was else 'continuous'} and is now "
            f"{'on/off' if is_ else 'continuous'} - the same actor output "
            f"means a different pedal position, so expect a dip before it "
            f"re-adapts")
    if old.get("cp_mode") and old["cp_mode"] != now.get("line", "cp_mode", "gate"):
        notes.append(f"checkpoints were credited by '{old['cp_mode']}' and are "
                     f"now by '{now.get('line', 'cp_mode', 'gate')}'")
    if old.get("control_hz") and abs(old["control_hz"] - args.control_hz) > 0.1:
        notes.append(f"control rate was {old['control_hz']}Hz, now "
                     f"{args.control_hz}Hz - every transition in the buffer "
                     f"is a different length of time to the new ones")
    if notes:
        print(f"  trained {old.get('saved', '?')} under different rules:",
              flush=True)
        for nte in notes:
            print(f"    - {nte}", flush=True)
        print("    The weights are still worth far more than a fresh start; "
              "this is a warning, not a problem.", flush=True)


def _tuning():
    """The tuning config the environment will load, read from the trainer.

    Same file, same loader: the warm-up performs hints and the reward pays for
    them, and those two must not read different numbers.
    """
    from env.config import TuningConfig, config_path
    return TuningConfig(config_path(ROOT, None, ""))


def auto_gradient_steps(instances: int, control_hz: float) -> int:
    """How many updates fit between two control ticks.

    This is the one place where "more instances" does NOT simply scale up, and
    it is worth being blunt about why.

    SB3 runs its gradient update on the same thread that steps the
    environment, so every update happens INSIDE the control period. Measured
    on this machine, batch 256 over a 256x256 net:

        CUDA   1 step 2.9ms   2 steps 5.9ms   4 steps 12.1ms   8 steps 27.2ms
        CPU    1 step 11.9ms  2 steps 25.7ms

    The period at 40Hz is 25ms. Overrun it and the car keeps driving while
    nothing is being sent to it, and - worse - every transition in the buffer
    still claims to be a 25ms one, so the model is learning a dynamics model
    of a game running at a rate it never ran at. Three instances at 6 updates
    a round measured 30Hz, not 40.

    So the update count is capped by the clock, not by the instance count.
    Adding instances therefore raises TRANSITIONS PER SECOND without raising
    updates per second, which lowers the updates-per-transition ratio. That is
    a genuine trade and it is the right side of it: more diverse, more
    on-policy data with fewer updates each beats hammering a small buffer,
    and the alternative corrupts the timing of every sample.

    Half the period is the budget, leaving the other half for the sockets,
    the observation and the vec-env pipes.
    """
    import torch
    budget_ms = 1000.0 / control_hz * 0.5
    # Calibrated from OVERRUNS, not from a benchmark. A synthetic timing of
    # the same nets says 1.2ms per step on this 4080, but that leaves out
    # replay sampling, the entropy-coefficient update and the vec-env round
    # trip: run for real at 20Hz, 8 gradient steps overran 5.4% of control
    # steps and 4 overran 1.3%. 6.0 is the number that makes this function
    # return the value that actually held its clock.
    per_step = 6.0 if torch.cuda.is_available() else 12.0
    fits = max(1, int(budget_ms / per_step))
    want = 2 * instances
    if want > fits:
        print(f"  {want} gradient steps would suit {instances} instances, but "
              f"only {fits} fit in {budget_ms:.0f}ms of a "
              f"{1000/control_hz:.0f}ms control period - using {fits}. "
              f"Overruns show as 'slip=' in the episode log.", flush=True)
    return min(want, fits)


def env_layout(seats: int, instances: int) -> list:
    """Flat list of (game, slot) for every car that feeds the learner.

    --seats and --instances are ADDITIVE, not either/or:

      --seats 4                -> [(0,0),(0,1),(0,2),(0,3)]            one game, 4 seats
      --instances 3            -> [(0,None),(1,None),(2,None)]         3 games, 1 car each
      --seats 4 --instances 2  -> [(0,0),(0,1),(0,2),(0,3),(1,None)]  game 0 split + game 1

    Only game 0 is splitscreen; the extra instances are single-car - a free
    account cannot host local multiplayer off local disk, and SAC_GetData only
    exposes the local player anyway. `slot` is None for a solo game (it uses
    instance_ports(game)) and an int for a seat (seat_ports(slot, game=game)).
    """
    seats = max(1, int(seats))
    instances = max(1, int(instances))
    layout = []
    for game in range(instances):
        for slot in range(seats):
            layout.append((game, slot if seats > 1 else None))
    return layout


def make_vec_env(instances: int, factory):
    """N games, ONE learner.

    This is the shape the whole fleet idea rests on and it is worth being
    explicit about, because the obvious alternative is wrong: running N copies
    of the trainer would give N separate models each learning from a third of
    the data, and they could not be combined afterwards. What we want is more
    SIMULATION feeding one policy - N environments filling ONE replay buffer,
    with one set of weights being updated from all of it. SAC is off-policy,
    so it does not care which game a transition came from.

    SubprocVecEnv rather than DummyVecEnv because each env spends its whole
    step blocked on a socket and a sleep; in one process they would serialise
    and N instances would run at 40/N Hz each. In separate processes they
    overlap, so N games cost the same wall-clock as one.

    One process is left as DummyVecEnv - no fork, no pipe, and a traceback you
    can actually read.
    """
    fns = [factory(i) for i in range(instances)]
    if instances == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


class EpisodeLog(BaseCallback):
    """One line per episode, plus periodic checkpoints and archive snapshots.

    Saving only at the end is not good enough when a run is measured in days -
    a crash or a killed container would lose everything. Save on a wall-clock
    interval and whenever a new best time lands.
    """

    def __init__(self, path, save_every_s=300, archive_every_s=1800,
                 n_envs=1, promote_to=""):
        super().__init__()
        self.n_envs = n_envs
        self.path = path
        self.name = os.path.basename(path)
        self.archive_dir = os.path.join(MODEL_DIR, "archive", self.name)
        self.save_every_s = save_every_s
        self.archive_every_s = archive_every_s
        # The shared "general driver": every checkpoint and every new best is
        # mirrored here so the next run - next stage, next track - starts from
        # the best brain so far. Weights only; the buffer stays per-run.
        self.promote_to = promote_to
        self.ep = 0
        self.best = None
        self.best_cp = 0
        self.t0 = time.time()
        self.last_save = time.time()
        self.last_archive = time.time()

    def _promote(self) -> None:
        """Mirror the current model to the shared general-driver path."""
        if not self.promote_to:
            return
        try:
            self.model.save(self.promote_to)
        except Exception as ex:
            print(f"  promote to {self.promote_to} failed: {ex}", flush=True)

    def _archive(self, tag: str) -> None:
        """Snapshot the model under a self-describing name.

        Model only, not the replay buffer: the buffer is ~100MB and, being
        off-policy data, stays valid for any policy you roll back to. Rolling
        back means restoring an old *policy* against the current experience,
        which is what you want anyway.
        """
        try:
            os.makedirs(self.archive_dir, exist_ok=True)
            self.model.save(os.path.join(self.archive_dir, tag))
            print(f"  archived {self.name}/{tag}.zip", flush=True)
        except Exception as ex:
            print(f"  archive failed: {ex}", flush=True)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []),
                              self.locals.get("dones", [])):
            if not done:
                continue
            self.ep += 1
            reason = "FINISH" if info.get("finished") else info.get("reason", "?")
            rt = info.get("race_time")
            steps = self.num_timesteps
            cp, cp_total = info.get("cp", 0), info.get("cp_total", 0)

            if info.get("finished") and rt:
                if self.best is None or rt < self.best:
                    self.best = rt
                    self.model.save(self.path + "_best")
                    self._promote()
                    print(f"  new best {rt/1000:.3f}s -> {self.path}_best.zip",
                          flush=True)
                    self._archive(f"ep{self.ep:05d}_step{steps//1000:05d}k"
                                  f"_t{rt/1000:.3f}s")
            elif cp > self.best_cp:
                # Getting further than ever before is worth a snapshot too -
                # it is the only progress signal on a track it has never
                # finished, which is most of them.
                self.best_cp = cp
                self._archive(f"ep{self.ep:05d}_step{steps//1000:05d}k_cp{cp}")

            mins = (time.time() - self.t0) / 60
            best = f"{self.best/1000:.3f}s" if self.best else "-"
            # Which game produced this episode, once there is more than one.
            # Without it a fleet log is an interleaved stream with no way to
            # tell "instance 2 has been stuck for ten minutes" from "the
            # policy is bad".
            inst = (f"i{info.get('instance', 0)} "
                    if self.n_envs > 1 else "")
            # Overruns mean the control loop did not finish inside its period,
            # so the transitions are not the 25ms ones the model thinks they
            # are. Silence here is what "it got slower and nobody noticed"
            # looks like.
            over = info.get("overruns", 0)
            slip = f"  slip={over}" if over else ""
            print(f"ep {self.ep:5d}  {inst}step {steps:7d}  "
                  f"{reason:8s}  t={(rt or 0)/1000:6.2f}s  "
                  f"cp={cp}/{cp_total}  best={best}  "
                  f"[{mins:.1f}m]{slip}", flush=True)

        now = time.time()
        if now - self.last_save > self.save_every_s:
            self.last_save = now
            self.model.save(self.path)
            self.model.save_replay_buffer(self.path + "_buffer")
            self._promote()
            print(f"  checkpoint saved ({self.num_timesteps} steps)", flush=True)
        if now - self.last_archive > self.archive_every_s:
            self.last_archive = now
            self._archive(f"ep{self.ep:05d}_step{self.num_timesteps//1000:05d}k")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", help="reference line json (not needed for --stage explore)")
    ap.add_argument("--gate-order",
                    help="comma-separated checkpoint order, e.g. 1,0 - overrides "
                         "the nearest-neighbour guess in explore mode")
    ap.add_argument("--maps", metavar="GLOB",
                    help="rotate the map during training, e.g. '*.Map.Gbx'. "
                         "One model across many tracks is the only thing that "
                         "turns 'drives this map' into 'drives Trackmania'. "
                         "Explore stage only - a race line is a recorded lap "
                         "and cannot be regenerated for an unseen map.")
    ap.add_argument("--map-every", type=int, default=0, metavar="N",
                    help="fixed episodes per map. 0 (default) uses the MASTERY "
                         "gate instead: move on only once the track has been "
                         "finished --map-finishes times and then stopped "
                         "improving. A track that is never finished never "
                         "advances, which is the point.")
    ap.add_argument("--map-finishes", type=int, default=5, metavar="N",
                    help="finishes needed before a map counts as learned")
    ap.add_argument("--map-patience", type=int, default=25, metavar="N",
                    help="finishes without a new best before moving on")
    ap.add_argument("--shared-config", action="store_true",
                    help="one tuning config for every map instead of one per "
                         "map. Use this with --maps: otherwise the reward "
                         "silently differs between tracks and the replay "
                         "buffer mixes them with no way to tell them apart.")
    ap.add_argument("--seats", type=int, default=1, metavar="N",
                    help="drive N cars inside ONE splitscreen game, instead of "
                         "N separate games. Each seat needs its own pad server "
                         "(8765, 8775, 8785, 8795) and they all share the one "
                         "broker.")
    ap.add_argument("--regress-window", type=int, default=20,
                    help="episodes in the rolling mean the regression guard "
                         "watches (0 disables it)")
    ap.add_argument("--regress-drop", type=float, default=0.25,
                    help="fraction of return that has to disappear before it "
                         "counts as a regression")
    ap.add_argument("--auto-rollback", action="store_true",
                    help="on a sustained regression, restore <name>_best.zip "
                         "in place. The replay buffer is kept.")
    ap.add_argument("--handover", type=int, default=0, metavar="N",
                    help="explore only: stop after N finishes with no further "
                         "improvement, build the race line from the best run, "
                         "and hand over. 0 (default) runs explore forever.")
    ap.add_argument("--handover-patience", type=int, default=25,
                    help="finishes without a new best before handing over")
    ap.add_argument("--then-race", action="store_true",
                    help="after handover, restart this process on the race "
                         "stage with the line it just built")
    ap.add_argument("--race-instances", type=int, default=0, metavar="N",
                    help="--then-race only: number of game instances for the "
                         "race stage (default: same as --instances). Set >1 to "
                         "pull extra games (e.g. a School-mode instance) into "
                         "the racer automatically at handover, on top of "
                         "--seats. The extra instances' plumbing must already "
                         "be up when the handover fires.")
    ap.add_argument("--stage", choices=("race", "explore"), default="race",
                    help="explore builds a provisional line through the map's "
                         "own checkpoint and finish landmarks, so it needs no "
                         "recorded lap; race uses --line")
    ap.add_argument("--steps", type=int, default=200_000)
    # Step budget for the race stage that --then-race exec's into. The explore
    # --steps is spent finding the line; the racer then refines lap time and
    # wants a much longer run. Override for a shorter/longer racer.
    ap.add_argument("--race-steps", type=int, default=10_000_000)
    # Warm-start a FRESH run from an existing policy: load actor+critic weights
    # from this .zip, keep an empty replay buffer, skip the scripted warm-up.
    # This is how the driving skill carries across stages and across tracks -
    # the handover uses it automatically, and you pass it by hand when starting
    # a new track from your best general model. Ignored when --resume finds a
    # local checkpoint (that path already has the weights AND the buffer).
    ap.add_argument("--init-from", type=str, default="", metavar="MODEL.zip")
    # Mirror every checkpoint / new best to this shared path - the "general
    # driver" that accumulates skill across every stage and track. Pair it
    # with --init-from <same path> and the runs feed each other continuously:
    # each one starts from the best brain so far and writes its gains back.
    ap.add_argument("--promote-to", type=str, default="", metavar="DRIVER.zip")
    ap.add_argument("--control-hz", type=float, default=40.0)
    # Everything from here to --cp-radius only SEEDS configs/<map>.json the
    # first time a map is seen. Once that file exists it wins, and it is
    # re-read while training runs - tune from the panel, not from here.
    ap.add_argument("--max-episode-s", type=float, default=120.0,
                    help="cap an episode. Two minutes: long enough that a car "
                         "still learning to steer can recover and carry on, "
                         "short enough that flailing does not burn the clock. "
                         "Explore raises it further on its own.")
    ap.add_argument("--stuck-seconds", type=float, default=5.0,
                    help="seconds below --stuck-speed before ending the episode")
    ap.add_argument("--stuck-speed", type=float, default=1.0,
                    help="m/s; below this counts as not moving (1.0 = 3.6 km/h)")
    ap.add_argument("--max-offset", type=float, default=30.0,
                    help="metres off the reference line before ending the episode")
    ap.add_argument("--w-soft", type=float, default=0.01,
                    help="penalty per metre per step outside the free corridor")
    ap.add_argument("--reset-mode", choices=("giveup", "restart"), default="giveup",
                    help="giveup presses the bound button and skips the ~8s intro")
    ap.add_argument("--giveup-button", default="b",
                    help="pad button bound to Give up in the game's controls")
    ap.add_argument("--finish-button", default="a",
                    help="pad button that takes 'Improve' on the finish screen")
    ap.add_argument("--giveup-settle-ms", type=float, default=400.0,
                    help="wait before pressing after a finish; grows on retry")
    ap.add_argument("--giveup-retries", type=int, default=3,
                    help="button attempts before falling back to a full restart")
    ap.add_argument("--w-weave", type=float, default=0.0,
                    help="penalty per unit of |steer change| - punishes twitching")
    ap.add_argument("--w-reversal", type=float, default=0.0,
                    help="penalty per full steering sign flip at speed")
    ap.add_argument("--cp-radius", type=float, default=20.0,
                    help="metres from a checkpoint landmark that counts as taken")
    ap.add_argument("--instances", type=int, default=1,
                    help="how many game instances feed the SAME model. Each "
                         "one needs its own running game, its own pad server "
                         "on 8765+i and its own broker on 8767+i - "
                         "tools/fleet.py brings those up")
    ap.add_argument("--gradient-steps", type=int, default=0,
                    help="gradient steps per collection round; 0 picks the "
                         "most that fit inside the control period (see "
                         "auto_gradient_steps)")
    ap.add_argument("--bootstrap", choices=("pursuit", "straight", "off"),
                    default="pursuit",
                    help="what to do during the warm-up, before the policy is "
                         "trained enough to act: 'pursuit' drives toward the "
                         "line at full throttle, 'straight' just accelerates, "
                         "'off' is SB3's uniform random flailing")
    ap.add_argument("--bootstrap-random", type=float, default=0.25,
                    help="fraction of warm-up steps left uniform random, so "
                         "the buffer still contains alternatives to compare "
                         "the scripted driving against")
    ap.add_argument("--learning-starts", type=int, default=2000,
                    help="transitions collected before the first gradient step")
    ap.add_argument("--buffer-size", type=int, default=2_000_000,
                    help="replay buffer capacity in transitions. Must be big "
                         "enough to still hold the run's GOOD episodes: at the "
                         "old 300k, a 2.5M-step run had aged out all eleven of "
                         "its completed laps and was training on a buffer with "
                         "no example of finishing at all. ~1KB per transition, "
                         "so 2M is about 2GB of RAM")
    ap.add_argument("--target-entropy", default="auto",
                    type=lambda v: v if v == "auto" else float(v),
                    help="how much randomness SAC insists on keeping. 'auto' "
                         "is SB3's -dim(action) = -3. A LESS negative number "
                         "asks for more randomness. Raising it to -1.0 was "
                         "tried to stop the entropy collapse (ent_coef falls "
                         "0.86 -> 0.02 by episode 100) and measured WORSE over "
                         "the first 70 episodes: 31%% of the way to the first "
                         "checkpoint against 55%% for stock, and no checkpoints "
                         "at all against 5. The extra randomness costs more "
                         "than the collapse does at that stage. If you retry "
                         "this, use a milder value (-2.0) and judge it past "
                         "episode 150, not before")
    ap.add_argument("--no-replay-capture", action="store_true",
                    help="do not copy the game's autosaved PB replays into runs/")
    ap.add_argument("--resume", action="store_true")
    # v2 observation (90 dims, with surfaces and effects) is not loadable by a
    # v1 model, so it gets its own name rather than a confusing shape error.
    # sac_tm.zip and its buffer stay on disk, still runnable with --name sac_tm.
    ap.add_argument("--name", default=None,
                    help="model name; defaults to sac_tm_v2 for race and "
                         "sac_explore for explore, so a stage-one run cannot "
                         "quietly overwrite the racing model")
    ap.add_argument("--save-every", type=float, default=300.0,
                    help="seconds between checkpoints")
    ap.add_argument("--archive-every", type=float, default=1800.0,
                    help="seconds between archive snapshots you can roll back to")
    args = ap.parse_args()

    if not args.name:
        args.name = "sac_explore" if args.stage == "explore" else "sac_tm_v2"
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, args.name)

    if args.stage == "explore":
        line = explore_line(args.gate_order)
        # The provisional line cuts through scenery, so the usual off-line
        # limit would end every episode instantly. Stage one's job is to find
        # the road at all; the lidar is what tells it where the road is.
        if args.max_offset < 100:
            args.max_offset = 250.0
        if args.max_episode_s < 90:
            args.max_episode_s = 120.0
        # The provisional line is a hint about WHERE THE FINISH IS, not a
        # racing line - it runs straight through scenery between checkpoints,
        # so the road almost never follows it. Charging the usual per-metre
        # penalty for leaving it punishes the car for driving on the only
        # surface available: episode 2 lost 23.8 to off_line against 18.0 of
        # progress earned. Progress toward the checkpoints is the whole signal
        # in this stage.
        args.w_soft = 0.0
        print("stage: EXPLORE - no recorded lap needed, driving toward the "
              "map's own checkpoints", flush=True)
    else:
        if not args.line:
            ap.error("--line is required unless --stage explore")
        line = Centerline.load(args.line)
    print(f"reference line: {len(line.points)} samples, {line.length:.0f}m",
          flush=True)

    # --seats and --instances are ADDITIVE (see env_layout). Explore keeps
    # instances at 1 - the extra games are for the racer - unless you really
    # asked for more games during explore.
    layout_instances = args.instances
    LAYOUT = env_layout(args.seats, layout_instances)

    def factory(k: int):
        """Build env k: the k-th car in LAYOUT, in whichever process owns it.

        LAYOUT[k] is (game, slot). A seat (slot is an int) shares its game's
        one plugin/broker and is told apart by `slot`; a solo game (slot None)
        has its own. A missing pad or broker shows up as a refused connection
        on a named port, not as two policies fighting over one car.
        """
        game, slot = LAYOUT[k]
        if slot is not None:
            ports = seat_ports(slot, game=game)          # a splitscreen seat
        elif game == 0 or args.seats <= 1:
            ports = instance_ports(game)                 # plain --instances: 8765, 8775, 8785…
        else:
            # additive: game 0 is splitscreen (pads 8765/8775/8785/8795), so a
            # solo extra game must take a pad clear of those - the seat range.
            ports = seat_ports(0, game=game)             # game 1 -> 8900, game 2 -> 8940

        def _make():
            return TrackmaniaEnv(
                line, control_hz=args.control_hz,
                profile="explore" if args.stage == "explore" else "",
                # One config for every map, so a rotation cannot silently
                # score track 2 differently from track 1.
                shared_config=bool(args.shared_config),
                # `instance` is the flat env id (unique per car, for the log);
                # ports come from (game, slot) above.
                instance=k, slot=slot,
                # Explore always rebuilds its own line from whatever map
                # ITS game has loaded. That is what lets instance 0 run one
                # track while instance 1 runs another: each env asks its own
                # game for landmarks rather than inheriting a line built from
                # somebody else's map.
                rebuild_line=(args.stage == "explore"),
                # host_for(), not localhost: with TMAI_HOSTS set, the games
                # can sit on other machines and still feed this one learner.
                pad_addr=(host_for(game), ports["pad"]),
                telem_addr=(host_for(game), ports["broker"]),
                max_episode_s=args.max_episode_s,
                stuck_seconds=args.stuck_seconds,
                stuck_speed=args.stuck_speed,
                max_offset=args.max_offset,
                w_soft=args.w_soft,
                reset_mode=args.reset_mode,
                giveup_button=args.giveup_button,
                finish_button=args.finish_button,
                giveup_settle_ms=args.giveup_settle_ms,
                giveup_retries=args.giveup_retries,
                w_weave=args.w_weave,
                w_reversal=args.w_reversal,
                cp_radius=args.cp_radius,
                capture_replays=not args.no_replay_capture)
        return _make

    n = len(LAYOUT)
    if n > 1:
        by_game = {}
        for k, (game, slot) in enumerate(LAYOUT):
            by_game.setdefault(game, []).append(slot)
        print(f"fleet: {n} cars -> ONE model", flush=True)
        for game, slots in sorted(by_game.items()):
            if len(slots) > 1:                       # splitscreen game
                pads = ", ".join(str(seat_ports(s, game=game)["pad"])
                                 for s in slots)
                print(f"  game {game}: {len(slots)} splitscreen seats, "
                      f"pads {pads}, broker {seat_ports(0, game=game)['broker']}",
                      flush=True)
            else:                                    # solo game
                p_ = instance_ports(game)
                print(f"  game {game}: 1 car, pad {p_['pad']}, "
                      f"plugin {p_['plugin']}, broker {p_['broker']}", flush=True)
    env = make_vec_env(n, factory)

    grad = args.gradient_steps or auto_gradient_steps(n, args.control_hz)
    print(f"obs dim {env.observation_space.shape[0]}, "
          f"control {args.control_hz:.0f}Hz, {grad} gradient steps/round",
          flush=True)

    if args.resume and os.path.exists(path + ".zip"):
        print("resuming from", path + ".zip", flush=True)
        model = BootstrapSAC.load(path, env=env)
        check_meta(path, args)
        # A resumed model has a trained policy, so there is nothing to warm up
        # and the scripted driver would only overwrite it.
        model.bootstrap = "off"
        # The instance count can change between runs; the update ratio has to
        # follow it rather than staying at whatever the saved model used.
        model.gradient_steps = grad
        # The buffer is the expensive part - every transition in it cost real
        # wall-clock. Resuming without it throws that away and relearns blind.
        buf = path + "_buffer.pkl"
        if os.path.exists(buf):
            model.load_replay_buffer(path + "_buffer")
            print(f"  replay buffer restored: {model.replay_buffer.size()} transitions",
                  flush=True)
        else:
            print("  no saved replay buffer - starting with an empty one", flush=True)
    else:
        model = BootstrapSAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            # Big enough to still contain the good episodes.
            #
            # At 300k this silently threw away everything it had learned. A
            # 5476-episode run reached 2.48M steps, so the buffer held the last
            # 12% - and ALL ELEVEN of its completed laps had aged out. The
            # policy was training on a buffer containing zero examples of
            # finishing the track, which is self-reinforcing: it drifts down,
            # the buffer refills with the worse episodes, and that is all there
            # is left to learn from. Mean distance peaked at 415 m while the
            # finishes were still in memory and fell to 272 m once they were
            # not.
            #
            # A transition is ~1KB here (119-dim obs stored twice), so 2M is
            # about 2GB of RAM - nothing on a 62GB machine, and it covers a
            # full overnight run.
            buffer_size=args.buffer_size,
            # Every transition costs real wall-clock, so start learning early
            # and take several gradient steps per environment step.
            learning_starts=args.learning_starts,
            batch_size=256,
            train_freq=1,
            gradient_steps=grad,
            tau=0.005,
            gamma=0.995,
            ent_coef="auto",
            # How much randomness SAC insists on keeping. There IS an entropy
            # collapse on this env - ent_coef falls 0.86 -> 0.02 by episode 100
            # and the run's best stretch is the one where it was still ~0.09 -
            # but simply asking for more entropy is NOT the fix. Raising the
            # target to -1.0 was measured over 70 fresh episodes against an
            # otherwise identical baseline and came out worse on every count
            # (31% of the way to the first checkpoint vs 55%, zero checkpoints
            # vs five). A more random policy explores more and drives worse,
            # and at this stage the driving matters more. Stock unless asked.
            target_entropy=args.target_entropy,
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=os.path.join(ROOT, "logs", "tb"),
            verbose=0,
            # The warm-up drives instead of flailing. Uniform random actions
            # over three axes essentially never produce "full throttle,
            # straight, for two seconds", so the critic's first opinion of the
            # world is formed entirely from footage of a car snaking into a
            # wall. See train/bootstrap.py.
            bootstrap=args.bootstrap,
            bootstrap_random=args.bootstrap_random,
            # Hints the warm-up should perform, e.g. a brake tap in the
            # 400-600km/h window to put speed slides in the buffer at all.
            # Read from the same tuning config the reward reads, so the
            # scripted tap and the reward that pays for it can never disagree
            # about how long a tap is.
            hints=_tuning().data.get("hints"),
            control_hz=args.control_hz,
        )
        if args.init_from and os.path.exists(args.init_from):
            # Carry the brain forward. set_parameters copies actor, critic and
            # target-critic weights; exact_match=False tolerates a changed
            # net_arch or obs layout as long as the shapes that DO line up are
            # loaded. The replay buffer stays empty on purpose - its rewards
            # were scored under a different regime. Whether to still run the
            # scripted warm-up is the caller's call via --bootstrap: a healthy
            # policy needs none ("off"), but a policy that has collapsed into a
            # bad basin benefits from re-seeding the fresh buffer with good
            # driving, so pass --bootstrap pursuit to do that.
            model.set_parameters(args.init_from, exact_match=False)
            model.bootstrap = args.bootstrap
            print(f"  warm-started from {args.init_from} "
                  f"(weights only, empty buffer, "
                  f"warm-up={args.bootstrap})", flush=True)
        elif args.init_from:
            print(f"  --init-from {args.init_from} not found - "
                  f"starting from random weights", flush=True)
        if args.bootstrap != "off":
            print(f"warm-up: scripted '{args.bootstrap}' driver for the first "
                  f"{args.learning_starts} transitions "
                  f"({args.bootstrap_random:.0%} left random)", flush=True)
        if getattr(model, "hints", None):
            print("warm-up hints: "
                  + ", ".join(f"{h.name} ({h.hold_ms:.0f}ms tap, "
                              f"{h.speed_from_kmh:.0f}-{h.speed_to_kmh:.0f}km/h)"
                              for h in model.hints), flush=True)

    # Ctrl-C (what the web panel's Stop sends) should save, not discard.
    def save_and_exit(signum, frame):
        print("\nstopping - saving model and replay buffer", flush=True)
        model.save(path)
        write_meta(path, env, args, grad)
        try:
            model.save_replay_buffer(path + "_buffer")
        except Exception as ex:
            print("could not save replay buffer:", ex, flush=True)
        env.close()
        sys.exit(0)

    # A background job started from a non-interactive shell inherits SIGINT
    # ignored, so the panel's Stop must be catchable as SIGTERM too.
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    handover = None
    if args.stage == "explore" and args.handover:
        handover = HandoverWatch(args.handover, args.handover_patience)

    callbacks = [
        # First in the list: it publishes each env's requested warm-up action
        # for the *next* sample, so it should run before anything that might
        # stop the rollout.
        HintRelay(),
        EpisodeLog(path, args.save_every, args.archive_every, n_envs=n,
                   promote_to=args.promote_to),
        WhyLog(os.path.join(ROOT, "logs", "why.jsonl")),
        NNProbe(os.path.join(ROOT, "logs", "nn_state.json")),
    ]
    if args.maps:
        if args.seats > 1:
            ap.error("--maps rotates the map for a whole game, and all four "
                     "splitscreen seats share one game - so they cannot be on "
                     "different maps. Use separate --instances for different "
                     "maps, and seats for more cars on the same one.")
        if args.stage != "explore":
            ap.error("--maps needs --stage explore: a race line is a recorded "
                     "lap of one track and cannot be rebuilt for an unseen map")
        from env.prefixes import docs_dir
        rotation = resolve_maps(args.maps, docs_dir(0))
        if not rotation:
            ap.error(f"no maps matched {args.maps!r}")
        print(f"map curriculum: {len(rotation)} map(s)", flush=True)
        for m in rotation:
            print(f"    {m}", flush=True)
        if not args.shared_config:
            print("  WARNING: without --shared-config each map gets its own "
                  "tuning file, so the reward will differ between tracks and "
                  "the buffer will mix them.", flush=True)
        if args.map_every:
            print(f"  gate: a fixed {args.map_every} episodes per map",
                  flush=True)
        else:
            print(f"  gate: mastery - {args.map_finishes} finishes and then "
                  f"{args.map_patience} without a new best", flush=True)
        callbacks.append(MapRotator(rotation, args.map_every,
                                    args.map_finishes, args.map_patience))

    if args.regress_window:
        callbacks.append(RegressionGuard(
            window=args.regress_window, drop=args.regress_drop,
            rollback_to=(path + "_best") if args.auto_rollback else None))
    if handover is not None:
        callbacks.append(handover)
        print(f"handover: will stop after {args.handover} finish(es) with no "
              f"improvement in {args.handover_patience}, then build the race "
              f"line" + (" and start the race stage" if args.then_race else ""),
              flush=True)

    try:
        model.learn(total_timesteps=args.steps,
                    callback=callbacks,
                    reset_num_timesteps=not args.resume)
    finally:
        model.save(path)
        write_meta(path, env, args, grad)
        try:
            model.save_replay_buffer(path + "_buffer")
        except Exception as ex:
            print("could not save replay buffer:", ex, flush=True)
        # Grab the map uid before the envs go away - the handover needs it to
        # find the traces, and a VecEnv cannot be asked once it is closed.
        try:
            args._map_uid = env.get_attr("map_uid")[0]
        except Exception:
            args._map_uid = None
        env.close()
        print("saved", path + ".zip", flush=True)

    if handover is not None and handover.done:
        return do_handover(args, handover)
    return 0


def do_handover(args, watch) -> int:
    """Explore is finished. Build the racer's inputs and, optionally, start it.

    Three artefacts move across, and it is worth being precise about which is
    which, because only one of them is the explorer's *driving*:

      the explored path  -> smoothed, then used as the progress axis and the
                            lookahead. A route, not a line to trace.
      the occupancy grid -> untouched. This is the map's own geometry and it
                            is what the lidar beams are cast against.
      the landmarks      -> untouched. The checkpoint gates.
      the DRIVING POLICY -> carried. The racer is warm-started from the explore
                            weights (--init-from) and keeps writing back to the
                            shared driver (--promote-to). The explorer's job is
                            to learn to drive - throttle, brake, steering,
                            staying on the road, taking a corner - and that
                            skill is track-agnostic, so the racer should not
                            relearn it from scratch. It only has to get fast.
    """
    uid = getattr(args, "_map_uid", None)
    if not uid:
        print("handover: no map uid known - cannot find the traces", flush=True)
        return 1

    line_path = build_race_line(ROOT, uid)
    if not line_path:
        return 1

    # Relax line-following for the race stage, in that map's own config, so it
    # survives a restart and shows up in the panel where you can see it.
    from env.config import config_path
    cfg_path = config_path(ROOT, uid, "")
    try:
        with open(cfg_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}
    data.setdefault("line", {}).update(RACE_PROFILE)
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"handover: {os.path.basename(cfg_path)} line profile relaxed "
          f"({RACE_PROFILE})", flush=True)

    if not args.then_race:
        print("\nhandover complete. Start the racer with:\n"
              f"  train/train_sac.py --stage race --line {line_path}\n"
              "(--then-race would have done that automatically)", flush=True)
        return 0

    # Restart this process as the race stage. exec rather than spawn: the
    # explore model, its buffer and its sockets are all saved and closed by
    # now, and the game itself is a separate process that stays up.
    # Carry --seats through, and let --race-instances add games on top of it
    # (--seats and --instances are additive - see env_layout). This is how the
    # racer picks up a 2nd instance with no hand intervention.
    race_instances = args.race_instances or args.instances
    # Carry the machine-shaped settings that must not regress across the
    # handover: the control rate the buffer was timed at, and the gradient-step
    # count that keeps that rate honest. Auto-sizing re-picks a higher number
    # as instances scale and that is exactly what let slip climb back to the
    # hundreds; carry the explicit value, fall back to a safe 2. Reward / stuck
    # / offset are NOT carried - those are race-specific and come from
    # configs/<map>.race.json.
    grad = args.gradient_steps or 2
    explore_model = os.path.join(MODEL_DIR, args.name or "sac_explore") + ".zip"
    # The shared general driver: carry whatever this run was feeding, else
    # default to models/driver.zip so the loop closes even on a bare command.
    driver = args.promote_to or os.path.join(MODEL_DIR, "driver.zip")
    argv = [sys.executable, os.path.abspath(__file__),
            "--stage", "race", "--line", line_path,
            "--name", "sac_tm_v2",
            "--control-hz", str(args.control_hz),
            "--gradient-steps", str(grad),
            "--steps", str(args.race_steps),
            "--seats", str(args.seats),
            "--instances", str(race_instances),
            "--init-from", explore_model,
            "--promote-to", driver]
    print(f"\nhandover: starting the race stage\n  "
          + " ".join(argv[1:]) + "\n", flush=True)
    os.execv(sys.executable, argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Gymnasium environment wrapping the live game.

Two sockets, both already proven:
  8765  pad server   - we write [steer, gas, brake]
  8767  telemetry broker - we read telemetry AND send commands (restart, playmap)

Point at the broker rather than the plugin's 8766 so the GUI and any listener
can watch the same stream while training runs.

This is a hard real-time environment. The game does not pause while the policy
thinks, so step() paces itself to a fixed control period and reads whatever the
newest telemetry sample is. Two consequences, both deliberate:

  * The measured action round-trip is ~31ms (see tools/closed_loop_test.py), so
    an action's effect shows up one to two steps later. The previous action is
    part of the observation to keep the MDP honest about that, which is the
    same trick tmrl's real-time RL formulation uses.
  * Nothing here blocks waiting for a *specific* sample. If the game stalls we
    reuse the last one rather than desynchronising the control clock.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import socket
import threading
import time

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # keeps the module importable for the recorder
    gym = None
    spaces = None

try:
    from tools.replay_watcher import ReplayWatcher
except ImportError:  # the env still works, it just won't capture replays
    ReplayWatcher = None

from .centerline import Centerline, car_frame
from .ports import broker_addr, instance_ports, pad_addr  # noqa: F401
from .config import TuningConfig, config_path
from .lidar import BEAM_ANGLES_DEG, Lidar, load_or_fetch
from .mapdata import FAR, EffectMap, Gates
from .surfaces import (EFFECT_CLASSES, GROUP_NAMES, N_GROUPS, group_index,
                       is_road_name,
                       is_border, material_name)

# Grip classes where a slide IS the fast line, so the anti-twitch weave
# penalty must not fire: sawing the wheel on ice/dirt/plastic is how you
# speedslide, and a policy taught to keep the wheel still there never learns
# the surface. Road/wood/metal keep the penalty - flutter there is just noise.
_SLIDE_GROUPS = frozenset({"plastic", "dirt", "grass", "ice", "wet"})
from .speedslide import evaluate as sd_evaluate
from .hints import (Tapper, active as active_hint, parse as parse_hints,
                    resolve_marks)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Instance 0's ports. Everything multi-instance goes through env/ports.py,
# which is stdlib-only so the pad server and the panel can share it.
PAD_ADDR = pad_addr(0)
TELEM_ADDR = broker_addr(0)   # the broker, not the plugin directly

LOOKAHEAD = (5.0, 10.0, 17.0, 27.0, 40.0, 60.0)

# Observation layout. Kept as a table because the panel's network view labels
# its input nodes from it, and because a silent mismatch between this and
# _observe() would be very hard to spot from the outside.
OBS_GROUPS = [
    ("speed", 1), ("vel", 3), ("up", 3), ("ahead", 3 * len(LOOKAHEAD)),
    ("offset", 1), ("slip", 4), ("adher", 1), ("grnd", 1), ("gear", 1),
    ("rpm", 1), ("prev", 3),
    # v2: what the tyres are on, and what is about to happen to the car.
    # wet = WetnessValue01, how wet the wheels are (0-1, non-linear grip loss,
    #       and on wood even 1% matters). submerged = WaterImmersionCoef, how
    #       deep the car is in water RIGHT NOW - a different physics regime
    #       (heavy drag, floaty) from skimming a water surface.
    ("surface", 4 * N_GROUPS), ("icing", 4), ("dirt", 4),
    ("wet", 1), ("submerged", 1),
    ("turbo", 3), ("reactor", 5), ("cruise", 1), ("slowmo", 1),
    ("side", 1), ("airbrake", 1),
    ("fx_ahead", 2 * len(EFFECT_CLASSES)),
    # Ground rays against the map's block grid. All 1.0 (nothing in range) when
    # no occupancy dump is available, so the vector is the same shape whether
    # or not the map has been scanned - a model must never see a different
    # observation length depending on what the plugin happened to answer.
    ("lidar", len(BEAM_ANGLES_DEG)),
    # --- edge awareness. APPENDED, and it must stay appended. ------------
    #
    # Everything above keeps the index it had at 119 dims, so an older model's
    # first layer can be migrated by copying its 119 input columns and
    # zero-filling these - identical behaviour on day one, and it can learn to
    # use them from there. Insert a group in the middle instead and every
    # index after it shifts, which silently invalidates the whole layer.
    #
    # border: is THIS wheel on a track border (the rubber kerb)? Measured on a
    #   surface test track: `Rubber` is the road border, and the project's own
    #   grip grouping files it under `road` - identical to asphalt - so the
    #   policy could not tell a kerb from the racing line. It is the only
    #   fine-grained per-wheel "you are at the edge" signal the game gives.
    #   Left in the `road` grip group as well: grip and meaning are different
    #   questions and both answers are wanted.
    #
    # sides_ahead: does the track have BARRIERS at each lookahead distance?
    #   Platform pieces have no sides, Road pieces do - except Open* ones,
    #   which are road-named and open. On the training track ALL SEVEN
    #   edgeless blocks sit between checkpoint 4 and checkpoint 5, which is
    #   exactly where it reaches CP4 78% of the time and CP5 never: the car
    #   spends four fifths of a lap learning that going wide is survivable and
    #   then the walls vanish with nothing in its observation to say so.
    ("border", 4),
    ("sides_ahead", len(LOOKAHEAD)),
    # --- air control. APPENDED, same reasoning as above. -----------------
    #
    # air: [ground_dist / 20, flying_seconds / 3]. Both arrive in the
    #   telemetry every frame and were being thrown away. ground_dist is how
    #   far the wheels are above whatever is below them, which is the whole
    #   question when timing a landing - "how long have I got left".
    #
    # omega: angular velocity in the car's own frame [pitch, yaw, roll] rate,
    #   rad/s / 5. THE reason this exists: a brake tap in the air stops PITCH
    #   but not YAW, and yaw is corrected by countersteering (user). Both are
    #   RATE control, and the observation had attitude (`up`) but no rate at
    #   all - so no reward shaping could ever have taught either, because the
    #   policy had no input to act on. Differenced from the dir/up/left basis
    #   across frames because the plugin does not ship it.
    ("air", 2),
    ("omega", 3),
    # --- chassis. APPENDED. Every one of these was measured to VARY over
    # 6300 samples of real driving; the fields that came back constant
    # (adherence, brake_coef, sim_coef, wear) are deliberately NOT here.
    #
    # damper: per-wheel suspension travel, 0..0.2. Load transfer under
    #   braking and cornering, and the clearest landing signal there is.
    # steer_angle: the front wheels' ACTUAL angle, which is not the
    #   commanded steering - the game ramps toward it per frame.
    # skidding: 0..4, how many wheels are skidding.
    # top_contact: the car is on its roof.
    ("damper", 4), ("steer_angle", 1), ("skidding", 1),
    ("top_contact", 1),
    # --- HOW FAR IS THE EDGE. APPENDED, same reasoning as above. ---------
    #
    # The observation had `offset` (metres from the reference line) and, since
    # the edge-awareness block, `sides_ahead` (are there barriers?). It never
    # had the track's WIDTH - `_track_hw` was loaded from the route model and
    # then read by nothing at all.
    #
    # So an offset of 5m was the same number on a 12m-half-width road as on an
    # 8m platform: comfortable in one case, one car-width from a fall in the
    # other, and no input distinguished them. The policy could learn "the
    # barriers stop here" from sides_ahead but never "and the ground stops
    # THERE", which is the question that actually kills it.
    #
    # Measured on Summer 2026-06: 109 of 183 episodes reach CP4 and 2 reach
    # CP5. Every stalled one dies between 1400m and 1446m, on the unbarriered
    # platform run (PlatformTechCurve2In -> PlatformTechBaseOnLandHill3),
    # driving off the side and falling.
    #
    # Note what the widths actually are on that track: roads are 6m half-width
    # and the platforms are 8m. The platforms are WIDER. So the danger is not
    # narrowness - it is that going wide on a road scrapes a barrier and is
    # recoverable, while going wide on a platform is a fall. The lethal signal
    # is therefore the CONJUNCTION "margin low AND sides_ahead 0", which the
    # policy can only learn once both halves exist. It had sides_ahead already
    # and no margin at all.
    #
    # The occupancy grid cannot substitute: its cells are 32m x 32m, so a
    # 16m-wide platform sits inside one cell and the lidar reads that whole
    # cell as solid ground.
    #
    #   margin:     (half_width - |offset|) / half_width, clipped 0..1, at the
    #               car. 1.0 = dead centre, 0.0 = at the edge. Normalised by
    #               the width so it means the same thing on any piece.
    #   width_ahead: half_width at each lookahead distance, / 16m. Anticipatory
    #               in the same way sides_ahead is - a road narrowing into a
    #               platform is visible before it arrives, not on entry.
    #
    # Unknown width reads as 1.0 (wide and safe) rather than 0.0: a model that
    # believes it is boxed in when it is not drives timidly, which is
    # recoverable, whereas believing a platform is a motorway is not.
    ("margin", 1),
    ("width_ahead", len(LOOKAHEAD)),
    # --- HOW FAR IS THE END. APPENDED, same reasoning as above. ----------
    #
    # Every other line-relative input is LOCAL: `ahead` is the next six points
    # in the car's own frame, `offset` and `margin` are lateral. Nothing said
    # where the car was along the lap, so start->CP1 and CP4->CP5 were
    # indistinguishable states if the geometry rhymed, and the episode's end
    # arrived from nowhere.
    #
    # That matters most for sector drilling: the attempt ends at a stop line
    # the network cannot see, paying a large terminal bonus the critic has no
    # way to anticipate - so the value gets smeared over every state that looks
    # similar. With this, the terminal is predictable and the policy can push
    # for the line the way a human drilling a sector does.
    #
    # Useful outside the curriculum too: on a full lap it is distance to the
    # finish, which is the "how much is left" signal a racing line depends on.
    # 1.0 = the whole line remains, 0.0 = at the end.
    ("dist_to_stop", 1),
    # --- WHERE AM I ON THIS TRACK. APPENDED. -----------------------------
    #
    # Absolute position along the lap, 0.0 at the start line and 1.0 at the
    # finish. Distinct from dist_to_stop, which is TASK-relative and resets
    # every time the curriculum advances a sector - this one means the same
    # thing in every phase and on every episode.
    #
    # It is what lets the policy learn the track rather than react to it: a
    # human knows the hairpin is coming at two thirds distance and brakes for
    # it before seeing it. Without an absolute position the network only had
    # the six lookahead points, so two corners of similar shape were the same
    # state and had to be driven the same way.
    #
    # The cost is honest: this invites memorising ONE track, which is the
    # opposite of a general road driver. It earns its place while chasing a
    # lap record on a known circuit; drop it when training for generality.
    ("lap_progress", 1),
]
OBS_DIM = sum(n for _, n in OBS_GROUPS)
# Width before the margin/width_ahead block. Migrate a 141-dim model forward
# with tools/migrate_obs.py --old-obs 141.
OBS_DIM_PRE_MARGIN = OBS_DIM - 1 - len(LOOKAHEAD) - 1
# Width before dist_to_stop. Migrate a 148-dim model with
# tools/migrate_obs.py --old-obs 148.
OBS_DIM_PRE_DIST = OBS_DIM - 2
# Width before lap_progress. Migrate a 149-dim model with --old-obs 149.
OBS_DIM_PRE_LAPPOS = OBS_DIM - 1
# The layout before edge awareness was added. A model saved at this width can
# be migrated forward; see tools/migrate_obs.py.
OBS_DIM_PRE_EDGE = OBS_DIM - 4 - len(LOOKAHEAD) - 2 - 3
# Width after edge awareness but before air control.
OBS_DIM_PRE_AIR = OBS_DIM - 2 - 3 - 7
# Width after air control but before the chassis block.
OBS_DIM_PRE_CHASSIS = OBS_DIM - 7

# CGamePlaygroundUIConfig::EUISequence. Only the ones we act on.
UI_NONE = 0
UI_PLAYING = 1
UI_INTRO = 2
UI_OUTRO = 3
UI_FINISH = 11


class TelemetryLink:
    """Owns the single plugin connection: telemetry in, commands out.

    Connects to the broker, which multiplexes the plugin's single client slot,
    so this can run alongside the GUI.
    """

    def __init__(self, addr=TELEM_ADDR, timeout=10.0):
        self.sock = socket.create_connection(addr, timeout=timeout)
        self.sock.settimeout(1.0)
        self.latest: dict | None = None
        self.replies: list[dict] = []
        self._buf = b""
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            if not data:
                return
            self._buf += data
            while b"\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with self._lock:
                    # Command replies have no "car" key; telemetry always does.
                    if "car" in rec:
                        self.latest = rec
                    else:
                        self.replies.append(rec)

    def get(self) -> dict | None:
        with self._lock:
            return self.latest

    def command(self, cmd: str, wait: float = 2.0) -> dict | None:
        with self._lock:
            self.replies.clear()
        self.sock.sendall((cmd + "\n").encode())
        deadline = time.time() + wait
        while time.time() < deadline:
            with self._lock:
                if self.replies:
                    return self.replies.pop(0)
            time.sleep(0.005)
        return None

    def close(self):
        self._stop = True
        try:
            self.sock.close()
        except OSError:
            pass


class PadLink:
    """Connection to the pad server, which reconnects if it drops.

    Restarting the pad server used to kill the trainer outright with a
    BrokenPipeError, throwing away however many hours were in the replay
    buffer. A run measured in days cannot be that brittle about a process it
    does not own - and with a fleet of them, one pad restarting must not take
    a learner down.
    """

    def __init__(self, addr=PAD_ADDR, timeout=5.0):
        self.addr = addr
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.drops = 0
        self._connect()

    def _connect(self) -> bool:
        try:
            self.sock = socket.create_connection(self.addr, timeout=self.timeout)
            self.sock.settimeout(1.0)
            return True
        except OSError:
            self.sock = None
            return False

    def _send(self, line: str, retry: bool = True) -> bool:
        if self.sock is None and not self._connect():
            return False
        try:
            self.sock.sendall(line.encode())
            try:
                self.sock.recv(64)
            except socket.timeout:
                pass
            return True
        except OSError:
            # Dropped mid-command. Reconnect once and resend, so a pad server
            # restart costs one lost action rather than the whole run.
            self.drops += 1
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
            if retry and self._connect():
                return self._send(line, retry=False)
            return False

    def act(self, steer: float, gas: float, brake: float):
        self._send(f"act {steer:.4f} {gas:.4f} {brake:.4f}\n")

    def press(self, button: str, hold_ms: float = 80.0):
        self._send(f"press {button} {hold_ms:.0f}\n")

    def reset(self):
        self.act(0.0, 0.0, 0.0)

    def close(self):
        try:
            self.reset()
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass


def _real_pos(pos) -> bool:
    """A position that is actually a position.

    The plugin's heartbeat record carries no `pos` at all, and a car sitting
    exactly on the world origin does not happen on any real map - every
    Trackmania map is laid out in positive coordinates well away from it. So
    "missing or the origin" is a reliable test for "this record is empty",
    which is worth having because the alternative reading - a car 560m from
    its line - sends you looking for a map mismatch that is not there.
    """
    if pos is None:
        return False
    try:
        return any(abs(float(q)) > 1e-6 for q in pos)
    except (TypeError, ValueError):
        return False


class TrackmaniaEnv(gym.Env if gym else object):
    metadata = {"render_modes": []}

    def __init__(self,
                 centerline: Centerline,
                 control_hz: float = 40.0,
                 max_episode_s: float = 90.0,
                 max_offset: float = 22.0,
                 stuck_speed: float = 1.0,
                 stuck_seconds: float = 5.0,
                 reset_mode: str = "giveup",
                 giveup_button: str = "b",
                 giveup_hold_ms: float = 250.0,
                 giveup_timeout: float = 4.0,
                 giveup_settle_ms: float = 400.0,
                 giveup_retries: int = 3,
                 finish_button: str = "a",
                 w_progress: float = 1.0,
                 step_cost: float = 0.02,
                 soft_offset: float = 8.0,
                 w_soft: float = 0.01,
                 w_weave: float = 0.0,
                 w_reversal: float = 0.0,
                 finish_bonus: float = 100.0,
                 off_line_penalty: float = 20.0,
                 stuck_penalty: float = 10.0,
                 cp_radius: float = 20.0,
                 runs_dir: str | None = None,
                 capture_replays: bool = True,
                 profile: str = "",
                 instance: int = 0,
                 pad_addr: tuple = PAD_ADDR,
                 telem_addr: tuple = TELEM_ADDR,
                 slot: int | None = None,
                 rebuild_line: bool = False,
                 shared_config: bool = False):
        self.line = centerline
        # (left, right) per-line-point half-widths from the occupancy track
        # model, when one is built. Not in the observation yet - stored so the
        # edge-offset channel can be added without another map load.
        self._track_hw = None
        # Per-line-sample 1.0/0.0: does the track have barriers here.
        self._track_sides = None
        # Orientation basis of the previous control step, for omega.
        self._prev_basis = None
        self._route_jumps = []
        # Jump patches, resolved to (s_start, s_end) arc-length ranges on the
        # loaded line. Over one of these the car is MEANT to be airborne and off
        # the narrow line, so the off-surface / off-line / air penalties are
        # suppressed there, and the run-up gets a speed carrot.
        self._jump_s: list[tuple[float, float]] = []
        # Which game this env is driving. Everything that writes to a shared
        # location has to know, because N envs run in N processes against N
        # copies of the game and they all see the same filesystem.
        self.instance = int(instance)
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self.giveup_hold_ms = giveup_hold_ms
        self.giveup_timeout = giveup_timeout
        self.episodes = 0

        # Every tunable lives in a per-map JSON file that this re-reads while
        # training runs. The constructor arguments seed a config that does not
        # exist yet; once the file is on disk it wins, because that is what the
        # panel edits and what you expect to survive a restart.
        self.seed_values = {
            "episode": {"max_episode_s": max_episode_s},
            "reset": {"mode": reset_mode, "button": giveup_button,
                      "finish_button": finish_button,
                      "hold_ms": giveup_hold_ms, "settle_ms": giveup_settle_ms,
                      "retries": giveup_retries},
            "stuck": {"speed": stuck_speed, "seconds": stuck_seconds},
            "line": {"max_offset": max_offset, "soft_offset": soft_offset,
                     "w_soft": w_soft, "cp_radius": cp_radius},
            "reward": {"w_progress": w_progress, "step_cost": step_cost,
                       "finish_bonus": finish_bonus,
                       "off_line_penalty": off_line_penalty,
                       "stuck_penalty": stuck_penalty,
                       "w_weave": w_weave, "w_reversal": w_reversal},
            "enabled": {"weave": bool(w_weave or w_reversal)},
        }
        self.profile = profile
        self.cfg = TuningConfig(config_path(ROOT, None, profile))
        self._seed_config()
        self._apply_config()

        # Addresses are per-instance so several games can be driven at once;
        # instance N uses pad 8765+N and broker 8767+N by convention.
        self.telem = TelemetryLink(addr=telem_addr)
        self.pad = PadLink(addr=pad_addr)

        self.prev_action = np.zeros(3, dtype=np.float32)
        self.prev_steer = 0.0
        self.hint_pressed = False
        self.hint_started = 0
        self.hint_name = ""
        # Commanded vs applied input. The plugin reports what the GAME
        # received (in_steer/in_gas/in_brake), which is not necessarily what we
        # sent: TM2020 ramps analogue steering per FRAME, so anything that cuts
        # the frame rate - an unfocused window most of all - means the ramp
        # never reaches full deflection and the policy's action lands softened.
        # Nothing checked this before, and a fleet where each instance is
        # applied differently depending on which window has focus is a fleet
        # whose transitions do not mean the same thing.
        self.applied_ratio = 1.0
        self._fidelity: list[float] = []
        self._steady = 0
        self._fidelity_warned = False
        self.hint_action = [None, None, None]
        self.tapper = Tapper()
        self.sd_grade = "none"
        self.sd_score = 0.0
        self.prev_s = 0.0
        self.prev_pos: np.ndarray | None = None
        self.prev_speed = 0.0
        self._next_step_at: float | None = None
        self.overruns = 0
        # Snapshot at each episode reset, so a per-episode count is
        # available as well as the run total.
        self._overruns_at_reset = 0
        self.gear = 0
        self.gear_held = 0
        self.steps = 0
        self.slow_for = 0
        self.moved = False
        # Furthest progress when the stall timer was last reset.
        self._stall_ref_s = 0.0
        self._stalled_for = 0
        # Curriculum: end the episode at this GATE INDEX instead of the
        # finish. None = drive the whole lap. Set live by
        # train.curriculum.SectorCurriculum through VecEnv.set_attr, so a
        # phase change needs no restart and keeps the replay buffer.
        self.sector_exit = None
        # Distance along the line at which to end a sector attempt,
        # set by the curriculum from the exit gate minus the margin.
        self.sector_exit_s = None
        # Sector attempt ended cleanly (not a crash) -> reset by
        # RESPAWN rather than give-up, to land on the entry gate.
        self._sector_respawn = False
        self._sector_entry_t = None
        # Which materials this map has actually put under the wheels. The
        # enum has 81 entries and a track uses about six, so the tuning UI has
        # no way to know which rows matter without being told.
        self.seen_materials: set[str] = set()
        self._materials_written = 0

        # Static map knowledge, loaded once per map.
        # Which splitscreen seat this env drives, or None for a single-player
        # game. In splitscreen every env shares ONE game and one telemetry
        # connection but has its own pad, so the slot is what separates them.
        self.slot = slot
        # Explore mode can regenerate its line from any map's own landmarks,
        # so it can be pointed at a different track mid-run. A race line is a
        # recorded lap and cannot be, which is why this is off by default.
        self.rebuild_line = bool(rebuild_line)
        # Pin one tuning file for every map. Per-map configs are right when
        # you are tuning one track; across a rotation they mean the reward
        # changes under the policy every time the map does, and the replay
        # buffer ends up holding transitions scored several different ways
        # with nothing to distinguish them.
        self.shared_config = bool(shared_config)
        self._last_uid: str | None = None
        self.empty_records = 0
        self.restart_refusals = 0
        # How long to wait for a car to reappear before giving up. Longer than
        # the 20-minute splitscreen round timer plus a reload, so a rollover
        # costs one long gap rather than the whole run.
        self.lobby_wait = 300.0
        self.all_seats: list = []
        # Set when the terminating step already asked for the respawn, so
        # reset() waits rather than pressing again.
        self._reset_pressed_at: float | None = None
        self._reset_from_time: float = 0.0
        # Non-blocking respawn, so one seat's reset never stalls the others.
        self._respawning = False
        self._respawn_from = 0.0
        self._respawn_from_s = 0.0
        self._respawn_started = 0.0
        self._respawn_presses = 0
        self._respawn_warned = False
        self._respawn_stuck_warned = False
        self._forced_restart_at = 0.0
        self.start_s = 0.0
        self.max_s = 0.0
        # Where on the line we were last step, so the projection can be a local
        # search instead of a global one (see Centerline.project_near). None
        # means "no idea yet" and falls back to the global search.
        self._line_idx: int | None = None
        # How close to the line's start the car must be before an episode is
        # allowed to begin. Generous - this is guarding against a stale
        # position, not measuring anything.
        self.respawn_near_start = 90.0  # line origin sits ~70m ahead of the splitscreen spawn on this map; accept it
        self._last_obs = None
        self.not_ready_steps = 0
        self.gates: Gates | None = None
        self.cp_overrides: dict = {}
        # race_time at each checkpoint, so a section can be timed rather than
        # only a whole lap. This is what "optimise section by section" needs.
        self.splits: list[int] = []
        self.effects: EffectMap | None = None
        self.lidar: Lidar | None = None
        self.cp_taken: set[int] = set()
        # Markers already paid this episode (see _marker_reward).
        self._markers_paid: set[int] = set()
        # |metres| from the line at the last scored step.
        self._last_offset: float = 0.0
        self._prev_cp_dist: float | None = None
        self._cp_approach_paid = 0.0
        self._jump_taken_at: set = set()   # jump lips credited this episode
        self._green_streak = 0             # consecutive in-green speedslide steps
        self._steps_since_progress = 0
        self._dv_ema = 0.0
        # Furthest checkpoint index this run has ever reached. Drives the
        # growing episode cap (see _episode_cap_steps).
        # Rolling-window frontier for the growing episode cap: the furthest
        # checkpoint reached in the last N episodes. A one-off lucky CP does
        # NOT buy every future episode extra time forever - if the policy
        # regresses the window ages out and the cap shrinks back to base.
        self._recent_cp = collections.deque(maxlen=12)
        self.max_cp_ever = 0   # kept as an attr name _episode_cap_steps reads
        self.bad_surface_steps = 0
        self.time_cost = self.step_cost
        self.map_uid: str | None = None
        self.map_name: str = ""
        self._landmarks_for: str | None = object()  # never equal to a real uid

        # Per-episode bookkeeping for the run traces and the WHY log.
        self.runs_dir = runs_dir or os.path.join(ROOT, "runs")
        self.trace: list[list] = []
        self.ep_parts: dict[str, float] = {}
        self.best_time: float | None = None
        self.best_dist = 0.0
        self.total_steps = 0

        # Replay capture watches ONE directory - the game's Autosaves folder -
        # and every instance on this machine shares it. N watchers on one
        # directory would each claim every PB any of them set, so only
        # instance 0 captures.
        self.watcher = None
        if capture_replays and self.instance == 0 and ReplayWatcher is not None:
            self.watcher = ReplayWatcher(os.path.join(self.runs_dir, "replays"))
            if not self.watcher.start():
                self.watcher = None

        if spaces is not None:
            # Symmetric box; SAC's squashed Gaussian expects [-1, 1].
            self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,),
                                                dtype=np.float32)
        self.obs_dim = OBS_DIM

    # -- config -----------------------------------------------------------

    def _seed_config(self) -> None:
        """Write the constructor's values into a config that has no file yet.

        An existing file is left completely alone: it is the tuning you have
        already done, and having a CLI flag quietly overwrite it would make
        every panel edit look like it had been ignored.
        """
        if os.path.exists(self.cfg.path):
            return
        from .config import deep_merge
        self.cfg.data = deep_merge(self.cfg.data, self.seed_values)
        try:
            self.cfg.save()
        except OSError as ex:
            print(f"could not write {self.cfg.path}: {ex}", flush=True)

    def _episode_cap_steps(self) -> int:
        """Episode length cap, in control steps. Grows with the track.

        Two ways to set it, checked in this order:

          episode.cp_seconds  - an explicit list of CUMULATIVE times, one per
            checkpoint (drive a real lap, read the split clock at each gate).
            The cap once the run has reached CP n is cp_seconds[n] * cp_slack,
            so a car that is progressing but slow gets slack_x the human time
            to get there and is not truncated mid-section.

          episode.max_episode_s + episode.grant_per_cp - the simple linear
            fallback: a base for "no checkpoint yet", plus grant_per_cp
            seconds for every checkpoint the run has ever reached.

        Either way the result is clamped to episode.episode_ceiling.
        """
        c = self.cfg
        base = c.get("episode", "max_episode_s", 75.0)
        per_cp = c.get("episode", "grant_per_cp", 0.0)
        ceiling = c.get("episode", "episode_ceiling", base + 8 * max(per_cp, 20))
        # getattr: _apply_config() runs from __init__ before max_cp_ever is set.
        cp_ever = getattr(self, "max_cp_ever", 0)
        cp_seconds = c.get("episode", "cp_seconds", None)
        if isinstance(cp_seconds, (list, tuple)) and cp_seconds:
            slack = c.get("episode", "cp_slack", 3.0)
            # index 0 = time to CP1; clamp to the list; before any CP use the
            # first entry so the opening section still gets its slack.
            ref = cp_seconds[min(cp_ever, len(cp_seconds) - 1)]
            secs = max(base, ref * slack)
        else:
            secs = base + per_cp * cp_ever
        return int(min(ceiling, secs) * self.control_hz)

    def _apply_config(self) -> None:
        c = self.cfg
        self.max_steps = self._episode_cap_steps()
        self.reset_mode = c.get("reset", "mode", "giveup")
        self.giveup_button = c.get("reset", "button", "b")
        # Respawn returns the car to the LAST CHECKPOINT PASSED. That is the
        # only positioning primitive the game offers - there is no teleport,
        # and the plugin's `goto` switches maps, not positions.
        #
        # LAUNCHED respawn, not standstill: it puts the car back roughly two
        # seconds before that gate carrying the speed it had, rather than
        # stationary on the line. That is what makes a drilled sector
        # representative - a sector entered from a dead stop is a different
        # driving problem from one entered at racing speed, and optimising the
        # first would teach the wrong thing for the lap.
        #
        # Sector 0 is the exception and needs no special case: with no
        # checkpoint yet passed, respawn returns to the start line, standing -
        # which is exactly how a lap begins.
        self.respawn_button = c.get("reset", "respawn_button", "y")
        self.respawn_hold_ms = float(c.get("reset", "respawn_hold_ms", 120.0))
        # How far SHORT of the exit gate a sector attempt stops. The whole
        # trick: cross the gate and respawn takes you to IT, so the sector
        # cannot be repeated. Stop short and respawn takes you back to the
        # sector's entry gate instead, so it can be drilled over and over from
        # a consistent standing start. Must exceed one step of travel - 2m at
        # 40m/s and 20Hz - or a fast car overshoots the check.
        self.sector_stop_short_m = float(
            c.get("stuck", "sector_stop_short_m", 12.0))
        self.finish_button = c.get("reset", "finish_button", "a")
        # How long each reset button is held. In 4-way splitscreen a 250ms tap
        # lands on a frame where that viewport is not the focused input often
        # enough to matter - a longer hold is caught far more reliably.
        self.giveup_hold_ms = c.get("reset", "hold_ms", 250.0)
        self.giveup_settle_ms = c.get("reset", "settle_ms", 400.0)
        self.giveup_retries = int(c.get("reset", "retries", 4))
        self.finish_fallback_after = int(c.get("reset", "finish_fallback_after", 2))
        # A splitscreen seat that cannot give-up-respawn after this many seconds
        # escalates to a full RequestRestartMap. That resets EVERY seat, but a
        # seat wedged at the start for a minute poisons far more data than a
        # clean restart of everyone costs. 0 disables the escalation (old
        # behaviour: hold neutral and keep mashing give-up).
        self.restart_after_s = float(c.get("reset", "restart_after_s", 12.0))
        self.quick_timeout = c.get("reset", "quick_timeout", 1.5)
        self.finish_timeout = c.get("reset", "finish_timeout", 12.0)
        self.skip_intro = bool(c.get("reset", "skip_intro", True))
        self.skip_button = c.get("reset", "skip_button", "a")
        self.skip_interval = c.get("reset", "skip_interval_ms", 250.0) / 1000.0
        self.stuck_speed = c.get("stuck", "speed", 1.0)
        # Seconds of no forward progress before an episode is called stuck,
        # whatever the speed. 0 disables. See the wedge note in the reward.
        self.no_progress_m = float(c.get("stuck", "no_progress_m", 2.0))
        # Absolute lap position is a track-MEMORISATION input. Switch it off
        # for general training and it feeds a constant 0 instead of being
        # removed: the observation keeps its width, so every model stays
        # loadable in either mode and no migration is ever needed. Dropping
        # the dimension would need one, and would strand a policy that had
        # learned to lean on it.
        self.use_lap_progress = bool(c.get("line", "use_lap_progress", True))
        _nps = float(c.get("stuck", "no_progress_s", 6.0))
        self.no_progress_steps = int(_nps * self.control_hz) if _nps > 0 else 0
        self.stuck_steps = int(c.get("stuck", "seconds", 5.0) * self.control_hz)
        self.max_offset = c.get("line", "max_offset", 30.0)
        self.soft_offset = c.get("line", "soft_offset", 8.0)
        self.w_soft = c.get("line", "w_soft", 0.01)
        self.cp_radius = c.get("line", "cp_radius", 20.0)
        self.cp_strict_order = bool(c.get("line", "cp_strict_order", False))
        r = c.data.get("reward", {})
        self.w_progress = r.get("w_progress", 1.0)
        # Extra pay per metre on unbarriered (platform) sections.
        self.w_platform = r.get("w_platform", 0.0)
        # Hand-placed one-off bonuses at distances along the line. See
        # _marker_reward. Per-track, in configs/<map>.explore.json:
        #   "markers": [{"s": 1420, "bonus": 150}, {"s": 1480, "bonus": 150}]
        raw_markers = self.cfg.data.get("markers") or []
        self.markers = []
        for m in raw_markers:
            try:
                # max_offset: optional LATERAL limit, metres from the line.
                # Without one a marker has no width at all - it is a scalar
                # test on progress - so a car that gains arc length by leaving
                # the track in roughly the right direction collects it without
                # ever crossing the spot. None keeps the old behaviour.
                lim = m.get("max_offset")
                self.markers.append((float(m["s"]),
                                     float(m.get("bonus", 100.0)),
                                     None if lim is None else float(lim)))
            except (TypeError, ValueError, KeyError):
                continue
        self.markers.sort()
        self.step_cost = r.get("step_cost", 0.02)
        self.finish_bonus = r.get("finish_bonus", 100.0)
        self.cp_bonus = r.get("cp_bonus", 0.0)
        # Dense, line-free shaping: reward per metre the car closes on the next
        # uncrossed gate (straight-line to its centre; the finish once every
        # checkpoint is taken). Potential-based, so it does not change the
        # optimal policy, only how fast the value function finds it. 0 = off.
        # Use this instead of w_progress on a map with no recorded lap, where
        # the provisional reference line is unreliable.
        self.w_cp_approach = r.get("w_cp_approach", 0.0)
        self.off_line_penalty = r.get("off_line_penalty", 20.0)
        self.stuck_penalty = r.get("stuck_penalty", 10.0)
        self.w_weave = r.get("w_weave", 0.0)
        self.w_reversal = r.get("w_reversal", 0.0)
        self.w_turbo_use = r.get("w_turbo_use", 0.0)
        self.w_air = r.get("w_air", 0.0)
        # Jump run-up speed carrot. A jump patch already stops the penalties
        # from firing over the gap (see _jump_s), but "carry speed into the
        # take-off" is only learned by trial and error unless it is paid for.
        # 0 = off (the default): still discovered the hard way. Set > 0 to pay
        # for approach speed in the jump_approach_m before each take-off, up to
        # jump_target_kmh.
        self.w_jump_speed = r.get("w_jump_speed", 0.0)
        self.jump_approach_m = r.get("jump_approach_m", 40.0)
        self.jump_target_kmh = r.get("jump_target_kmh", 200.0)
        self.par_speed = r.get("par_speed", 0.0)
        self.charge_unused_time = bool(r.get("charge_unused_time", True))
        self.w_gear = r.get("w_gear", 0.0)
        self.gear_hold_steps = int(r.get("gear_hold_steps", 8))
        self.top_gear = max(1, int(r.get("top_gear", 5)))
        self.w_downshift = r.get("w_downshift", 0.0)
        self.w_gas = r.get("w_gas", 0.0)
        self.w_accel = r.get("w_accel", 0.0)
        self.w_both_pedals = r.get("w_both_pedals", 0.0)
        self.surface_grace = int(c.data.get("surface_grace_steps", 4))

        ss = c.data.get("speedslide", {})
        self.ss_w = ss.get("w", 0.0)              # per-step reward at full GREEN
        self.ss_w_blue = ss.get("w_blue", 0.0)
        self.ss_floor = ss.get("speed_floor_kmh", 0.0)
        # Any real slide (orange up) pays this floor; the rest ramps to ss_w
        # across the band, shaped by ss_gamma so the green end dominates
        # (gamma>1 = convex: yellow is worth little, green is worth a lot).
        self.ss_w_any = ss.get("w_any", 0.0)
        self.ss_gamma = ss.get("gamma", 2.5)
        # STAYING in green: a multiplier that grows with the consecutive-green
        # step count, log so it climbs forever but never runs away.
        self.ss_streak_gain = ss.get("streak_gain", 0.6)
        self.ss_streak_cap = ss.get("streak_cap", 4.0)
        # Slide reward pays nothing after this many steps with no new ground
        # (20Hz -> 15 steps ~= 0.75s). Kills slide-brake-slide farming.
        self.ss_stall_steps = int(ss.get("stall_steps", 15))
        # A real speedslide gains speed. Scale the reward by smoothed dv:
        # accel_floor when flat, 0 when decelerating, up to 1 when pulling.
        self.ss_accel_floor = ss.get("accel_floor", 0.15)
        self.ss_accel_gain = ss.get("accel_gain", 3.0)

        self.hints = parse_hints(c.data.get("hints"))
        self.marks = c.data.get("marks") or {}
        if self.hints and getattr(self, "line", None) is not None:
            missing = resolve_marks(self.hints, self.marks, self.line)
            if missing:
                print(f"  hints reference unknown marks: "
                      f"{', '.join(sorted(set(missing)))} - record them with "
                      f"tools/mark.py or the panel's 'Mark here'", flush=True)
        if not hasattr(self, "tapper"):
            self.tapper = Tapper()

        a = c.data.get("action", {})
        self.binary_gas = bool(a.get("binary_gas", True))
        self.binary_brake = bool(a.get("binary_brake", True))
        self.gas_threshold = a.get("gas_threshold", 0.0)
        self.brake_threshold = a.get("brake_threshold", 0.0)
        # A pedal threshold at or above 1.0 can NEVER be met: the action is
        # clipped to [-1, 1] before the comparison, so the pedal is not "rare",
        # it is disconnected - and nothing downstream says so, because a policy
        # that never brakes looks exactly like a policy that has chosen not to.
        #
        # This cost most of a day. brake_threshold sat at 1.2 on this map's
        # explore config, so for ~7 hours the cars had no brake AND no reverse
        # (TM2020's brake is reverse when stationary). They could not slow for
        # a corner and could not back off a wall once wedged, which is exactly
        # what the logs showed: every episode ending 'stuck', and only 7% of
        # wedges ever recovering.
        for label, thr, on in (("brake", self.brake_threshold, self.binary_brake),
                               ("gas", self.gas_threshold, self.binary_gas)):
            if on and thr >= 1.0:
                print(f"  !! action.{label}_threshold is {thr}, but the action "
                      f"is clipped to [-1, 1] - the {label} can never be "
                      f"applied. Set it below 1.0 (0.0 is the usual value).",
                      flush=True)
        # Steering slew-rate limit, full-deflection units per second. 0 (the
        # default) = snap: hand the policy's target straight to the pad, like a
        # keyboard key snapping to full. TM2020 applies its own wheel-angle
        # ramp internally whatever the input, so the car is never jerked, and
        # "the game only applied 30%" is just that internal ramp caught mid-way
        # by a metric sampling faster than SAC holds an action. > 0 only makes
        # the POLICY smoother (and slower to turn in than a keyboard player).
        # How fast the commanded steering may move, units/s (lock-to-lock is
        # 2.0). 0 = no limit.
        #
        # This is a PER-STAGE setting, not a schedule. The explorer's job is to
        # find the route, and it does that better with a steady wheel: on the
        # continuous action space its steering reversed 41% of consecutive
        # control steps, sawing 8 times a second, which averages out to a car
        # that barely turns. The racer is the one that needs instant
        # lock-to-lock flicks, because some TM technique depends on them - so
        # <uid>.explore.json caps it and <uid>.race.json sets 0.
        #
        # KNOW THIS about the handover: the racer is `--init-from` the explore
        # model, so it INHERITS the explorer's weights (actor + critics; the
        # buffer is deliberately left empty). It therefore inherits a policy
        # that learned while its steering output was a TARGET the env ramped
        # toward, and then runs with that ramp removed - the same command now
        # lands instantly, so expect it to oversteer until it adapts. That is
        # accepted: the mismatch is in the forgiving direction (more authority,
        # not less), the empty buffer refills entirely under the new regime
        # from step one, and the handover already changes the reward, the line
        # and the offsets, so the policy is re-adapting regardless. If a
        # handover ever visibly collapses into oversteer, the fix is to ramp
        # steer_rate off over the early race stage rather than drop it at once.
        self.steer_rate = float(a.get("steer_rate", 0.0))
        # Notches per side: 1 -> {-1, 0, +1} (pure keyboard), 2 -> halves,
        # 10 -> every 10%, like ten steering keys. 0 keeps it continuous.
        self.steer_levels = int(a.get("steer_levels", 0) or 0)

        self.cp_mode = c.get("line", "cp_mode", "gate")
        self.cp_half_width = c.get("line", "cp_half_width", 18.0)
        self.cp_height = c.get("line", "cp_height", 10.0)
        # Checkpoints are not all one size, and the game tells us nothing
        # about their dimensions - a landmark is a position, a tag and an
        # order, and a waypoint is two booleans. So the global above is the
        # default and this names the exceptions, by ordered gate index.
        f = c.data.get("input_fidelity", {})
        self.fidelity_window = int(f.get("window", 40))
        self.fidelity_settle = int(f.get("settle_steps", 3))
        self.fidelity_floor = float(f.get("warn_below", 0.95))
        self.cp_overrides = c.data.get("line", {}).get("cp_overrides") or {}
        # getattr, not self.gates: _apply_config() runs from __init__ before
        # the attribute exists, and again on every hot reload once it does.
        gates = getattr(self, "gates", None)
        if gates is not None:
            gates.set_sizes(self.cp_overrides)

    def _switch_config(self, uid: str | None) -> None:
        """Move to this map's config file, creating it from the current values
        the first time the map is seen so there is something to edit."""
        path = config_path(ROOT, None if self.shared_config else uid,
                           self.profile)
        if path == self.cfg.path:
            return
        existed = os.path.exists(path)
        self.cfg = TuningConfig(path)
        if not existed:
            self._seed_config()
        self._apply_config()
        print(f"config: {os.path.relpath(path, ROOT)}"
              f"{'' if existed else ' (created from current settings)'}",
              flush=True)

    # -- helpers ----------------------------------------------------------

    def _decode(self, action) -> tuple[float, float, float]:
        """Three network outputs in [-1, 1] -> steering and two pedals.

        The pedals threshold to on/off by default. TM2020's throttle is a
        switch in practice - the fast line is nearly always fully on or fully
        off - so a continuous pedal hands the optimiser a whole continuum of
        options that are all slower than the two ends, and 0.43 gas looks to
        the critic much like 0.47. Keeping the ACTION SPACE continuous while
        making the PEDAL binary is deliberate: SAC's squashed Gaussian needs
        a continuous space, but the car only ever does one of two things.
        """
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        steer = float(a[0])
        # Snap the steering to a few fixed notches.
        #
        # Same argument as the binary pedals, and measured: the continuous
        # output was changing sign between 41% of consecutive control steps
        # and swinging more than lock-to-lock on 22% of them - the car was
        # sawing at the wheel 8 times a second, which averages out to barely
        # turning at all however hard it "wants" to. 0.43 and 0.47 of steering
        # look the same to the critic but the difference between them, applied
        # 20 times a second, is noise the car has to drive through.
        #
        # Notches give the policy a small set of things it can mean. Combined
        # with steer_rate below this is exactly how a keyboard drives TM2020 -
        # a discrete target plus a ramp toward it - and keyboard players are
        # competitive on this game, so it is known to be enough control.
        if self.steer_levels > 0:
            q = float(self.steer_levels)
            steer = float(np.round(steer * q) / q)
        if self.binary_gas:
            gas = 1.0 if a[1] > self.gas_threshold else 0.0
        else:
            gas = float((a[1] + 1.0) * 0.5)
        if self.binary_brake:
            brake = 1.0 if a[2] > self.brake_threshold else 0.0
        else:
            brake = float((a[2] + 1.0) * 0.5)
        return steer, gas, brake

    def _wait_for(self, predicate, timeout: float, poll: float = 0.02):
        deadline = time.time() + timeout
        while time.time() < deadline:
            rec = self._slot_view(self.telem.get())
            if rec is not None and predicate(rec):
                return rec
            time.sleep(poll)
        return None

    def _wait_for_respawn(self, prev_time: float, timeout: float):
        """Wait for the car to be back on the start line, skipping the intro.

        Two things beyond a plain wait:

        * The intro fly-in is skippable with any button, so it gets mashed
          rather than sat through. That is ~8s per reset on the paths that
          replay it, which on a 25s lap is a third of all training wall-clock.
        * Success requires being out of the Intro sequence, not just a low
          race clock. The clock reads ~0 during the fly-in, so testing the
          clock alone would hand back control before the car can be steered
          and burn the first second of the episode on inputs that go nowhere.
        """
        deadline = time.time() + timeout
        floor = max(prev_time - 500, 3000)
        last_skip = 0.0
        while time.time() < deadline:
            rec = self._slot_view(self.telem.get())
            if rec is not None:
                ui = rec.get("ui", UI_NONE)
                if (rec.get("car") and ui not in (UI_INTRO, UI_OUTRO)
                        and (rec.get("race_time") or 0) < floor):
                    return rec
                now = time.time()
                if (self.skip_intro and ui in (UI_INTRO, UI_OUTRO)
                        and now - last_skip > self.skip_interval):
                    last_skip = now
                    self.pad.press(self.skip_button, 60.0)
            time.sleep(0.02)
        return None

    def _wait_for_car(self, timeout: float = 15.0) -> dict | None:
        """Latest record in which THIS seat actually has a car.

        Taking whatever arrived is wrong in splitscreen, and quietly so. The
        plugin emits a heartbeat whenever the viewed vehicle state is null -
        common between rounds and in menus - and that heartbeat carries no
        position and no map uid. Reading it gives a car at the origin, which
        then measures as exactly the map's X offset from the reference line
        and fails as "the line belongs to a different map". It does not; the
        record was simply empty.

        Seats also come up at different times, so seat 3 having no car yet
        while seat 0 is driving is normal rather than an error.
        """
        deadline = time.time() + timeout
        rec = None
        while time.time() < deadline:
            rec = self._slot_view(self.telem.get())
            if rec and rec.get("car") and _real_pos(rec.get("pos")):
                return rec
            time.sleep(0.05)
        return rec

    def _slot_view(self, rec: dict | None) -> dict | None:
        """This env's car, out of a splitscreen record.

        The plugin emits a `players` array, one entry per local seat, and the
        top-level fields describe whichever car the camera is on. Merging the
        seat over the top means everything downstream - observation, reward,
        checkpoints, traces - keeps reading the same field names and never has
        to know which mode the game is in.

        Two fields need care rather than a straight copy:

        `finished` is only meaningful per seat, and the top-level one belongs
        to the viewed car. Each terminal reports its own UI sequence, so a seat
        is finished when *its* UI says Finish - taking the shared flag would
        end all four episodes whenever any one car crossed the line.

        The effects block (turbo, reactor, icing, wetness) has no per-seat
        equivalent, so it is left as the viewed car's. Wrong for the other
        seats, and knowingly so: those inputs are small, the alternative is no
        data at all, and the plugin change to fix it properly is a bigger one
        than this.
        """
        if rec is None:
            return rec
        players = rec.get("players")
        if not players:
            return rec
        if self.slot is None:
            # A single-seat env pointed at a splitscreen game. The top-level
            # record describes the camera's car, which between seats is
            # nothing at all - so without this the env waits fifteen seconds
            # and reports "no car" about a game with four of them. Take the
            # first seat that has one.
            seat = next((p for p in players if p.get("car")), None)
            if seat is None or rec.get("car"):
                return rec
            merged = dict(rec)
            merged.update({k: v for k, v in seat.items() if k != "slot"})
            merged["finished"] = int(seat.get("ui", 0)) == UI_FINISH
            return merged
        # Kept so this seat can see the others. They share one race, and a
        # seat that ignores that ends up driving on while another is resetting.
        self.all_seats = players
        seat = next((p for p in players if p.get("slot") == self.slot), None)
        if seat is None or not seat.get("car"):
            # The seat exists but has no car yet - report "no car" rather than
            # silently handing back the viewed player's, which would train this
            # env on somebody else's driving.
            merged = dict(rec)
            merged["car"] = False
            return merged
        merged = dict(rec)
        merged.update({k: v for k, v in seat.items() if k != "slot"})
        merged["finished"] = int(seat.get("ui", 0)) == UI_FINISH
        return merged

    def _observe(self, rec: dict) -> tuple[np.ndarray, float, float]:
        pos = np.asarray(rec.get("pos", [0, 0, 0]), dtype=np.float64)
        vel = np.asarray(rec.get("vel", [0, 0, 0]), dtype=np.float64)
        dir_ = np.asarray(rec.get("dir", [1, 0, 0]), dtype=np.float64)
        up = np.asarray(rec.get("up", [0, 1, 0]), dtype=np.float64)
        left = np.asarray(rec.get("left", [0, 0, 1]), dtype=np.float64)

        moved = (float(np.linalg.norm(pos - self.prev_pos))
                 if self.prev_pos is not None else None)
        idx, s, offset = self.line.project_near(pos, self._line_idx, moved)
        self._line_idx = idx
        ahead = self.line.lookahead(idx, LOOKAHEAD)

        # Everything spatial in the car's own frame - absolute world
        # coordinates would just teach it this one track's geography.
        local = np.concatenate([
            car_frame(p - pos, dir_, up, left) / 50.0 for p in ahead
        ])
        vel_local = car_frame(vel, dir_, up, left) / 100.0
        world_up_local = car_frame(np.array([0.0, 1.0, 0.0]), dir_, up, left)

        obs = np.concatenate([
            [rec.get("speed", 0.0) / 100.0],
            vel_local,
            world_up_local,
            local,
            [offset / 30.0],
            np.asarray(rec.get("slip", [0, 0, 0, 0]), dtype=np.float64),
            [rec.get("adherence", 1.0)],
            [1.0 if rec.get("ground") else 0.0],
            [rec.get("gear", 0) / 5.0],
            [rec.get("rpm", 0.0) / 12000.0],
            self.prev_action,
            self._surface_obs(rec),
            self._effect_obs(rec, s),
            self._lidar_obs(pos, dir_, left),
            self._border_obs(rec),
            self._sides_ahead_obs(idx),
            self._air_obs(rec),
            self._omega_obs(dir_, up, left),
            self._chassis_obs(rec),
            # Appended last, so every index above keeps the value it had at
            # 141 dims and a 141-wide model migrates by zero-filling these.
            [self._margin_obs(idx, offset)],
            self._width_ahead_obs(idx),
            [self._dist_to_stop_obs()],
            [self._lap_progress_obs()],
        ]).astype(np.float32)
        return obs, s, offset

    @staticmethod
    def _chassis_obs(rec: dict) -> np.ndarray:
        """Suspension, real wheel angle, skidding, and on-its-roof.

        All four measured to vary over real driving. `adherence`, `brake_coef`,
        `sim_coef` and `wear` were measured CONSTANT and are left out - an
        input that never changes is parameters to train for no information.
        (`adher` is already in the observation above and is one of those: it
        read exactly 1.0 for all 6359 samples. Left in place because removing
        it would shift every index after it.)
        """
        d = rec.get("damper") or [0.0, 0.0, 0.0, 0.0]
        d = list(d)[:4] + [0.0] * max(0, 4 - len(d))
        return np.array([
            min(float(d[0]), 0.4) / 0.4, min(float(d[1]), 0.4) / 0.4,
            min(float(d[2]), 0.4) / 0.4, min(float(d[3]), 0.4) / 0.4,
            float(rec.get("steer_angle") or 0.0),
            float(rec.get("skidding") or 0.0) / 4.0,
            1.0 if rec.get("top_contact") else 0.0,
        ])

    @staticmethod
    def _air_obs(rec: dict) -> np.ndarray:
        """[height above whatever is below, airtime so far], both scaled.

        Straight from the telemetry, which has been sending them all along.
        `ground_dist` is the one that matters: timing anything in the air is a
        question of how long you have left, and the policy had no way to know.
        Scales chosen so ordinary driving sits near 0 and a big jump saturates
        rather than dominating the input.
        """
        gd = rec.get("ground_dist")
        fl = rec.get("flying")
        gd = 0.0 if gd is None else float(gd)
        fl = 0.0 if fl is None else float(fl)
        # `flying` is MILLISECONDS. Measured: 0 on the ground, 3990 for a
        # four-second jump. Dividing by 3 as if it were seconds would peg
        # this input at its ceiling on every hop.
        return np.array([min(gd, 20.0) / 20.0,
                         min(fl / 1000.0, 4.0) / 4.0])

    def _omega_obs(self, dir_, up, left) -> np.ndarray:
        """Angular velocity in the car's own frame, [pitch, yaw, roll] rad/s.

        Not in the telemetry, so it is differenced from the orientation basis
        across control steps: R = [dir, up, left] each frame, and the rotation
        between two frames gives the angular velocity. The antisymmetric part
        of R_prev^T @ R_now is the small-angle rotation vector.

        This is the input that makes the user's air technique learnable at all:
        a brake tap kills PITCH and countersteering kills YAW, and both are
        rate control. The observation had `up` (attitude) and nothing about how
        fast that attitude was changing, so no reward could have taught either.

        Returns zeros on the first step of an episode, which is honest - there
        is no previous frame to difference against.
        """
        R = np.stack([np.asarray(dir_, dtype=np.float64),
                      np.asarray(up, dtype=np.float64),
                      np.asarray(left, dtype=np.float64)], axis=1)
        prev = self._prev_basis
        self._prev_basis = R
        if prev is None or self.dt <= 0:
            return np.zeros(3)
        try:
            dR = prev.T @ R
            skew = (dR - dR.T) * 0.5
            omega = np.array([skew[2, 1], skew[0, 2], skew[1, 0]]) / self.dt
        except Exception:                                 # noqa: BLE001
            return np.zeros(3)
        return np.clip(omega / 5.0, -3.0, 3.0)

    @staticmethod
    def _border_obs(rec: dict) -> np.ndarray:
        """1.0 per wheel that is on a track BORDER (the rubber kerb).

        Separate from the grip groups on purpose. `Rubber` sits in the `road`
        grip class - correctly, a kerb is grippy - which meant the 36-dim
        surface block could not distinguish a kerb from the racing line, and the
        one fine-grained "you are at the edge" signal the game gives was being
        discarded one step before the policy saw it.
        """
        mats = rec.get("mat") or []
        out = np.zeros(4)
        for i, m in enumerate(mats[:4]):
            if is_border(m):
                out[i] = 1.0
        return out

    def _sides_ahead_obs(self, idx: int) -> np.ndarray:
        """Does the track have BARRIERS at each lookahead distance?

        1.0 = barriered, 0.0 = drive straight off. Unknown reads 1.0, which is
        the safe direction: a car that believes there is a wall drives more
        carefully than one that believes there is not.

        This is the input the training track was missing. All seven of its
        edgeless blocks sit between checkpoint 4 and 5, and the policy reaches
        CP4 in 78% of episodes and CP5 in none - it has no way to know the
        walls stop. Sampled at the same distances as `ahead`, so it is
        anticipatory rather than a report of what is already under the wheels.
        """
        n = len(LOOKAHEAD)
        if self._track_sides is None or self.line is None:
            return np.ones(n)
        try:
            s_here = float(self.line.s[idx])
            targets = s_here + np.asarray(LOOKAHEAD, dtype=np.float64)
            return np.interp(targets, self.line.s,
                             np.asarray(self._track_sides, dtype=np.float64))
        except Exception:                                 # noqa: BLE001
            return np.ones(n)

    #: Half-width treated as "wide open" when normalising width_ahead. Roads
    #: on this game's tech blocks run ~12m half-width and platforms ~8m, so 16
    #: puts the interesting range in the middle of 0..1 rather than squashed
    #: against the top.
    WIDE_M = 16.0

    #: `sides` value the route model gives a barriered road piece.
    BARRIERED_SIDES = 0.5

    def _margin_obs(self, idx: int, offset: float) -> float:
        """How much road is left between the car and the edge, 1 = centred.

        `offset` alone cannot answer this. Five metres off the line is
        comfortable on a 12m-half-width road and one car-width from a fall on
        an 8m platform, and until now nothing in the observation distinguished
        them - `_track_hw` was loaded from the route model and read by nothing.

        Normalised BY THE LOCAL WIDTH deliberately, so the number means the
        same thing everywhere: 1.0 is the middle of whatever you are on, 0.0 is
        its edge. A policy can then learn one rule ("keep margin up") instead
        of a different safe-offset constant per block type.
        """
        if self._track_hw is None or self.line is None:
            return 1.0
        try:
            hw = float(np.interp(float(self.line.s[idx]), self.line.s,
                                 np.asarray(self._track_hw, dtype=np.float64)))
            if hw <= 0.1:
                return 1.0
            return float(np.clip(1.0 - abs(offset) / hw, 0.0, 1.0))
        except Exception:                                 # noqa: BLE001
            return 1.0

    def _width_ahead_obs(self, idx: int) -> np.ndarray:
        """Half-width at each lookahead distance, normalised by WIDE_M.

        Anticipatory for the same reason sides_ahead is: a road narrowing into
        a platform has to be visible BEFORE the car is on it, because by the
        time the margin at the car has dropped there is no room left to act.

        Unknown reads 1.0 (wide) rather than 0.0. A model that thinks a
        platform is a motorway drives off it; one that thinks a motorway is a
        platform merely drives timidly, and that is the recoverable error.
        """
        n = len(LOOKAHEAD)
        if self._track_hw is None or self.line is None:
            return np.ones(n)
        try:
            s_here = float(self.line.s[idx])
            targets = s_here + np.asarray(LOOKAHEAD, dtype=np.float64)
            hw = np.interp(targets, self.line.s,
                           np.asarray(self._track_hw, dtype=np.float64))
            return np.clip(hw / self.WIDE_M, 0.0, 1.0)
        except Exception:                                 # noqa: BLE001
            return np.ones(n)

    def _marker_reward(self) -> float:
        """One-off bonuses at hand-picked distances along the line.

        WHY, when `progress` already pays per metre: the problem this solves is
        not signal density, it is CREDIT-ASSIGNMENT DISTANCE. Measured on this
        track, half the episodes that reach CP4 stop dead at the entrance to a
        194m unbarriered run and refuse to enter. For entering to look
        worthwhile, the critic has to carry CP5's bonus backwards across all
        194m - and every sample it has of that stretch ends badly. A marker
        halfway across is a nearer, already-achievable target, so the value has
        half as far to propagate.

        Paid once per episode, and keyed on `max_s` rather than the current
        position, so it cannot be farmed by reversing over the same spot.

        A PER-TRACK TOOL. Markers are hand-placed scaffolding for one hard
        section; leave the list empty on any map that does not need it, and
        remove them once the section is learned.
        """
        if not self.markers:
            return 0.0
        total = 0.0
        for i, (at_s, bonus, lim) in enumerate(self.markers):
            if i in self._markers_paid:
                continue
            if lim is not None and abs(self._last_offset) > lim:
                continue
            if self.max_s >= at_s:
                self._markers_paid.add(i)
                total += bonus
                print(f"    marker {i} at {at_s:.0f}m reached (+{bonus:.0f})",
                      flush=True)
        return total

    def _lap_progress_obs(self) -> float:
        """How far round the lap the car is, 0 at the start and 1 at the end.

        Uses max_s, the furthest point reached, so it never runs backwards
        when the car does - the policy is being told where on the track it is,
        not how much ground it just lost.
        """
        if not self.use_lap_progress:
            return 0.0
        try:
            if self.line is None:
                return 0.0
            total = float(self.line.length) or 1.0
            return float(np.clip(self.max_s / total, 0.0, 1.0))
        except Exception:                                  # noqa: BLE001
            return 0.0

    def _dist_to_stop_obs(self) -> float:
        """Fraction of the line still to run before the episode ends.

        1.0 at the start, 0.0 at the stop line. In sector mode that is the
        sector's stop; otherwise it is the end of the line.

        Normalised by the WHOLE line rather than by the sector, so the number
        means the same thing in every phase - a sector-relative version would
        rescale under the policy every time the curriculum advanced.
        """
        try:
            if self.line is None:
                return 1.0
            end = self.sector_exit_s if self.sector_exit_s else self.line.length
            total = float(self.line.length) or 1.0
            return float(np.clip((end - self.max_s) / total, 0.0, 1.0))
        except Exception:                                  # noqa: BLE001
            return 1.0

    def _compute_sector_stop(self) -> None:
        """Where to stop, in metres along the line, for the current sector.

        Derived here rather than passed in: the env is the only thing that has
        both the line and the gates, and the gates are not known until the map
        loads. The curriculum sets a gate INDEX and this turns it into a
        distance.

        Stops `sector_stop_short_m` before the gate so respawn returns the car
        to the sector's ENTRY, not its exit.
        """
        try:
            if self.line is None or not self.gates:
                return
            idx = self.sector_exit - 1          # gate index of the exit
            centres = list(getattr(self.gates, "centres", []) or [])
            if idx >= len(centres):
                # The last sector's exit is the FINISH, which is not a gate in
                # `centres` - the plugin reports it as a flag. Its stop line is
                # the end of the reference line.
                self.sector_exit_s = max(1.0, float(self.line.length)
                                         - self.sector_stop_short_m)
                print(f"  sector {self.sector_exit} (to the finish): stop at "
                      f"{self.sector_exit_s:.0f}m of {self.line.length:.0f}m",
                      flush=True)
                return
            if idx < 0:
                return
            _, s_gate, _ = self.line.project_near(
                np.asarray(centres[idx], dtype=np.float64), None)
            stop = float(s_gate) - self.sector_stop_short_m
            self.sector_exit_s = max(1.0, stop)
            print(f"  sector {self.sector_exit}: stop at {self.sector_exit_s:.0f}m "
                  f"({self.sector_stop_short_m:.0f}m short of the gate at "
                  f"{s_gate:.0f}m)", flush=True)
        except Exception as ex:                            # noqa: BLE001
            print(f"  could not place the sector stop line ({ex})", flush=True)

    def _platform_factor(self) -> float:
        """1.0 on an unbarriered section, 0.0 on barriered road.

        Platform pieces report the SAME material as road (Asphalt/Concrete), so
        nothing material-keyed can tell them apart. The route model's
        containment can: 0.0 = no sides, 0.5 = barriered. That is the only
        signal available, and it is the same one `sides_ahead` feeds the policy.
        """
        if self._track_sides is None or self.line is None:
            return 0.0
        try:
            idx = self._line_idx
            if idx is None:
                return 0.0
            sides = float(np.interp(float(self.line.s[idx]), self.line.s,
                                    np.asarray(self._track_sides,
                                               dtype=np.float64)))
            return float(np.clip(1.0 - sides / self.BARRIERED_SIDES, 0.0, 1.0))
        except Exception:                                 # noqa: BLE001
            return 0.0

    def _lidar_obs(self, pos, dir_, left) -> np.ndarray:
        """How far the ground extends in each direction, in the car's frame.

        1.0 means "nothing within range", which is also what an unscanned map
        reports for every beam - so the observation keeps its shape whether or
        not the occupancy dump succeeded.
        """
        if self.lidar is None:
            return np.ones(len(BEAM_ANGLES_DEG))
        try:
            return self.lidar.normalised(pos, dir_, left)
        except Exception:
            return np.ones(len(BEAM_ANGLES_DEG))

    @staticmethod
    def _surface_obs(rec: dict) -> np.ndarray:
        """What each tyre is actually on.

        A six-class grip group per wheel rather than a raw material one-hot:
        there are 81 materials, a track uses about six of them, and grip class
        is the part that transfers to a map with a different palette.

        Per-wheel rather than one value for the car, because two wheels on
        tarmac and two on grass is a completely different situation from four
        on either, and it is the one that spins you.
        """
        mats = rec.get("mat") or [0, 0, 0, 0]
        onehot = np.zeros(4 * N_GROUPS, dtype=np.float64)
        for w in range(4):
            if w < len(mats):
                onehot[w * N_GROUPS + group_index(mats[w])] = 1.0
        return np.concatenate([
            onehot,
            np.asarray(rec.get("icing") or [0, 0, 0, 0], dtype=np.float64)[:4],
            np.asarray(rec.get("dirt") or [0, 0, 0, 0], dtype=np.float64)[:4],
            [rec.get("wetness", 0.0)],
            [rec.get("water", 0.0)],
        ])

    def _effect_obs(self, rec: dict, s: float) -> np.ndarray:
        """Effects acting on the car now, and effects coming up.

        The "coming up" half is the point. Reacting to a booster once you are
        on it is too late to have chosen a line into it; distance-to-next-boost
        is what lets the policy set up for one.
        """
        turbo = np.asarray([
            1.0 if rec.get("turbo") else 0.0,
            rec.get("turbo_time", 0.0),
            rec.get("turbo_lvl", 0) / 5.0,
        ])
        rtype = np.zeros(3)
        # 0 none, 1 up, 2 down - one-hot because they are not on a scale.
        rtype[min(int(rec.get("reactor_type", 0) or 0), 2)] = 1.0
        reactor = np.concatenate([
            [(rec.get("reactor_lvl", 0) or 0) / 3.0],
            rtype,
            [rec.get("reactor_timer", 0.0) or 0.0],
        ])

        if self.effects is not None and self.cfg.enabled("effects"):
            ahead = []
            for cls in EFFECT_CLASSES:
                dist, off = self.effects.ahead(s, cls)
                ahead += [dist / FAR, off / 40.0]
            fx = np.asarray(ahead)
        else:
            # No map dump, or effects switched off: report "nothing ahead"
            # rather than zeros, which would read as "a booster right here".
            fx = np.tile([1.0, 0.0], len(EFFECT_CLASSES))

        return np.concatenate([
            turbo, reactor,
            [1.0 if (rec.get("cruise") or 0) else 0.0],
            # 1.0 is normal time; the slow-motion gates step it down.
            [rec.get("sim_coef", 1.0)],
            [(rec.get("side_speed", 0.0) or 0.0) / 50.0],
            [rec.get("air_brake", 0.0) or 0.0],
            fx,
        ])

    def _compute_jump_spans(self) -> None:
        """Turn jump patches ([x0,z0,x1,z1] world) into (s0,s1) arc ranges on
        the current line, so the reward can ask "is the car over a gap right
        now" with one number instead of a point-in-polygon test every step.

        Also BRIDGES the lidar over each gap: without ground in the occupancy
        grid there, the forward beam reports an edge right at the take-off and
        the policy brakes into it. Fill the corridor with solid cells so the
        beam marches across and the far side reads as reachable."""
        self._jump_s = []
        if self.line is None or not self._route_jumps:
            return
        for j in self._route_jumps:
            try:
                a = np.array([j[0], 0.0, j[1]], dtype=np.float64)
                b = np.array([j[2], 0.0, j[3]], dtype=np.float64)
            except (TypeError, IndexError):
                continue
            # project() ignores Y via its own indexing; pass the XZ we have.
            ia, sa, _ = self.line.project(a)
            ib, sb, _ = self.line.project(b)
            lo, hi = sorted((sa, sb))
            if hi <= lo:
                continue
            self._jump_s.append((lo, hi))
            self._bridge_lidar_gap(a, b,
                                   float(self.line.points[ia][1]),
                                   float(self.line.points[ib][1]))
        self._jump_s.sort()

    def _bridge_lidar_gap(self, a: np.ndarray, b: np.ndarray,
                          ya: float, yb: float) -> None:
        """Add solid cells along the a->b corridor (with y lerped) so lidar
        beams see continuous ground across a jump gap."""
        grid = getattr(self, "lidar", None)
        grid = grid.grid if grid is not None else None
        if grid is None:
            return
        dist = float(np.hypot(b[0] - a[0], b[2] - a[2]))
        steps = max(2, int(dist / (min(grid.block[0], grid.block[2]) * 0.5)) + 1)
        for k in range(steps + 1):
            t = k / steps
            p = (a[0] + (b[0] - a[0]) * t,
                 ya + (yb - ya) * t,
                 a[2] + (b[2] - a[2]) * t)
            cx, cy, cz = grid.world_to_cell(p)
            for dy in (-1, 0, 1):
                if cx >= 0 and cy + dy >= 0 and cz >= 0:
                    grid._solid.add((int(cx) << 24) | ((int(cy) + dy) << 12)
                                    | int(cz))

    def _over_jump(self, s: float) -> bool:
        return any(lo <= s <= hi for lo, hi in self._jump_s)

    def _jump_runup(self, s: float) -> bool:
        return any(lo - self.jump_approach_m <= s < lo
                   for lo, hi in self._jump_s)

    # -- checkpoints ------------------------------------------------------

    def _load_map(self, uid: str | None) -> None:
        """Everything static about this map: gates, then effects.

        Counting checkpoints by proximity to landmarks is the only reliable
        way - the player API's RaceWaypointTimes reads 0 in Time Attack no
        matter how many you have actually passed, which is why every log line
        used to say cp=0 during runs that were plainly reaching checkpoints.
        """
        self.gates = None
        self.effects = None
        self.lidar = None
        self.map_uid = uid
        self.seen_materials = set()
        self._materials_written = 0

        try:
            reply = self.telem.command("landmarks", wait=3.0)
        except OSError:
            reply = None
        if reply and reply.get("ok"):
            # Take the uid from the LANDMARKS REPLY, not from the telemetry.
            #
            # In splitscreen the top-level record is often a heartbeat that
            # carries no map uid, so this arrived as None - and everything
            # keyed on it then went to the wrong place: the occupancy grid was
            # looked up as maps/None.json and reported "no occupancy grid" for
            # a map that had been dumped, the auto-dump saved under the wrong
            # name, and materials landed in maps/unknown.materials.json. The
            # plugin answers this authoritatively, and we are already asking
            # it, so use its answer.
            if not uid and reply.get("map"):
                uid = reply["map"]
                self.map_uid = uid
                self._last_uid = uid
                self._landmarks_for = uid
                # The config was chosen against the wrong (missing) uid.
                self._switch_config(uid)
                print(f"  map uid resolved from the plugin: {uid}", flush=True)
            # A broker mid-reconnect can hand back a truncated / wrong reply -
            # seen as items being an int, which then crashes the worker and
            # takes the whole SubprocVecEnv down. Re-ask a few times before
            # trusting it; if it never becomes a list, carry on with none
            # rather than raise.
            items = reply.get("items", [])
            tries = 0
            while not isinstance(items, list) and tries < 5:
                tries += 1
                time.sleep(0.5)
                r2 = self.telem.command("landmarks", wait=3.0) or {}
                items = r2.get("items", [])
                if isinstance(items, list):
                    reply = r2
            if not isinstance(items, list):
                print(f"  landmarks reply never returned a list "
                      f"(got {type(items).__name__}) - no gates this load",
                      flush=True)
                items = []
            self.gates = Gates(items, self.line,
                               cluster_m=max(48.0, self.cp_radius * 2))
            self.gates.set_sizes(self.cp_overrides)
            # Rebuild the reference line for the NEW map.
            #
            # Without this the env keeps the line it was constructed with, so
            # loading a second map leaves the policy steered at the first map's
            # geometry - and the failure is silent, because a line is still
            # present and still produces lookahead points. It is the one thing
            # standing between "trains on one track" and "trains on fifty",
            # which is what a driver that generalises actually needs.
            #
            # Only in explore mode: a race line is a recorded lap of one
            # specific track and cannot be regenerated from landmarks. A
            # rotation over maps is therefore an explore-stage activity.
            if self.rebuild_line and self.gates.checkpoints or (
                    self.rebuild_line and self.gates.finish):
                from .mapdata import provisional_line
                try:
                    self.line = provisional_line(self.gates)
                    print(f"  reference line rebuilt for {uid}: "
                          f"{len(self.line.points)} samples, "
                          f"{self.line.length:.0f}m", flush=True)
                except Exception as ex:
                    print(f"  could not rebuild the line for {uid}: {ex}",
                          flush=True)
            print(f"landmarks: {self.gates.describe()}  (map {uid})", flush=True)
            # How wide each gate actually is, so a checkpoint that never
            # counts can be recognised as a sizing problem rather than a
            # driving one. `lateral` is measured from the gate's own
            # landmarks - one per block - so a five-lane gate reads 128m and a
            # 1x1 gate reads 0 and is entirely margin.
            for i, sp in enumerate(self.gates.spans()):
                w, h = self.gates.size_of(i, self.cp_half_width, self.cp_height)
                mark = "  <- overridden" if str(i) in self.cp_overrides \
                    or i in self.cp_overrides else ""
                print(f"  cp {i}: {sp['blocks']} block(s), {sp['lateral']:.0f}m "
                      f"wide, using half_width {w:.0f}m height {h:.0f}m{mark}",
                      flush=True)
        else:
            why = (reply or {}).get("err", "no reply")
            print(f"landmarks unavailable ({why}) - checkpoint counting off",
                  flush=True)

        try:
            dump = self.telem.command("dumpmap", wait=8.0)
        except OSError:
            dump = None
        if dump and dump.get("ok"):
            self.effects = EffectMap(dump, self.line)
            skipped = dump.get("free_skipped", 0)
            note = (f", {skipped} free blocks skipped (need Developer mode)"
                    if skipped else "")
            print(f"map effects: {self.effects.describe()}"
                  f"  [{dump.get('blocks', 0)} blocks, "
                  f"{dump.get('items', 0)} items{note}]", flush=True)
        else:
            why = (dump or {}).get("err", "no reply")
            print(f"map dump unavailable ({why}) - the policy will not be told "
                  f"about boosters ahead", flush=True)

        # Occupancy is a separate, much larger dump, so it is cached to
        # maps/<uid>.json and only fetched the first time a map is seen.
        # Positions the car provably occupies, used once to resolve how block
        # footprints sit relative to their stored coordinate. A recorded lap is
        # the strongest evidence there is - every sample came from a car that
        # was actually there.
        #
        # In explore mode it is the opposite: the line is a straight run
        # through the checkpoints that deliberately cuts across scenery, so
        # feeding it in scored every convention at 63% and told us nothing.
        # Landmarks only there.
        on_track = [] if self.profile == "explore" else list(self.line.points[::10])
        if self.gates is not None:
            if self.gates.spawn is not None:
                on_track.append(self.gates.spawn)
            on_track += list(self.gates.centres)
        try:
            grid = load_or_fetch(
                ROOT, uid,
                lambda: self.telem.command("dumpmap occupancy", wait=20.0),
                on_track=on_track)
        except Exception as ex:
            grid, _ = None, print(f"occupancy dump failed: {ex}", flush=True)
        if grid is not None and len(grid):
            self.lidar = Lidar(grid)
            print(f"lidar: {self.lidar.n} ground beams over "
                  f"{len(grid)} occupied cells, range "
                  f"{self.lidar.max_range:.0f}m", flush=True)
        else:
            print("no occupancy grid - lidar beams will read 'nothing in range'",
                  flush=True)

        # Explore-stage reference line: layer the best route estimate available.
        #
        #     hand edits  >  learned-from-runs  >  roadtrace  >  provisional
        #
        # roadtrace follows the actual road-block ribbon in 3D and recovers the
        # real checkpoint order; learnedmap refines whatever the explorer has
        # driven; edits are polyline patches drawn in the panel over the gaps
        # the automatic layers cannot see. A rasterised occupancy grid was
        # tried here first and pulled - it is too coarse and its recovered
        # order was not stable between builds.
        if (self.rebuild_line and self.gates is not None
                and self.gates.checkpoints and self.gates.finish):
            try:
                from .routemodel import merged_line
                dump = None
                if grid is not None:
                    dump = {"boxes": getattr(grid, "boxes", None),
                            "names": getattr(grid, "names", None),
                            "base_height": getattr(grid, "base_height", 8),
                            "block_size": list(getattr(grid, "block",
                                                       (32, 8, 32)))}
                rm = merged_line(ROOT, uid, gates=self.gates, dump=dump)
                if rm is not None:
                    order = rm.get("order")
                    if (order and len(order) == len(self.gates.centres)
                            and sorted(order) == list(range(len(order)))
                            and order != list(range(len(order)))):
                        self.gates.reorder(order)
                        print(f"  checkpoint order set by {rm['source']}: "
                              f"{order}", flush=True)
                    self.line = rm["line"]
                    self._track_hw = rm.get("half_width")
                    self._track_sides = rm.get("sides")
                    self._route_jumps = rm.get("jumps", [])
                    self._compute_jump_spans()
                    print(f"  reference line [{rm['source']}]: "
                          f"{len(self.line.points)} pts, "
                          f"{self.line.length:.0f} m"
                          + (f"  ({len(self._jump_s)} jump span(s))"
                             if self._jump_s else ""), flush=True)
            except Exception as ex:                       # noqa: BLE001
                print(f"  route model unavailable ({ex}) - "
                      f"keeping the provisional line", flush=True)
        elif self.line is not None and self._track_hw is None:
            self._adopt_track_geometry(uid)

    def _adopt_track_geometry(self, uid: str | None) -> None:
        """Give a NON-rebuilt line the map's edge geometry.

        The race stage runs with rebuild_line=False - it is handed an explicit
        --line and must not re-derive one. But `half_width` and `sides` were
        only ever loaded inside that rebuild, so the racer had `_track_hw` and
        `_track_sides` at None, and every edge feature silently returned its
        safe default: sides_ahead all 1.0 ("fully barriered"), margin 1.0
        ("dead centre"), width_ahead all 1.0 ("wide open").

        That is 13 of 148 inputs frozen at constants the model was trained to
        read as real signals - and the three that exist specifically to handle
        the unbarriered platform run. A policy that could see the edge in
        explore is blind to it in race, which looks exactly like it has
        forgotten the track.

        The arrays belong to the MAP, not to any one line - they come from the
        occupancy dump via roadtrace. They are indexed by the roadtrace line's
        samples though, so they are transferred by nearest point rather than
        copied: for each sample of OUR line, take the value at the closest
        roadtrace sample.
        """
        if not uid:
            return
        try:
            import json as _json
            path = os.path.join(ROOT, "maps", f"{uid}.roadtrace.json")
            with open(path) as fh:
                rt = _json.load(fh)
            src = np.asarray(rt.get("points") or [], dtype=np.float64)
            hw = rt.get("half_width")
            sd = rt.get("sides")
            if src.size == 0 or hw is None or sd is None:
                return
            hw = np.asarray(hw, dtype=np.float64)
            sd = np.asarray(sd, dtype=np.float64)
            n = min(len(src), len(hw), len(sd))
            src, hw, sd = src[:n, :3], hw[:n], sd[:n]

            mine = np.asarray(self.line.points, dtype=np.float64)[:, :3]
            # Nearest roadtrace sample for each of ours, in XZ - height differs
            # between a driven line and the block ribbon and would dominate.
            d2 = ((mine[:, None, 0] - src[None, :, 0]) ** 2 +
                  (mine[:, None, 2] - src[None, :, 2]) ** 2)
            idx = np.argmin(d2, axis=1)
            self._track_hw = hw[idx]
            self._track_sides = sd[idx]
            unb = int((self._track_sides < 0.25).sum())
            print(f"  track geometry adopted from roadtrace: "
                  f"{len(idx)} samples, {unb} unbarriered "
                  f"({100.0 * unb / max(len(idx), 1):.0f}%)", flush=True)
        except Exception as ex:                           # noqa: BLE001
            print(f"  could not adopt track geometry ({ex}) - edge features "
                  f"will read as 'barriered everywhere'", flush=True)

    def _check_input_fidelity(self, rec: dict, steer: float,
                              gas: float) -> None:
        """How much of the commanded input the game actually applied.

        Only sampled while the command has been held STEADY for a few steps.
        Measuring during a change would measure the game's input ramp, which
        is meant to lag - the thing worth catching is a command held at full
        deflection that never gets there.
        """
        if abs(steer - self.prev_steer) < 0.02:
            self._steady += 1
        else:
            self._steady = 0

        applied = rec.get("in_steer")
        if applied is None or self._steady < self.fidelity_settle \
                or abs(steer) < 0.5:
            return
        self._fidelity.append(min(abs(float(applied)) / abs(steer), 1.5))
        if len(self._fidelity) > self.fidelity_window:
            self._fidelity.pop(0)
        if len(self._fidelity) < self.fidelity_window:
            return

        self.applied_ratio = float(np.mean(self._fidelity))
        if self.applied_ratio < self.fidelity_floor and not self._fidelity_warned:
            self._fidelity_warned = True
            print(f"  INPUT FIDELITY: the game is applying "
                  f"{self.applied_ratio * 100:.0f}% of the steering we send. "
                  f"That is almost always an unfocused window throttling the "
                  f"frame rate - TM2020 ramps steering per frame. Give this "
                  f"instance its own display (Xvfb) so it is always focused; "
                  f"transitions recorded like this do not mean the same thing "
                  f"as focused ones.", flush=True)

    def _next_gate_pos(self) -> "np.ndarray | None":
        """World-space centre of the next thing to drive at: the lowest-index
        uncrossed checkpoint, or the finish once they are all taken. Used by
        the line-free cp_approach shaping. Returns None if the map has no
        gate data yet."""
        g = self.gates
        if g is None:
            return None
        k = len(self.cp_taken)
        if k < len(getattr(g, "centres", [])):
            return g.centres[k]
        if getattr(g, "finish", None):
            return np.mean(np.asarray(g.finish[0], dtype=np.float64), axis=0)
        return None

    def _count_cp(self, pos: np.ndarray, race_time=None) -> int:
        """How many distinct checkpoint gates have been taken this episode.

        A checkpoint is credited for GOING THROUGH THE GATE, not for being
        near it: the step has to cross the gate's plane, inside its width and
        within a grid level of its height. Proximity alone credited a
        checkpoint for passing it on the road running alongside, which made
        the reward reachable without ever driving through anything.

        Order-free by default, because that is the game's own rule: TM2020
        requires all checkpoints before the finish counts but does not fix the
        sequence unless the mapper has linked them. Counting strictly in order
        would refuse to credit a checkpoint the game itself credited, and would
        punish a policy that found a faster ordering.

        `cp_strict_order` restores sequential counting for a map that really
        does link its checkpoints; `cp_mode: "sphere"` restores the old
        proximity test, which is only useful for comparing against traces
        recorded before this changed.
        """
        if self.gates is None:
            return len(self.cp_taken)
        prev = self.prev_pos
        gate = self.cp_mode != "sphere" and prev is not None
        if self.cp_strict_order:
            nxt = len(self.cp_taken)
            took = (self.gates.crossed(prev, pos, nxt, self.cp_half_width,
                                       self.cp_height) if gate
                    else self.gates.hit(pos, nxt, self.cp_radius))
            if took:
                self.cp_taken.add(nxt)
                self._split(race_time)
        else:
            i = (self.gates.crossed_any(prev, pos, self.cp_taken,
                                        self.cp_half_width, self.cp_height)
                 if gate else
                 self.gates.hit_any(pos, self.cp_taken, self.cp_radius))
            if i is not None:
                self.cp_taken.add(i)
                self._split(race_time)
        return len(self.cp_taken)

    def _split(self, race_time) -> None:
        """Record the race clock at a checkpoint.

        The game's clock, not ours: it is what the leaderboard would show and
        it is unaffected by a control-loop overrun, so two runs are comparable
        even when one of them stuttered.
        """
        if race_time is not None:
            self.splits.append(int(race_time))

    # -- run traces -------------------------------------------------------

    def _map_dir(self) -> str:
        return os.path.join(self.runs_dir, self.map_uid or "unknown")

    def _write_splits(self, race_time, finished: bool) -> None:
        """Append this episode's section times to runs/<map>/splits.jsonl.

        Every episode, not just the good ones. A section is optimised by
        comparing its own best against its typical, and throwing away the
        typical leaves nothing to compare with. The file is one JSON object
        per line, so it can be tailed while training runs.
        """
        if not self.splits and not finished:
            return
        try:
            d = self._map_dir()
            os.makedirs(d, exist_ok=True)
            row = {"step": int(self.total_steps), "cp": len(self.splits),
                   "splits": [int(x) for x in self.splits],
                   "finished": bool(finished),
                   "race_time": int(race_time) if race_time is not None else None,
                   "instance": int(self.instance)}
            with open(os.path.join(d, "splits.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            # Losing a split line must never take a training run down.
            pass

    def _write_trace(self, dist: float, race_time: float | None,
                     finished: bool) -> None:
        """Keep the trace of any run that got further or faster than the best
        so far. The game only autosaves a replay for a *finished* PB, so
        without this every run that died two thirds of the way round - which is
        most of them, and the interesting ones - would leave nothing behind."""
        if not self.trace:
            return
        d = os.path.join(self._map_dir(), "traces")
        stamp = f"t{race_time / 1000:.3f}s" if (finished and race_time) else \
                f"d{dist:.0f}m"
        # The instance index is in the name because N envs share one runs/
        # directory and would otherwise overwrite each other's episode 12.
        inst = f"_i{self.instance}" if self.instance else ""
        # Prefix the file with a slug of the real track title when we have one,
        # so the traces dir is browsable without a uid->name lookup.
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.map_name).strip("-").lower()
        pfx = f"{slug}_" if slug else ""
        name = (f"{pfx}ep{self.episodes:05d}_step{self.total_steps // 1000:05d}k"
                f"{inst}_{stamp}.json")
        try:
            os.makedirs(d, exist_ok=True)
            # A sidecar so anything (the panel, tools) can turn the uid back
            # into the human title without carrying telemetry.
            if self.map_name:
                try:
                    with open(os.path.join(self._map_dir(), "name.txt"), "w") as nf:
                        nf.write(self.map_name + "\n")
                except OSError:
                    pass
            tmp = os.path.join(d, name + ".tmp")
            with open(tmp, "w") as f:
                json.dump({
                    "map": self.map_uid,
                    "map_name": self.map_name or None,
                    "episode": self.episodes,
                    "step": self.total_steps,
                    "finished": finished,
                    "race_time": race_time,
                    "distance": dist,
                    "checkpoints": len(self.cp_taken),
                    "fields": ["t", "x", "y", "z", "speed",
                               "steer", "gas", "brake", "cp"],
                    "samples": self.trace,
                }, f)
            os.replace(tmp, os.path.join(d, name))
        except OSError as ex:
            print(f"  could not write trace: {ex}", flush=True)

    # -- gym API ----------------------------------------------------------

    def _hard_restart(self):
        """RequestRestartMap. Correct but expensive: it replays the whole intro
        sequence. Only for the first episode and as a fallback - and even then
        _wait_for_respawn mashes past the fly-in rather than sitting through it.

        Do NOT wait on SpawnStatus: it reads 0 (NotSpawned) even while the car
        is demonstrably driving, the same way RaceWaypointTimes reads 0.
        Waiting on it burned the full timeout every reset.

        **Never in splitscreen.** RequestRestartMap restarts the map for the
        whole game, so one seat failing to respawn would reset all four cars -
        teleporting three policies mid-episode and recording transitions that
        never happened. One seat's problem must not become everybody's. There
        it keeps pressing give-up instead, and says so rather than escalating
        silently.
        """
        if self.slot is not None:
            self.restart_refusals += 1
            print(f"  seat {self.slot}: give-up is not taking, but a map "
                  f"restart would reset all four cars - retrying give-up "
                  f"instead (refusal #{self.restart_refusals}). If this keeps "
                  f"up, that seat's pad has stopped reaching the game.",
                  flush=True)
            for _ in range(3):
                rec = self._slot_view(self.telem.get()) or {}
                prev_time = rec.get("race_time") or 0
                self.pad.press(self.giveup_button, self.giveup_hold_ms)
                out = self._wait_for_respawn(prev_time, timeout=6.0)
                if out is not None:
                    return out
            return self._slot_view(self.telem.get())

        self.telem.command("restart")
        self._wait_for(lambda r: r.get("ui") != UI_PLAYING, timeout=3.0)
        return self._wait_for_respawn(prev_time=5000, timeout=20.0)

    def _begin_reset(self, rec: dict) -> None:
        """Press the button that starts the respawn, and return immediately.

        ALWAYS give up, including from the finish screen.
        
        Measured: a give-up respawn lands in 0.02s. "Improve" replays the
        intro and takes seconds, which is why `_reset_sequence` only falls
        back to it after `finish_fallback_after` failed give-ups rather than
        reaching for it first. An earlier version of this method picked
        Improve on UI_FINISH, which skipped that ladder and sent every finish
        down the slow path - the exact stall it was meant to remove.
        """
        try:
            # A clean sector attempt resets by RESPAWN, which lands on the
            # last checkpoint passed - the sector's entry gate, because the
            # attempt deliberately stopped short of the exit. Give-up would
            # send it back to the start line and throw away the whole point.
            # Give-up must be HELD; respawn is a TAP. Holding y for the
            # give-up duration risks the game reading it as something else -
            # a double tap is a STANDING respawn, which would throw away the
            # entry speed the whole drill depends on.
            if self._sector_respawn:
                btn, hold = self.respawn_button, self.respawn_hold_ms
            else:
                btn, hold = self.giveup_button, self.giveup_hold_ms
            self.pad.press(btn, hold)
            self._reset_pressed_at = time.time()
            self._reset_from_time = rec.get("race_time") or 0
        except Exception:
            # A failed early press just means reset() does it the slow way.
            self._reset_pressed_at = None

    def _reset_sequence(self):
        """Put the car back on the start line, as cheaply as possible.

        Two states are actionable:

          Playing (1)  a run in progress; the bound give-up button works.
          Finish (11)  the run ended and the Improve / Save Replay / Exit
                       screen is up.

        On the finish screen the give-up button still works and respawns
        instantly - but only once the screen has actually accepted input.
        Pressing too early is how resets were getting eaten after a personal
        best. So attempts escalate the *settle delay* first, and only fall back
        to "Improve" after `finish_fallback_after` tries.

        That fallback order matters more than it looks. Improve always works,
        but it goes through the full restart WITH the ~8s intro - the exact
        cost give-up exists to avoid. Reaching for it first makes every reset
        after a PB eight seconds slower, and if the wait is then too short the
        ladder retries on top of a reset that was already running.

        Anything else - loading, intro, outro - is waited out rather than
        pressed into, because a button fired at a menu that is not listening is
        silently lost.

        There is no script API for give up (CSmArenaRulesEvent.GiveUp is a
        read-only server-side event), so it goes through the virtual pad.

        Success is the race clock going backwards. SpawnStatus is not usable
        here - it reports NotSpawned while the car is driving.

        This works per seat in splitscreen: the race clock is PER SEAT, not
        shared. Four seats that started together read the same value, which
        looks shared and is not - giving up on seat 0 alone took it from
        119670 to 2450 while the other three carried on to 123650.

        `dist` is not a substitute, incidentally: it is cumulative distance
        driven and does NOT reset with the car, so a respawn leaves it
        unchanged.
        """
        attempt = 0
        waits = 0
        while attempt < self.giveup_retries:
            rec = self._slot_view(self.telem.get()) or {}
            ui = rec.get("ui", UI_NONE)
            prev_time = rec.get("race_time") or 0

            if ui not in (UI_PLAYING, UI_FINISH):
                waits += 1
                if waits > 3:
                    return None
                self._wait_for(lambda r: r.get("ui") in (UI_PLAYING, UI_FINISH),
                               timeout=3.0)
                continue

            settle = self.giveup_settle_ms * (1 + attempt) / 1000.0
            use_improve = (ui == UI_FINISH
                           and attempt >= self.finish_fallback_after)
            button = self.finish_button if use_improve else self.giveup_button
            # Improve replays the intro, so give it room; a give-up respawn
            # lands in well under a second and should not sit on a long wait.
            timeout = self.finish_timeout if use_improve else self.quick_timeout
            attempt += 1

            if ui == UI_FINISH or attempt > 1:
                time.sleep(settle)
            self.pad.press(button, self.giveup_hold_ms)

            out = self._wait_for_respawn(prev_time, timeout)
            if out is not None:
                return out
        return None

    def reset(self, *, seed=None, options=None):
        if gym is not None:
            super().reset(seed=seed)
        # Baseline for the per-episode overrun count (see "overruns_ep").
        self._overruns_at_reset = self.overruns
        self.pad.reset()

        # A seat NEVER blocks here.
        #
        # SB3's step_wait waits for every worker, and a worker sitting in
        # reset() freezes the whole batch - so the other cars receive no new
        # actions, and a gamepad latches, so they keep driving on their last
        # input. Their recorded transitions then claim a 25ms step that really
        # lasted seconds. That is the "one car finishes and everything stops"
        # behaviour, and no amount of making the respawn faster removes it:
        # the fix is not to wait here at all.
        #
        # So: ask for the respawn, hand back the last observation, and let
        # step() carry on. The seat idles at the start line for a few steps
        # while the others drive on, and those steps are flagged `not_ready`.
        #
        # The first episode still blocks - there is no previous observation to
        # hand back, and nobody else is running yet.
        if (self.slot is not None and self.episodes > 0
                and self._last_obs is not None
                and self.reset_mode == "giveup"):
            self._request_respawn()
            self._respawning = True
            return self._last_obs, {}

        if self.reset_mode == "giveup" and self.episodes > 0:
            # The terminating step already pressed the button, so the respawn
            # has been running for however long the rest of the batch took.
            # Check whether it has already landed before spending the full
            # sequence on it - on a fleet that is most of the saving, because
            # the other workers' step time is exactly the time this had to
            # complete in.
            rec = None
            if self._reset_pressed_at is not None:
                rec = self._wait_for_respawn(self._reset_from_time, timeout=0.6)
                self._reset_pressed_at = None
            if rec is None:
                rec = self._reset_sequence()
            if rec is None:
                # Only complain once the whole ladder is spent. The old code
                # printed on the first miss, including on resets that worked.
                print(f"give-up reset did not take after {self.giveup_retries}"
                      " tries - falling back to restart", flush=True)
                rec = self._hard_restart()
        else:
            rec = self._hard_restart()

        if rec is None or not rec.get("car") or not _real_pos(rec.get("pos")):
            rec = self._wait_for_car()
        if rec is None:
            raise RuntimeError("no telemetry after reset - is the plugin loaded?")
        if not rec.get("car") or not _real_pos(rec.get("pos")):
            # A vanished car is usually temporary, so wait it out rather than
            # ending the run.
            #
            # A splitscreen lobby reloads when its round timer expires - the
            # limit is whatever it was set to - and drops back to a single
            # player until the seats rejoin. A run that aborted on that could
            # not outlive the timer, and the traceback would blame the seat
            # rather than the clock.
            #
            # So: keep waiting, say so once, and pick up when the car returns.
            seat = "" if self.slot is None else f"seat {self.slot}: "
            print(f"  {seat}no car - waiting for the lobby to come back "
                  f"(round timer, map reload, or a seat that has not "
                  f"rejoined). Will keep trying for "
                  f"{self.lobby_wait / 60:.0f} minutes.", flush=True)
            rec = self._wait_for_car(timeout=self.lobby_wait)
            if rec is None or not rec.get("car") or not _real_pos(rec.get("pos")):
                raise RuntimeError(
                    f"{seat}still no car after "
                    f"{self.lobby_wait / 60:.0f} minutes. The game is not in a "
                    f"race, or this seat has no player in it - in splitscreen "
                    f"every seat needs its own pad bound before it will spawn "
                    f"a car.")
            print(f"  {seat}car is back after "
                  f"{self.episodes} episode(s) - carrying on", flush=True)
        return self._start_episode(rec)

    def _start_episode(self, rec: dict):
        """Everything that needs the car to be BACK before it can be set up.

        Split out of reset() so it can also be called from step(), which is
        what lets a seat hand control back immediately and finish arriving a
        moment later. See _request_respawn.
        """
        self.episodes += 1

        # Only ask on a map change. Retrying every episode would spend the
        # command timeouts per reset on any map where it isn't available.
        uid = rec.get("map") or self._last_uid or None
        if uid:
            self._last_uid = uid
        nm = (rec.get("map_name") or "").strip()
        if nm:
            self.map_name = nm
        if uid != self._landmarks_for:
            self._landmarks_for = uid
            self._switch_config(uid)
            self._load_map(uid)
            if self.watcher is not None:
                self.watcher.out_dir = os.path.join(self._map_dir(), "replays")
        elif self.cfg.maybe_reload():
            self._apply_config()

        self._reset_pressed_at = None
        self.prev_action[:] = 0.0
        self.prev_steer = 0.0
        self.hint_pressed = False
        self.hint_started = 0
        self.tapper.reset()
        self.prev_pos = np.asarray(rec.get("pos", [0, 0, 0]), dtype=np.float64)
        self.prev_speed = rec.get("speed", 0.0) or 0.0
        self.gear = int(rec.get("gear", 0) or 0)
        self.gear_held = 0
        self._next_step_at = None
        self.steps = 0
        self.slow_for = 0
        self.moved = False
        # Furthest progress when the stall timer was last reset.
        self._stall_ref_s = 0.0
        self._stalled_for = 0
        # A LAUNCHED RESPAWN puts the car back on the track shortly before the
        # checkpoint it respawned to, carrying its momentum - so a sector
        # attempt does not begin at gate 0, it begins at the sector's entry
        # gate with those gates already banked by the game.
        #
        # Starting cp_taken empty made the episode expect gate 0 next, so
        # re-crossing the entry gate credited nothing, `cp` stayed 0, and the
        # sector timer never armed. Seeding the gates behind the car is what
        # makes the drill loop work at all.
        entry_gate = (self.sector_exit - 1) if self.sector_exit else 0
        self.cp_taken = set(range(entry_gate)) if entry_gate > 0 else set()
        # Markers pay once per EPISODE, not once per run.
        self._markers_paid = set()
        self._sector_entry_t = None
        self._sector_respawn = False
        self._prev_cp_dist = None
        self._cp_approach_paid = 0.0
        self._jump_taken_at = set()
        self._green_streak = 0
        self._steps_since_progress = 0
        self._dv_ema = 0.0
        self.splits = []
        self.bad_surface_steps = 0
        self.trace = []
        self.ep_parts = {}
        # The car has been teleported back to the start, so last episode's
        # place on the line says nothing about this one: search globally once.
        self._line_idx = None
        self._prev_basis = None
        obs, s, offset = self._observe(rec)
        self._last_obs = obs
        self.prev_s = s
        # Where this episode began, and the furthest it ever got. Net progress
        # alone cannot distinguish "never moved" from "went a long way and
        # came back", and it reads as negative if the episode began part-way
        # along the line - which is exactly the confusion this pair resolves.
        self.start_s = s
        self.max_s = s

        # A car that starts further from the reference line than the episode
        # limit cannot do anything except terminate on step one, forever. That
        # is never a driving failure - it means the line belongs to a different
        # map, or was recorded on a different route - so it must stop the run
        # loudly rather than burn thousands of one-step episodes looking like
        # the policy is simply bad. (It did: 2140 episodes in 60 seconds.)
        if offset > self.max_offset:
            pos = rec.get("pos")
            seat = "" if self.slot is None else f", seat {self.slot}"
            # Say WHICH of the two it is. A car at the origin is an empty
            # telemetry record, not a mismatched line, and the two want
            # completely different fixes.
            origin = pos is None or all(abs(float(q)) < 1e-6 for q in pos)
            raise RuntimeError(
                f"the car spawns {offset:.0f}m from the reference line, which "
                f"is past the {self.max_offset:.0f}m limit - so every episode "
                f"would end instantly.\n"
                f"  car position: {pos}{seat}"
                + ("  <- the origin, i.e. the telemetry record was EMPTY. "
                   "That is not a line mismatch: the game was between rounds "
                   "or this seat had no car yet.\n" if origin else "\n")
                + f"  line runs: {self.line.points[0].round(1).tolist()} -> "
                  f"{self.line.points[-1].round(1).tolist()}\n"
                f"  map loaded: {self.map_uid}\n"
                f"  this is almost always a line recorded on a different map. "
                f"Record one for this map, or use --stage explore, which needs "
                f"no recorded lap at all.")
        return obs, {}

    def _request_respawn(self, rec: dict | None = None) -> None:
        """Ask for the respawn and return at once. Does not wait."""
        if rec is None:
            rec = self._slot_view(self.telem.get()) or {}
        self._respawn_from = rec.get("race_time") or 0
        self._respawn_started = time.time()
        self._respawn_presses = 1
        self._respawn_warned = False
        self._respawn_stuck_warned = False
        self._forced_restart_at = 0.0
        # A clean sector attempt respawns to its entry checkpoint instead of
        # giving up to the start line. Patching only the early-press site left
        # this one pressing give-up, so every sector drill silently ran from
        # the start - the times were a full run, not a sector.
        # Where the car was when respawn was asked for, so "has it moved back
        # yet?" is answerable.
        self._respawn_from_s = float(self.max_s)
        if self._sector_respawn:
            self.pad.press(self.respawn_button, self.respawn_hold_ms)
        else:
            self.pad.press(self.giveup_button, self.giveup_hold_ms)

    def _respawn_landed(self, rec: dict) -> bool:
        """Is this seat's car back on the start line?

        The race clock is per seat and resets with the car, which is the one
        signal that distinguishes seats. Being out of the intro matters too:
        the clock reads ~0 during the fly-in, so the clock alone would hand
        back control before the car can be steered.
        """
        if not rec.get("car") or not _real_pos(rec.get("pos")):
            return False
        if rec.get("ui") not in (UI_PLAYING, UI_NONE):
            return False
        # A LAUNCHED respawn to a checkpoint does not reset the race clock -
        # it carries on from that checkpoint's split. The clock test below is
        # written for give-up, which does reset it, so a sector respawn never
        # read as "landed" and the retry loop hammered the button forever.
        # Being drivable again IS the landing here.
        if self._sector_respawn:
            # ...but only once the car has actually MOVED BACK. Returning True
            # the moment it is drivable hands control back before the respawn
            # has happened, so the next attempt starts still sitting past the
            # stop line and terminates instantly - a 0.00s "sector".
            try:
                here = self.line.project_near(
                    np.asarray(rec.get("pos"), dtype=np.float64),
                    None)[1]
            except Exception:                              # noqa: BLE001
                return True
            if here < (self._respawn_from_s - 20.0):
                return True
            # Fallback: never hang if the car legitimately respawned somewhere
            # that does not read as "back" (a line that folds on itself).
            return (time.time() - self._respawn_started) > 3.0
        rt = rec.get("race_time") or 0
        if rt >= max(self._respawn_from - 500, 3000):
            return False

        # The clock is not enough on its own. It resets in ~0.02s, but the
        # POSITION takes another frame or two to follow - so accepting on the
        # clock alone starts the episode with the car still reported at its
        # old spot, hundreds of metres along the line. `prev_s` is then set
        # from that stale position, and the rest of the episode measures
        # progress backwards from it: episode 87 ended at 154m having
        # "started" at 220m and was paid -66 for a run that took both
        # checkpoints.
        #
        # A give-up respawn puts the car on the start line, so requiring it to
        # actually be there closes the gap. Give up after a while rather than
        # hang - being a little late is recoverable, waiting forever is not.
        pos = rec.get("pos")
        if pos is not None and self.line is not None:
            start = self.line.points[0]
            back = float(np.linalg.norm(np.asarray(pos, dtype=np.float64)
                                        - start))
            if back > self.respawn_near_start:
                waited = time.time() - self._respawn_started
                if waited < 3.0:
                    return False
                if not self._respawn_warned:
                    self._respawn_warned = True
                    print(f"  seat {self.slot}: clock reset but the car is "
                          f"still {back:.0f}m from the start line after "
                          f"{waited:.1f}s - starting anyway", flush=True)
        return True

    def _step_respawning(self):
        """One control step spent waiting for this seat's car to come back.

        This is the whole point of the non-blocking reset. SB3's step_wait
        does not return until EVERY worker has, and a worker that blocks in
        reset() therefore freezes the entire batch - so the other cars get no
        new actions, and because a gamepad latches they carry on driving on
        whatever they were last told. The transitions recorded for them then
        describe a 25ms step that really lasted several seconds.

        Returning promptly with the pad held neutral keeps every car
        independent: this seat idles at the start line for a few steps while
        the others drive on normally.
        """
        self.pad.act(0.0, 0.0, 0.0)

        now = time.perf_counter()
        if self._next_step_at is None or now - self._next_step_at > 0.5:
            self._next_step_at = now
        self._next_step_at += self.dt
        delay = self._next_step_at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

        rec = self._slot_view(self.telem.get()) or {}
        self.not_ready_steps += 1

        if self._respawn_landed(rec):
            self._respawning = False
            obs, _info = self._start_episode(rec)
            self._last_obs = obs
            return obs, 0.0, False, False, {"not_ready": True,
                                            "instance": self.instance}

        # Give up did not take. In splitscreen there is no "Improve" button -
        # the finish-screen restart is single-player only - so mashing the
        # give-up button is the ONLY per-seat recovery. Hammer it: a press
        # every ~0.8s, and once finish_fallback_after presses have gone by with
        # nothing, alternate a longer double-hold in case the short taps are
        # landing on unfocused frames. (finish_button is still tried, but only
        # if the config sets finish_fallback_after low AND you are not in
        # splitscreen - leave it high for splitscreen.)
        waited = time.time() - self._respawn_started
        # One press, and only one. Respawn is instant; re-pressing it just
        # respawns again, which is what "it keeps pressing respawn" looked
        # like from the game side.
        if self._sector_respawn:
            return self._last_obs, 0.0, False, False, {
                "not_ready": True, "instance": self.instance}
        if waited > self._respawn_presses * 0.8:
            self._respawn_presses += 1
            in_splitscreen = self.slot is not None
            use_finish = (not in_splitscreen
                          and self._respawn_presses > self.finish_fallback_after)
            if self._sector_respawn and not use_finish:
                button, hold = self.respawn_button, self.respawn_hold_ms
            else:
                button = self.finish_button if use_finish else self.giveup_button
                hold = self.giveup_hold_ms
            if self._respawn_presses > self.finish_fallback_after:
                hold = max(hold, 600.0)     # longer hold once short taps fail
            self.pad.press(button, hold)

        # The game is in the intro - a restart already happened (ours, another
        # seat's, or a lobby rollover). Do NOT fire another: the intro lasts
        # longer than restart_after_s, so re-firing here is an infinite loop
        # that never lets the intro finish. Sit quietly and reset the wedge
        # clocks so the intro time is not counted as "still wedged".
        if rec.get("ui") in (UI_INTRO, UI_OUTRO):
            self._respawn_started = time.time()
            self._forced_restart_at = time.time()
            return (self._last_obs, 0.0, False, False,
                    {"not_ready": True, "instance": self.instance})

        # Escalation: a splitscreen seat that still has not come back after
        # restart_after_s of mashing give-up asks for a full map restart. Yes,
        # that resets the other seats mid-episode - but they detect the intro
        # in step() and abort their episode cleanly, which is a far smaller
        # loss than this seat sitting borked at the start for a minute while
        # its "episode" clock keeps running and poisoning the buffer. Never
        # fire two within 20s - the intro must be allowed to complete first.
        if (self.slot is not None and self.restart_after_s > 0
                and waited > self.restart_after_s
                and time.time() - self._forced_restart_at
                    > max(self.restart_after_s, 20.0)):
            self._forced_restart_at = time.time()
            n = "" if self.slot is None else f"seat {self.slot}: "
            print(f"  {n}give-up has not taken in {waited:.0f}s - forcing a "
                  f"full map restart (resets every seat; they abort their "
                  f"episodes on the intro)", flush=True)
            try:
                self.telem.command("restart")
            except Exception as e:
                print(f"  {n}restart command failed: {e}", flush=True)

        if waited > self.lobby_wait and not self._respawn_stuck_warned:
            # Do NOT raise. A hard exception here kills this SubprocVecEnv
            # worker; SB3's step_wait then gets EOFError on the closed pipe and
            # the ENTIRE run goes down - one wedged splitscreen seat should not
            # do that. Instead hold this seat neutral, keep pressing give-up,
            # and let the other seats carry on training. If the car ever comes
            # back _respawn_landed() picks it up and the episode resumes.
            self._respawn_stuck_warned = True
            print(f"  seat {self.slot}: no respawn after {waited:.0f}s - "
                  f"holding this seat neutral and still retrying; the run "
                  f"keeps going on the other seats", flush=True)

        return (self._last_obs, 0.0, False, False,
                {"not_ready": True, "instance": self.instance})

    def step(self, action):
        # Still waiting for this seat's car to come back from the last
        # episode. Costs one control period and returns, so the batch is
        # never held up.
        if self._respawning:
            return self._step_respawning()

        steer, gas, brake = self._decode(action)
        # Slew-rate limit the steering: the policy's raw output is the TARGET,
        # and we walk toward it at most steer_rate*dt per step. Without this
        # the game only ever sees a fraction of a slammed input (its own ramp
        # can't keep up), and the transitions in the buffer are a lie about
        # what the car did. prev_steer is the last value we actually sent.
        if self.steer_rate > 0.0:
            max_d = self.steer_rate * self.dt
            steer = max(self.prev_steer - max_d,
                        min(self.prev_steer + max_d, steer))
        self.pad.act(steer, gas, brake)

        # Hold the control period against an ABSOLUTE clock, not against the
        # time this call started.
        #
        # The difference matters as soon as anything else is sharing the loop.
        # SB3 runs its gradient update between env steps, in the same thread;
        # sleeping `dt` from the top of step() would put the learner's time on
        # TOP of the control period, so a 10ms update turns 40Hz into 28Hz
        # while every transition in the buffer still claims to be a 25ms one.
        # Waiting until the next scheduled instant instead absorbs the
        # learner into the period, and the rate stays what the model thinks it
        # is right up until the work genuinely does not fit.
        now = time.perf_counter()
        if self._next_step_at is None or now - self._next_step_at > 0.5:
            self._next_step_at = now      # first step, or we fell badly behind
        self._next_step_at += self.dt
        delay = self._next_step_at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            # Overrunning means the control rate is a lie. Count it rather
            # than silently drifting; the panel and the log both surface it.
            self.overruns += 1
            self._next_step_at = time.perf_counter()

        rec = self._slot_view(self.telem.get())
        if rec is None:
            raise RuntimeError("telemetry stopped mid-episode")

        # The map is restarting under us - our own doing (another seat wedged
        # and called for it) or a lobby rollover. The car is about to be
        # teleported to spawn and the intro replayed. Don't drive through it
        # and don't record the teleport as a 25ms step: end the episode now,
        # truncated (a cutoff, not a crash), on the last frame we trust.
        # reset() then waits for the car and starts everyone clean.
        if (self.steps > 2 and not self._respawning
                and rec.get("ui") in (UI_INTRO, UI_OUTRO)):
            self.pad.act(0.0, 0.0, 0.0)
            info = {"aborted": True, "instance": self.instance,
                    "max_distance": self.max_s, "not_ready": True}
            return self._last_obs, 0.0, False, True, info

        # An empty record mid-episode is not a car at the origin. The plugin
        # emits a heartbeat with no position whenever the viewed vehicle state
        # is null - between rounds, during a respawn, and routinely in
        # splitscreen where the camera is on one seat at a time. Reading it as
        # a position teleports the car to [0,0,0], which measures as hundreds
        # of metres off the line and ends the episode on step four.
        #
        # Holding the previous frame is the honest response: the car has not
        # moved, we simply were not told where it is.
        if not _real_pos(rec.get("pos")):
            self.empty_records += 1
            if self.prev_pos is None:
                raise RuntimeError("no car position yet this episode")
            held = dict(rec)
            held["pos"] = self.prev_pos.tolist()
            held["speed"] = 0.0
            held["vel"] = [0.0, 0.0, 0.0]
            rec = held

        obs, s, offset = self._observe(rec)
        self.steps += 1
        self.total_steps += 1

        # Progress along the reference line is the primary signal; it rewards
        # getting round the track rather than merely going fast.
        #
        # Pay only for NEW forward ground - a gain in the furthest arc length
        # reached this episode - not raw step-to-step delta. project() is a
        # global nearest-point search, and a real road ribbon folds back on
        # itself, so a car jittering near one fold has its arc position flip
        # between nearby and far samples; rewarding that delta let a parked car
        # farm +80 of "progress". Reaching genuinely new line cannot be
        # oscillated, and dawdling is already charged by the time cost.
        # Over a jump patch the car is meant to be airborne and off the chord
        # line; in the run-up it needs to be carrying speed. Gate reward terms
        # below, and widen the progress clamp - a leap legitimately advances
        # the arc by a whole gap's worth in one step.
        over_jump = self._over_jump(self.max_s) or self._over_jump(s)
        jump_runup = self._jump_runup(self.max_s)
        _prev_max_s = self.max_s

        gain = s - self.max_s
        # Normally a >50m single-step gain is a projection fold / respawn warp
        # and is zeroed. Across a jump it is the real distance flown, so pay it
        # (up to a gap's worth) and DO advance max_s - otherwise clearing the
        # gap earns nothing and the car has no reason to commit.
        ceiling = 150.0 if over_jump else 50.0
        if gain > ceiling:
            gain = 0.0               # and do NOT let it poison max_s
        else:
            self.max_s = max(self.max_s, s)
        progress = max(0.0, gain)
        self.prev_s = s
        # Steps since the car last reached genuinely new ground. Used to gate
        # rewards that a stalled car could otherwise farm in place - e.g. the
        # speedslide term: slide -> green -> collect, brake, slide again.
        self._steps_since_progress = 0 if progress > 0 else \
            self._steps_since_progress + 1

        speed = rec.get("speed", 0.0)
        pos = np.asarray(rec.get("pos", [0, 0, 0]), dtype=np.float64)
        cp_before = len(self.cp_taken)
        cp = self._count_cp(pos, rec.get("race_time"))

        # Picking up edits made in the panel while this run is going. The check
        # self-throttles to once a second, so it is a stat() in the control
        # loop, not a read.
        if self.cfg.maybe_reload():
            self._apply_config()
            print(f"  config reloaded (gen {self.cfg.generation})", flush=True)

        # Every term named, so the WHY log can say which one is actually
        # driving behaviour instead of just reporting the total.
        # The time charge. par_speed above 0 replaces the flat step cost with
        # one measured in the same units as progress, so the two terms sum to
        # w_progress * dt * (speed_along_line - par_speed): at par the step is
        # worth nothing, above par it pays, below par it costs. That is the
        # ratio between "get further" and "stop dawdling", set directly
        # instead of falling out of two unrelated constants.
        time_cost = self.step_cost
        if self.par_speed > 0:
            time_cost += self.w_progress * self.par_speed * self.dt
        self.time_cost = time_cost

        # Pay MORE PER METRE across unbarriered sections.
        #
        # A PER-TRACK SCAFFOLD, NOT A GENERAL RULE. w_platform defaults to 0.0
        # and is set only in configs/<map>.explore.json, because the goal is a
        # driver that handles whatever surface the track puts under it - not one
        # paid a premium for platforms. It exists to unstick ONE learned
        # behaviour on ONE track: the policy stopping dead at the entrance to
        # the unbarriered run between CP4 and CP5. Turn it off (or leave it
        # unset) once the crossing is learned, and never promote it into
        # env/config.py DEFAULTS.
        #
        # The policy learned to stop just before the platform run, and that was
        # rational: past CP4 the only outcomes it had ever seen were falling off
        # and being charged for it, so parking at the entrance banked the lap's
        # progress for a flat stuck penalty. Lowering the off-road charge
        # restored the option; this pays for taking it.
        #
        # Scaled BY PROGRESS rather than paid per step on purpose - a standing
        # bonus for being on a platform is farmable by sitting on one, which is
        # the failure mode we are already trying to leave.
        plat = self._platform_factor() if self.w_platform else 0.0
        parts = {
            "progress": self.w_progress * progress,
            "step_cost": -time_cost,
            "off_line": -self.w_soft * max(0.0, offset - self.soft_offset),
        }
        if plat and progress:
            parts["platform"] = self.w_platform * progress * plat
        # Lateral distance from the line, for any marker that declares a
        # max_offset. Taken here rather than stored in _observe because the
        # reward is the only consumer and this is where `offset` is in scope.
        self._last_offset = float(offset)
        mk = self._marker_reward()
        if mk:
            parts["marker"] = mk
        # Taking a checkpoint pays once, the first time.
        #
        # Progress along the line is a smooth signal and it is the main one,
        # but it is measured against a line that may be provisional - in
        # explore mode it runs straight through scenery - so it can reward
        # heading roughly the right way while the car misses the gate it
        # actually has to go through. A checkpoint is the game's own
        # definition of having got somewhere, and it cannot be farmed: the
        # set only grows, so re-crossing one pays nothing.
        if self.cp_bonus and cp > cp_before:
            parts["checkpoint"] = self.cp_bonus * (cp - cp_before)

        # Dense checkpoint-approach shaping, independent of the reference line.
        # Reward is w_cp_approach per metre the car closed on the next gate's
        # centre this step. The baseline is re-seeded whenever the target
        # changes (a gate was just taken) or the car jumps by >50m (respawn /
        # map restart), so neither of those scores as a giant step.
        #
        # One guard, because this term is a coordinate proxy: its cumulative
        # payout per episode is clamped to [0, cp_bonus], so it can never
        # outweigh actually taking the checkpoint. (A race_time>0 gate was
        # tried and removed - race_time reads 0 for ~half the seat-reads in
        # splitscreen, so it silently zeroed the shaping on those episodes.)
        if self.w_cp_approach:
            tgt = self._next_gate_pos()
            if tgt is not None:
                d_now = float(np.linalg.norm(pos - tgt))
                if (self._prev_cp_dist is None or cp > cp_before
                        or abs(self._prev_cp_dist - d_now) > 50.0):
                    self._prev_cp_dist = d_now
                step_pay = self.w_cp_approach * (self._prev_cp_dist - d_now)
                self._prev_cp_dist = d_now
                room_up = max(0.0, self.cp_bonus - self._cp_approach_paid)
                step_pay = min(step_pay, room_up)
                step_pay = max(step_pay, -self._cp_approach_paid)
                self._cp_approach_paid += step_pay
                parts["cp_approach"] = step_pay

        mats = rec.get("mat") or []
        surface_names = [material_name(m) for m in mats]
        # Which grip class most wheels are on. Ties and an empty read fall
        # back to "road", i.e. "keep the anti-twitch penalty on".
        grp = "road"
        if mats:
            gi = [group_index(m) for m in mats]
            grp = GROUP_NAMES[max(set(gi), key=gi.count)]

        if (self.cfg.enabled("weave") and (self.w_weave or self.w_reversal)
                and grp not in _SLIDE_GROUPS):
            # Brute-forcer flutter: high-frequency steering that averages out to
            # nothing but costs grip. Penalise the change, and penalise a full
            # sign flip harder, since that is the signature of the twitch.
            # Skipped on slide surfaces (see _SLIDE_GROUPS) where the wiggle is
            # the technique, not the bug.
            reversal = (steer * self.prev_steer < 0
                        and abs(steer) > 0.3 and abs(self.prev_steer) > 0.3)
            parts["weave"] = -(self.w_weave * abs(steer - self.prev_steer)
                               + self.w_reversal * (1.0 if reversal else 0.0))
        self.prev_steer = steer
        if surface_names and not self.seen_materials.issuperset(surface_names):
            self.seen_materials.update(surface_names)
            self._write_materials()
        if self.cfg.enabled("surfaces") and surface_names and not over_jump:
            # Per WHEEL, so one tyre clipping the grass costs a quarter of what
            # putting the whole car on it does.
            #
            # Skipped entirely over a jump patch: whatever the wheels graze
            # mid-flight (scenery below, the far lip) is not a shortcut the car
            # chose, and charging it teaches the policy to not take the gap.
            #
            # An explicit per-material weight wins; everything else that is not
            # in the `road` grip group falls back to `_non_road`. Without that
            # fallback only the four listed materials were charged and a
            # shortcut over Sand, Gravel, Rock, Snow, Green or Dirt earned full
            # progress reward for free.
            nr = self.cfg.non_road_weight()
            pen = 0.0
            for n in surface_names:
                w = self.cfg.surface_weight(n)
                if w:
                    pen += w
                elif nr and not is_road_name(n):
                    pen += nr
            if pen:
                parts["surface"] = pen

        if self.cfg.enabled("turbo_bonus") and self.w_turbo_use and rec.get("turbo"):
            # Carrot, not rule. It is rewarded for using a boost it is already
            # on; it is never told it must take one. Lap time stays dominant,
            # so a detour to farm a booster still loses.
            parts["turbo_use"] = self.w_turbo_use * max(0.0, gas)

        if self.w_air and not rec.get("ground") and not over_jump:
            parts["air"] = -self.w_air

        # Take-off speed bonus: a ONE-OFF at the lip of each jump, scaled by
        # the speed the car crosses it at. NOT per-step over the run-up - that
        # integrates to a speed-independent constant (per-step reward is
        # proportional to speed, time-in-zone is inversely proportional, they
        # cancel), so it paid the same for creeping as for charging. This pays
        # only for how fast you actually leave the ramp.
        if self.w_jump_speed and self._jump_s:
            for lo, hi in self._jump_s:
                if _prev_max_s < lo <= self.max_s and lo not in self._jump_taken_at:
                    self._jump_taken_at.add(lo)
                    frac = min(1.0, (speed * 3.6) / max(self.jump_target_kmh, 1.0))
                    parts["jump_speed"] = parts.get("jump_speed", 0.0) \
                        + self.w_jump_speed * frac

        # -- gears --------------------------------------------------------
        #
        # Paid for the gear it is IN, but only once it has held that gear for
        # a moment. That single condition is what stops the flip-flop: an
        # oscillation between two gears never accumulates hold time, so it
        # never earns, and there is nothing to farm by bouncing off a shift
        # point. Committing to the higher gear is the only way to collect.
        gear = int(rec.get("gear", 0) or 0)
        if self.cfg.enabled("gear") and (self.w_gear or self.w_downshift):
            if gear == self.gear:
                self.gear_held += 1
            else:
                if self.w_downshift and gear < self.gear and gear > 0:
                    parts["downshift"] = -self.w_downshift
                self.gear_held = 0
            # Gear 1 (crawling off the line) earns NOTHING. From gear 2 up it
            # is a real, linearly-ramping reward - gear 2 already scores well,
            # gear 5 pays the full w_gear.
            #   frac = (gear-1)/(top-1)  -> 0 at gear 1, .25 at gear 2, 1 at top
            g = min(gear, self.top_gear)
            if (self.w_gear and g >= 2
                    and self.gear_held >= self.gear_hold_steps):
                frac = (g - 1) / max(self.top_gear - 1, 1)
                parts["gear"] = self.w_gear * frac
        self.gear = gear

        # -- throttle and acceleration ------------------------------------
        #
        # Two separate carrots. w_gas pays for holding the pedal down at all,
        # which is the "just accelerate in a straight line" prior a person
        # tries first and that a uniform-random warm-up produces only in
        # flickers. w_accel pays for the metres per second actually gained, so
        # holding the throttle while pinned against a wall earns the first and
        # not the second.
        dv = speed - self.prev_speed
        # Smoothed acceleration, for the speedslide term below: a real SD gains
        # speed through the slide, so a slide that only holds (or bleeds)
        # speed should not pay like one that accelerates.
        self._dv_ema = 0.85 * self._dv_ema + 0.15 * dv
        if self.cfg.enabled("accel"):
            if self.w_gas and gas > 0:
                parts["gas"] = self.w_gas * gas
            if self.w_accel and dv > 0:
                parts["accel"] = self.w_accel * min(dv, 5.0)
            if self.w_both_pedals and gas > 0.5 and brake > 0.5:
                parts["both_pedals"] = -self.w_both_pedals
        self.prev_speed = speed

        # -- speedslide ---------------------------------------------------
        #
        # Above the threshold speed a controlled slide is the fast line, not a
        # mistake, so the target is a band rather than "as little slip as
        # possible". Below the threshold this term does nothing at all: at
        # 150km/h the same slide is just scrubbing speed off.
        self.sd_grade = "none"
        self.sd_score = 0.0
        if self.cfg.enabled("speedslide") and (self.ss_w or self.ss_w_blue):
            # The front-left wheel, because that is the one SDHelper switches
            # on. Which wheel it is barely matters on a uniform surface and
            # matching the helper matters more than averaging four.
            fl = surface_names[0] if surface_names else "Asphalt"
            grade, score, _det = sd_evaluate(
                rec.get("side_speed", 0.0) or 0.0, speed, fl,
                floor_kmh=self.ss_floor)
            self.sd_grade, self.sd_score = grade, score
            if grade == "green":
                self._green_streak += 1
            else:
                self._green_streak = 0
            # A slide only counts if the car is still getting down the track.
            # Slide-brake-slide in place makes no new ground, so it pays
            # nothing - which is what kills the farm.
            if self._steps_since_progress > self.ss_stall_steps:
                score = 0.0
                self._green_streak = 0
            if self.ss_w and score:
                # floor for ANY slide, then a convex ramp to ss_w at full green
                shaped = score ** max(self.ss_gamma, 0.1)
                r = self.ss_w_any + (self.ss_w - self.ss_w_any) * shaped
                # STAYING in green pays far more than touching it
                if grade == "green" and self.ss_streak_gain:
                    mult = 1.0 + self.ss_streak_gain * math.log1p(self._green_streak)
                    r *= min(mult, self.ss_streak_cap)
                # A real SD ACCELERATES. Scale by smoothed dv: a slide that
                # only holds speed pays the floor fraction, one that bleeds
                # speed pays nothing, one that pulls pays full.
                accel_f = self.ss_accel_floor + self._dv_ema * self.ss_accel_gain
                r *= max(0.0, min(1.0, accel_f))
                parts["speedslide"] = r
            elif self.ss_w_blue and grade == "blue":
                parts["speedslide"] = -self.ss_w_blue

        # -- hints --------------------------------------------------------
        #
        # Paid for a brake TAP, not for holding the brake: the pulse is the
        # whole point, and a term that pays per step for a pedal being down
        # would be maximised by standing on it. So the clock starts when the
        # pedal goes down and the payment stops when hold_ms is up, whether or
        # not the policy lets go.
        hint = active_hint(self.hints, speed, s, cp) if self.hints else None
        self.hint_name = hint.name if hint else ""
        pedal = brake if (hint is None or hint.control == "brake") else gas
        pressed = pedal > 0.5
        if pressed and not self.hint_pressed:
            self.hint_started = self.steps
        self.hint_pressed = pressed
        if hint is not None and hint.w and pressed:
            held_ms = (self.steps - self.hint_started) * self.dt * 1000.0
            if held_ms <= hint.hold_ms:
                parts["hint"] = hint.w

        # What the warm-up driver *should* do here, published for the trainer
        # to pick up. The environment is the only thing that knows arc length
        # and checkpoint count, and under SubprocVecEnv it is in a different
        # process from the driver - so instead of teaching the driver to see
        # them, the env says what it wants and the answer rides back in
        # `info`. One control step of latency, which at 40Hz is 25ms.
        self.hint_action = [None, None, None]
        if hint is not None:
            forced = hint.forced()
            for i, key in enumerate(("steer", "gas", "brake")):
                if key in forced:
                    self.hint_action[i] = float(forced[key])
            axis = 2 if hint.control == "brake" else 1
            if hint.hold_ms and self.hint_action[axis] is None:
                press = self.tapper.should_press(
                    hint, self.steps * self.dt * 1000.0)
                self.hint_action[axis] = 1.0 if press else -1.0
        else:
            self.tapper.reset()

        self._check_input_fidelity(rec, steer, gas)

        self.prev_action = np.asarray(
            [steer, gas * 2 - 1, brake * 2 - 1], dtype=np.float32)

        self.trace.append([
            rec.get("race_time"), round(float(pos[0]), 2), round(float(pos[1]), 2),
            round(float(pos[2]), 2), round(float(speed), 2),
            round(steer, 3), round(gas, 3), round(brake, 3), cp,
        ])

        terminated = False
        truncated = False
        finished = bool(rec.get("finished"))

        race_time = rec.get("race_time")

        # --- sector curriculum ------------------------------------------
        #
        # Ending at the target gate is what makes drilling a sector cheaper
        # than a lap, and the bonus is what stops a partial lap reading as a
        # failure: without it the policy would be paid to abandon the sector,
        # because a finish bonus it can no longer reach is a finish bonus it
        # stops chasing.
        sector_done = False
        overshot = False
        if self.sector_exit is not None and self.sector_exit_s is None:
            self._compute_sector_stop()
        if self.sector_exit is not None and not finished:
            # Note the moment the sector was ENTERED, so it can be timed
            # without the game's split - which we deliberately never earn,
            # because earning it means crossing the gate.
            if self._sector_entry_t is None and cp >= self.sector_exit - 1:
                self._sector_entry_t = race_time
            if self.sector_exit_s is not None and self.max_s >= self.sector_exit_s:
                sector_done = True
                terminated = True
                self._sector_respawn = True
            elif cp >= self.sector_exit:
                # Overshot the stop line and took the gate anyway. Respawn now
                # lands on the EXIT gate, so the next attempt would silently
                # drill the following sector. Fall back to a full reset to
                # resync rather than quietly train the wrong thing.
                sector_done = True
                overshot = True
                terminated = True
                self._sector_respawn = False

        # Surfaces that end the run outright. This is the honest way to express
        # "don't crash into the barriers": a -500 per-step penalty would swamp
        # every other term and teach the policy that the reward is broken,
        # whereas ending the episode says "that was a crash". Empty the list
        # later to let it hug the edges.
        bad = self.cfg.terminate_surfaces()
        if bad and surface_names and not over_jump:
            if bad.intersection(surface_names):
                self.bad_surface_steps += 1
            else:
                self.bad_surface_steps = 0
        elif over_jump:
            self.bad_surface_steps = 0

        # Over a jump the line is a straight chord and the car flies an arc, so
        # it is legitimately far from the line - widen the tolerance rather
        # than end the episode for doing the jump right.
        eff_max_offset = self.max_offset * (4.0 if over_jump else 1.0)

        if finished:
            parts["finish"] = self.finish_bonus
            terminated = True
        elif self.bad_surface_steps >= self.surface_grace:
            parts["surface_term"] = -self.off_line_penalty
            terminated = True
        elif offset > eff_max_offset:
            parts["off_line_term"] = -self.off_line_penalty
            terminated = True
        else:
            # Don't arm the stuck detector until the car has actually got
            # going once. Every episode starts stationary on the line, so
            # counting from step 0 killed the episode before the car could
            # possibly accelerate.
            if speed >= self.stuck_speed:
                self.moved = True
            self.slow_for = self.slow_for + 1 if speed < self.stuck_speed else 0
            if self.moved and self.slow_for >= self.stuck_steps:
                parts["stuck"] = -self.stuck_penalty
                terminated = True
            # WEDGED BUT MOVING. The check above only sees standing still, so a
            # car jammed against a wall that rocks back and forth - throttle,
            # reverse, throttle - keeps speed above the threshold and resets
            # the counter every step. Those episodes run to the full cap: ~88s
            # at 20Hz is 1760 transitions of a car achieving nothing, and they
            # dominate the buffer precisely because they last longest.
            #
            # Progress along the line is the honest test. max_s only ever
            # increases, so "no new ground for N seconds" catches the wedge
            # without a speed threshold to fool. Generous by default, because a
            # car legitimately reversing to recover a corner must be allowed to.
            elif self.moved and self.no_progress_steps > 0:
                if self.max_s > self._stall_ref_s + self.no_progress_m:
                    self._stall_ref_s = self.max_s
                    self._stalled_for = 0
                else:
                    self._stalled_for += 1
                    if self._stalled_for >= self.no_progress_steps:
                        parts["stuck"] = -self.stuck_penalty
                        terminated = True

        # An episode that ends in failure is charged for the time it did not
        # use. Without this, a per-step time cost makes crashing PROFITABLE:
        # ending at step 100 of 1800 avoids 1700 steps of charge, which is
        # worth far more than any crash penalty, and the policy learns to
        # drive into the nearest wall to stop the bleeding. A finish is never
        # charged - getting there sooner is the whole objective.
        if sector_done:
            # Same weight as a finish: completing the sector IS the objective
            # for this phase, so it must not be scored as a failed lap.
            parts["sector"] = self.finish_bonus
        if terminated and not finished and not sector_done \
                and self.charge_unused_time:
            unused = max(0, self.max_steps - self.steps)
            if unused:
                parts["unused_time"] = -self.time_cost * unused

        reward = float(sum(parts.values()))
        for k, v in parts.items():
            self.ep_parts[k] = self.ep_parts.get(k, 0.0) + v

        if self.steps >= self.max_steps:
            truncated = True

        info = {"race_time": race_time, "offset": offset, "speed": speed,
                "cp": cp, "cp_total": len(self.gates) if self.gates else 0,
                "distance": s, "start_distance": self.start_s,
                "max_distance": self.max_s, "instance": self.instance,
                "overruns": self.overruns,
                # Overruns THIS EPISODE. `overruns` is cumulative for the life
                # of the env and is never reset, so dividing it by one
                # episode's step count overstates the rate by however many
                # episodes have already run - which read as "slip climbing from
                # 1% to 25%" when the true rate was under 1% throughout.
                "overruns_ep": self.overruns - self._overruns_at_reset,
                "empty_records": self.empty_records,
                "not_ready_steps": self.not_ready_steps,
                # Named so the panel and the WHY log can say *which* slide
                # grade and *which* hint were live, not just that the term
                # fired. "the speedslide term paid 0.3" is unactionable;
                # "yellow, 8km/h under the green band" is not.
                "sd_grade": self.sd_grade, "sd_score": round(self.sd_score, 3),
                "hint": self.hint_name, "hint_action": self.hint_action,
                "applied_ratio": round(self.applied_ratio, 3),
                # Which generation of the tuning config scored this step. The
                # replay buffer keeps transitions scored under older ones, and
                # that mixture is a real cause of "it suddenly got worse".
                "cfg_gen": self.cfg.generation}
        if sector_done:
            # Timed from entering the sector to the stop line - our own clock,
            # since the game's split for the exit gate is never earned. It is
            # a consistent measure across attempts, which is all a "better than
            # last time" comparison needs.
            if self._sector_entry_t is not None and race_time is not None:
                info["sector_time"] = round(
                    (race_time - self._sector_entry_t) / 1000.0, 3)
            info["sector_exit"] = self.sector_exit
            info["sector_overshot"] = overshot
            info["reason"] = "SECTOR-OVER" if overshot else "SECTOR"
        elif finished:
            info["finished"] = True
        elif terminated:
            info["reason"] = ("surface" if "surface_term" in parts else
                              "off_line" if "off_line_term" in parts else "stuck")
        elif truncated:
            info["reason"] = "timeout"

        # The gate test needs where the car was as well as where it is, so
        # this is the last thing the step does.
        self.prev_pos = pos

        if terminated or truncated:
            # Start the respawn NOW, not when reset() is called.
            #
            # SB3's step_wait waits for every worker, and a worker whose
            # episode ended does its reset inside that same call - so the
            # whole batch blocks while one car gives up and waits to respawn.
            # Every other car is latched on its last input for the duration
            # and receives no new actions, which makes their recorded
            # transitions describe something that did not happen.
            #
            # The press itself is instant; only the waiting is slow. Issuing
            # it here means the respawn is already in flight by the time
            # reset() runs, so the barrier shrinks to whatever is left of it
            # rather than the whole thing - and every car stays independent,
            # which synchronising their episodes would have cost.
            self._begin_reset(rec)
            info["reward_parts"] = dict(self.ep_parts)
            # Section times, in the game's own milliseconds. `splits[i]` is
            # the clock at checkpoint i; the section times are the successive
            # differences, which is what tools/splits.py reports on.
            info["splits"] = list(self.splits)
            self._write_splits(race_time, finished)
            self._on_episode_end(s, race_time, finished)
            # Move the episode cap with the ROLLING frontier - the furthest
            # checkpoint reached in the last few episodes, not the run's all-
            # time best. One fluke CP2 no longer grants +grant_per_cp forever;
            # a stalled policy has its cap fall back toward the base.
            self._recent_cp.append(len(self.cp_taken))
            frontier = max(self._recent_cp)
            if frontier != self.max_cp_ever:
                self.max_cp_ever = frontier
                new_cap = self._episode_cap_steps()
                if new_cap != self.max_steps:
                    self.max_steps = new_cap
                    print(f"  episode cap -> {new_cap / self.control_hz:.0f}s "
                          f"(rolling frontier CP{frontier})", flush=True)

        # Kept so a non-blocking reset has something valid to hand back.
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def _write_materials(self) -> None:
        """Publish the materials this map has actually put under the wheels.

        The tuning UI otherwise has to offer all 81 EPlugSurfaceMaterialId
        entries, most of which are ShootMania leftovers that no Trackmania
        track contains, and there is no way to tell from the outside which six
        of them this track uses. Written at most once every few seconds and
        only when the set actually grew.
        """
        now = time.time()
        if now - self._materials_written < 5.0:
            return
        self._materials_written = now
        path = os.path.join(ROOT, "maps", f"{self.map_uid or 'unknown'}.materials.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"map": self.map_uid,
                           "materials": sorted(self.seen_materials)}, f)
            os.replace(tmp, path)
        except OSError:
            pass

    def _on_episode_end(self, dist: float, race_time: float | None,
                        finished: bool) -> None:
        improved = False
        if finished and race_time:
            if self.best_time is None or race_time < self.best_time:
                self.best_time = race_time
                improved = True
                # The game autosaves the replay a beat after the finish screen
                # appears; tell the watcher whose it is before it lands.
                if self.watcher is not None:
                    self.watcher.note(self.episodes, self.total_steps, race_time)
        if dist > self.best_dist + 5.0:
            self.best_dist = dist
            improved = True
        if improved:
            self._write_trace(dist, race_time, finished)

    def close(self):
        if self.watcher is not None:
            self.watcher.stop()
        self.pad.close()
        self.telem.close()

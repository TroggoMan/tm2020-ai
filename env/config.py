"""Per-map tuning config, re-read while training runs.

Every knob used to be a CLI flag, which meant changing one cost a restart, and
a restart costs the warm-up all over again. This holds them in a JSON file per
map that the environment stats on a timer and reloads when it changes - so you
can watch a run, see it clip the same barrier every lap, raise the grass
penalty, and have the next step already using the new number.

Two rules keep that safe:

  * Only things that can change mid-episode live here. Anything that would
    change the *shape* of the observation (which materials are grouped, what
    goes in the vector) is fixed at construction, because changing it under a
    trained model would silently feed it garbage.
  * The file is merged over DEFAULTS rather than replacing them, so a config
    written today still works after new keys are added, and a hand-edited file
    with a typo loses one setting instead of crashing the run.
"""
from __future__ import annotations

import copy
import json
import os
import time

DEFAULTS: dict = {
    "episode": {
        # control_hz is NOT hot-reloadable: it sets the action repeat the model
        # was trained against. Changing it mid-run changes the meaning of every
        # transition already in the buffer.
        # The episode cap GROWS with the track. max_episode_s is what a car
        # that has never reached a checkpoint gets - keep it short so early
        # flailing resets fast. Each checkpoint the run has EVER reached adds
        # grant_per_cp seconds, up to episode_ceiling. So the cap always sits
        # a bit past the current frontier: a car scraping through the back
        # half at 40s/checkpoint is not truncated before it can finish, and a
        # car that cannot clear CP1 is not left idling for two minutes.
        "max_episode_s": 75.0,
        "grant_per_cp": 30.0,
        "episode_ceiling": 210.0,
    },
    "reset": {
        "mode": "giveup",
        "button": "b",
        "finish_button": "a",
        "hold_ms": 250.0,
        "settle_ms": 400.0,
        "retries": 4,
        # On the finish screen the give-up button retries instantly, but only
        # once the screen has actually accepted input - pressing too early is
        # how resets got eaten. "Improve" (the finish_button) always works, but
        # it goes through the full restart WITH the ~8s intro, which is the
        # whole cost give-up exists to avoid. So: give-up first, Improve only
        # after this many failed attempts.
        "finish_fallback_after": 2,
        # Splitscreen only: if a seat still has not respawned after this many
        # seconds of mashing give-up, escalate to a full RequestRestartMap.
        # That resets every seat, but the other seats catch the intro and
        # abort their episode cleanly - far cheaper than one seat wedged at
        # the start for a minute with its episode clock still running. 0 turns
        # the escalation off (hold neutral and keep retrying give-up forever).
        "restart_after_s": 12.0,
        # A give-up respawn lands in well under a second, so a failed attempt
        # is cheap to retry and should not sit on a long timeout.
        "quick_timeout": 1.5,
        # Improve replays the intro, so the clock takes far longer to reset.
        # Too short here and the ladder retries on top of a reset that is
        # already working, which is exactly what made resets look slow.
        "finish_timeout": 12.0,
        # Any button skips the intro fly-in, so there is no reason to sit
        # through it on the paths that do replay it (Improve, and a full
        # restart). Turning this off is only useful for watching a reset
        # happen at normal speed.
        "skip_intro": True,
        "skip_button": "a",
        "skip_interval_ms": 250.0,
    },
    "stuck": {
        "speed": 1.0,
        "seconds": 5.0,
    },
    "line": {
        # All three are LATERAL DISTANCE FROM THE REFERENCE LINE, in metres.
        # None of them has anything to do with drifting the car: sliding is
        # rewarded or punished only through lap time, and per-wheel slip is an
        # observation input, never a reward term. A speedslide that carries
        # speed is strictly better here, because it finishes sooner.
        "max_offset": 30.0,      # metres off the line before the episode ends
        "soft_offset": 8.0,      # free corridor; no penalty inside it
        # Per metre per step outside the corridor. Deliberately tiny: the
        # reference line came from a human lap, so leaning on this hard would
        # cap the policy at copying that lap instead of beating it.
        "w_soft": 0.01,
        # How a checkpoint is credited.
        #
        # "gate" is the real thing: the car has to CROSS THE PLANE of the
        # checkpoint, inside its width and within a grid level of its height -
        # which is what the game itself requires. "sphere" is the old
        # be-somewhere-near-it test, kept only because it is what every trace
        # recorded before this change was scored with.
        #
        # Proximity alone credited a checkpoint for driving PAST one on the
        # road alongside it, which then made the reward look achievable
        # without ever going through the gate.
        "cp_mode": "gate",
        # Half the gate's width, metres. A checkpoint block is 32m wide, so 16
        # is the block; a couple of metres of slack costs nothing because the
        # plane test is already doing the real work.
        "cp_half_width": 18.0,
        # How far above or below the gate still counts. One grid level is 8m,
        # so this rejects a road stacked directly over the checkpoint.
        "cp_height": 10.0,
        # Only used by cp_mode "sphere".
        "cp_radius": 20.0,
        # Checkpoints are not all one size and the game exposes no dimensions
        # at all, so the two settings above are the default and this names the
        # exceptions, by ordered gate index (the same numbers the checkpoint
        # counter and the split times use):
        #
        #   "cp_overrides": {"2": {"half_width": 48, "height": 16}}
        #
        # The env prints each gate's measured width at startup - a gate that
        # never counts is usually a sizing problem, not a driving one.
        "cp_overrides": {},
        # TM2020 does not fix checkpoint order unless the mapper linked them,
        # so by default any untaken gate counts when you reach it. Turn this
        # on for a map that really does require a sequence.
        "cp_strict_order": False,
    },
    "reward": {
        "w_progress": 1.0,
        "step_cost": 0.02,
        "finish_bonus": 100.0,
        # Paid once for each checkpoint taken, the first time it is taken.
        #
        # Off by default so it never changes the reward under a run that is
        # already going - turn it on deliberately. It earns its place most in
        # explore mode, where the reference line is provisional and runs
        # through scenery, so progress along it can pay for heading roughly
        # the right way while the car misses the gate it has to cross.
        # Somewhere around a quarter of finish_bonus is a sensible start: big
        # enough to steer toward, small enough that finishing still dominates.
        "cp_bonus": 0.0,
        # Dense, reference-line-free shaping: w_cp_approach reward per metre the
        # car closes on the next uncrossed gate (then the finish). Potential-
        # based, so it cannot be farmed and does not shift the optimal policy.
        # This is the term to use on a map with no recorded lap - it replaces
        # w_progress, which needs a real line to mean anything. 0 = off.
        # 1.0 makes "drive at the next gate" worth ~1 point/m, on the same
        # scale as cp_bonus and finish_bonus.
        "w_cp_approach": 0.0,
        "off_line_penalty": 20.0,
        "stuck_penalty": 10.0,
        # Anti-twitch, both OFF by default and worth leaving off while you are
        # trying to teach slides. w_weave charges for the size of a steering
        # change; w_reversal charges for a full lock-to-lock sign flip - which
        # is also what a neoslide looks like from the outside. If you want to
        # kill brute-force flutter without killing the technique, raise w_weave
        # a little and leave w_reversal at zero.
        "w_weave": 0.0,
        "w_reversal": 0.0,
        # Carrot, not rule: reward using a boost you are already on, do not
        # mandate full throttle over one. Keep it small - lap time has to stay
        # the dominant term or you get a policy that farms boosters.
        "w_turbo_use": 0.0,
        "w_air": 0.0,            # per step with no wheel on the ground

        # --- time vs progress -------------------------------------------
        #
        # "progress" is already ARC LENGTH ALONG THE REFERENCE LINE, not
        # distance driven: cutting a corner advances it by exactly as much as
        # going the long way round, so there is no incentive to add distance.
        # What it does NOT do on its own is make being slow hurt. At 100 km/h
        # a step earns ~0.7 of progress against a step_cost of 0.02 - a 35:1
        # ratio, so crawling is barely worse than flying.
        #
        # par_speed fixes the ratio directly. Above 0 the time charge becomes
        # `par_speed * dt` in the same units as progress, so the two terms sum
        # to `w_progress * dt * (speed_along_line - par_speed)`: driving at par
        # earns nothing, faster earns, slower actively loses. For roughly the
        # 2:1 you want at a target speed, set par_speed to half of it
        # (m/s - 30 here is ~108 km/h, so 60 km/h is break-even).
        "par_speed": 0.0,
        # Charge an episode that ENDS EARLY IN FAILURE for the time it did not
        # use. Without this, a per-step time cost makes crashing profitable:
        # dying at step 100 of 1800 saves 1700 steps of charge, which is worth
        # far more than the crash penalty. A finish is never charged - beating
        # the clock is the entire objective.
        "charge_unused_time": True,

        # --- gears -------------------------------------------------------
        #
        # Reward per step for being in a high gear, scaled (gear / top_gear).
        # Held gears only: the gear has to have been the same for
        # gear_hold_steps before it pays, so an oscillation between 3 and 4
        # earns nothing at all and there is nothing to farm by flip-flopping.
        # Small on purpose - gear is a proxy for speed, and speed is already
        # paid for by progress. This is a nudge toward committing to a gear,
        # not a second speed reward.
        "w_gear": 0.02,
        "gear_hold_steps": 8,    # 0.2s at 40Hz
        "top_gear": 5,
        # Optional extra charge for dropping a gear. Leave at 0 unless you are
        # specifically fighting a lift-off habit - downshifting is correct
        # into a slow corner and punishing it teaches the car not to brake.
        "w_downshift": 0.0,

        # --- acceleration -------------------------------------------------
        #
        # w_gas pays for holding the throttle at all: the "just accelerate"
        # prior, which is what a human tries first on a new track and what a
        # uniform-random warm-up almost never does for more than a moment.
        # w_accel pays for the result - metres per second actually gained -
        # so coasting at speed earns nothing while pulling away earns.
        "w_gas": 0.01,
        "w_accel": 0.05,
        # Charged when both pedals are down at once. Physically it is just
        # slow, but it is also a common local optimum for a binary throttle.
        "w_both_pedals": 0.0,
    },
    # High-speed handling, which is a different problem from low-speed
    # handling and wants a different rule.
    #
    # Below the threshold, sliding is a mistake. Above it, a controlled slide
    # IS the fast line - a speedslide carries speed through a corner that grip
    # driving cannot hold. So the target is a BAND, not a minimum: too little
    # and you are understeering wide, too much and you are scrubbing speed off
    # in a spin.
    #
    # The band numbers are NOT ours. They come from the SDHelper plugin
    # (Tobirousch/sdhelper), which is installed in this prefix and is what
    # human SD players steer by - see env/speedslide.py. Worth being clear
    # about what SDHelper does, because it looks like it reads skidmarks and
    # it does not: it reads sideways speed and forward speed, and then swaps
    # the skidmark TEXTURE on disk so the marks come out green/yellow/orange.
    # The colour is its output. Both of its inputs are already in our
    # observation, so nothing new has to be seen for this to be learnable.
    "speedslide": {
        # Paid per step, multiplied by the 0..1 slide score: 1.0 across
        # SDHelper's green band, tapering to 0 at the edges of orange. A ramp
        # rather than four steps, because a step function gives the policy no
        # direction to move in.
        "w": 0.0,
        # Charged per step when the car is fast enough to be sliding usefully
        # and isn't - SDHelper's "blue". Leave at 0 until w alone is working;
        # two terms pulling at once is how you get a policy that spins.
        "w_blue": 0.0,
        # 0 means use SDHelper's own per-surface floor (400km/h on road,
        # 200 on grass and dirt). Override only to experiment.
        "speed_floor_kmh": 0.0,
    },
    # Is the game actually applying what we send?
    #
    # The plugin reports `in_steer` - the input the GAME received - which is
    # not necessarily the input we sent. TM2020 ramps analogue steering per
    # FRAME, so anything that cuts the frame rate means the ramp never reaches
    # full deflection: an unfocused window is the usual culprit, and it shows
    # up as the game applying ~90% of a command held at full lock.
    #
    # That matters more than it sounds. A fleet where each instance is applied
    # differently depending on which window has focus is a fleet whose
    # transitions do not mean the same thing, and the replay buffer mixes them
    # with no way to tell them apart afterwards. The structural fix is to give
    # every instance its own X display so it is always focused; this is the
    # detector that tells you when you have not.
    "input_fidelity": {
        "window": 40,          # steady-state samples before judging
        "settle_steps": 3,     # steps a command must be held before sampling
        "warn_below": 0.95,    # fraction applied that counts as a problem
    },
    # Marks: named places on the track, recorded by driving there.
    #
    # A mark is a world position, not an arc length, because arc length is a
    # property of whichever reference line is loaded and lines get re-recorded.
    # Store the place; resolve it against the current line at load time. A
    # mark then survives a straightened line and a line built by the explore
    # stage.
    #
    #   "marks": {"before-the-jump": [560.0, 10.0, 812.5]}
    #
    # Record one by driving to the spot and pressing "Mark here" in the panel,
    # or `tools/mark.py before-the-jump`. A hint can then say
    # `"mark_from": "before-the-jump"` instead of a number you had to guess.
    "marks": {},
    # Hints: "try tapping the brake here, for about this long".
    #
    # Empty by default. A hint is a nudge toward a behaviour that random
    # exploration will never stumble into - a speed slide needs a short brake
    # pulse at high speed, and the odds of SAC rolling one by accident and
    # then rolling the steering that keeps it are nil. See env/hints.py.
    #
    #   {"name": "sd-entry", "speed_from_kmh": 400, "speed_to_kmh": 600,
    #    "hold_ms": 120, "gap_ms": 700, "w": 0.05}
    #
    # `w` is small on purpose. It buys the behaviour a fair trial; lap time
    # decides whether it survives.
    "hints": [],
    # How the three action outputs become pedal positions.
    #
    # TM2020's throttle is not really analogue in the way a sim's is: on a
    # keyboard it is a switch, and the fast line is very nearly always "full
    # throttle or nothing". Letting SAC hold 0.43 gas gives it a whole
    # continuum of mediocre options to get lost in, and every one of them is
    # slower than the two ends. Thresholding at zero keeps the action space
    # continuous for the optimiser (SAC needs that) while making the car do
    # only the two things a driver actually does.
    #
    # Steering stays analogue - it genuinely is.
    "action": {
        "binary_gas": True,
        "binary_brake": True,
        "gas_threshold": 0.0,
        "brake_threshold": 0.0,
        # Steering slew-rate cap, full-deflection units per second. 0 = snap:
        # send the policy's target straight to the pad, exactly like a keyboard
        # player's key snapping to full - TM2020 then applies its OWN internal
        # wheel-angle ramp (~0.15s to lock) regardless of input source, so the
        # car is never actually jerked. Set this > 0 only to make the POLICY
        # itself smoother (double-ramps: policy ramp THEN game ramp, so the car
        # turns in slower than a keyboard player). e.g. 5 = policy reaches lock
        # in 0.2s. Default 0 because the game already does the right thing.
        "steer_rate": 0.0,
    },
    # Per-step penalty per wheel, keyed by EPlugSurfaceMaterialId name.
    # Negative numbers are penalties. This is where "road borders: -500" goes,
    # though see terminate_on_surface below for what you probably actually want.
    "surfaces": {},
    # Materials that end the episode outright. A -500 per-step penalty swamps
    # every other term and teaches the policy that the reward function is
    # broken; ending the episode says "that was a crash" cleanly. Empty the
    # list later to let it hug the edges.
    "terminate_on_surface": [],
    "surface_grace_steps": 4,    # wheels must stay on it this long to count
    "enabled": {
        "weave": False,
        "surfaces": False,
        "effects": True,
        "turbo_bonus": False,
        "gear": True,
        "accel": True,
        # Off until the car is fast enough for it to ever fire. Turning it on
        # early costs nothing but it also does nothing, and it is one more
        # term muddying the WHY log.
        "speedslide": False,
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_path(root: str, map_uid: str | None, profile: str = "") -> str:
    """One file per map *and stage*.

    Explore and race want very different numbers on the same map - explore
    opens the off-line limit to 250m because its line cuts through scenery,
    and that value leaking into a racing run would disable the limit entirely
    without anyone noticing. Keeping them in separate files means both stay
    live-tunable and neither can overwrite the other.
    """
    suffix = f".{profile}" if profile else ""
    return os.path.join(root, "configs", f"{map_uid or 'default'}{suffix}.json")


class TuningConfig:
    """One config file, watched for changes.

    Reload is by mtime, checked at most every `poll` seconds. That is a stat()
    call - cheap enough to do in the control loop, which is the point: a change
    should take effect on the next step, not the next episode.
    """

    def __init__(self, path: str, poll: float = 1.0):
        self.path = path
        self.poll = poll
        self.data = copy.deepcopy(DEFAULTS)
        self._mtime = 0.0
        self._checked = 0.0
        self.generation = 0
        self.load()

    # -- io ---------------------------------------------------------------

    def load(self) -> bool:
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        try:
            with open(self.path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as ex:
            # A half-written or malformed file must not take the run down.
            # Keep the values we already have and say so once per mtime.
            if st.st_mtime != self._mtime:
                print(f"config {os.path.basename(self.path)} unreadable: {ex}",
                      flush=True)
                self._mtime = st.st_mtime
            return False
        self.data = deep_merge(DEFAULTS, raw)
        self._mtime = st.st_mtime
        self.generation += 1
        return True

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)
        try:
            self._mtime = os.stat(self.path).st_mtime
        except OSError:
            pass

    def maybe_reload(self) -> bool:
        """True when the file changed and the new values are now live."""
        now = time.time()
        if now - self._checked < self.poll:
            return False
        self._checked = now
        try:
            mtime = os.stat(self.path).st_mtime
        except OSError:
            return False
        if mtime == self._mtime:
            return False
        return self.load()

    # -- access -----------------------------------------------------------

    def get(self, section: str, key: str, default=None):
        return self.data.get(section, {}).get(key, default)

    def enabled(self, key: str) -> bool:
        return bool(self.data.get("enabled", {}).get(key, False))

    def surface_weight(self, name: str) -> float:
        try:
            return float(self.data.get("surfaces", {}).get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0

    def terminate_surfaces(self) -> set[str]:
        v = self.data.get("terminate_on_surface") or []
        return set(v) if isinstance(v, list) else set()

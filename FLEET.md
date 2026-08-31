# Scaling to a fleet

Target: N containerised game instances, each driven by its own worker, all
feeding one SAC replay buffer, with every screen visible in one place.

## Port convention

Instance `i` owns a block, so nothing has to be discovered. The three game
ports step by **ten**, not one - they are adjacent base ports, so stepping each
by `i` walks them straight into each other: instance 1's pad would be 8766,
which is instance 0's *plugin*, and the symptom is a pad server and a game
silently fighting over one socket.

    pad server     8765 + 10i
    plugin         8766 + 10i      # set in each game's own Openplanet settings
    broker         8767 + 10i
    VNC            5900 + i
    noVNC          6080 + i

    instance 0     pad 8765   plugin 8766   broker 8767
    instance 1     pad 8775   plugin 8776   broker 8777
    instance 2     pad 8785   plugin 8786   broker 8787

Instance 0 keeps exactly the ports everything already uses, so a single-game
setup is unchanged. The mapping lives in `env/ports.py` - stdlib only, because
the pad server runs on the system python (it needs `evdev` and uinput
permission) and cannot import anything from the venv.

`tools/fleet.py` starts and supervises the pad servers and brokers;
`tools/fleet.py --check` reports which ports are actually listening. The
plugin port is the one thing neither can set: it lives in each game's own
Openplanet config, inside that game's Wine prefix.

`TrackmaniaEnv(pad_addr=..., telem_addr=...)` already takes these, and the pad
server takes `--instance i` so each game sees a *distinctly named* uinput device
(`TMAI Virtual Xbox 360 Controller #3`) rather than all binding to instance 0's.

## Container shape

    nvidia/cuda base  (needs the NVIDIA container toolkit for GPU access)
      + Xvfb            :99, a small resolution - see below
      + x11vnc          exposes :99
      + Proton/Wine     + the TM2020 install
      + Openplanet      + our TMAITelemetry.op
      + pad server      one per container, on its own uinput device
      + broker

`/dev/uinput` must be passed into the container (`--device /dev/uinput`) and the
container needs `CAP_SYS_ADMIN` or an appropriate udev rule to create devices.
This is the piece most likely to need iteration.

**Low resolution is the lever here.** It does not speed up episodes (physics is
locked to 100Hz wall-clock) but it raises how many instances fit on one GPU,
which is the actual ceiling - and it raises the telemetry rate, which is
frame-bound and currently *below* the physics tick. See the README.

## Accounts: log in once, then persist the prefix

**Do not script the Ubisoft login.** It is the most fragile possible integration
point - form changes, 2FA, captcha - and a failed login loop looks exactly like
credential stuffing from Ubisoft's side.

Instead:

1. Create each account **manually**, once.
2. Log in **once** per Wine prefix, interactively.
3. Snapshot that prefix. It contains the saved session.
4. Each container mounts its own prefix as a volume and starts already logged in.

Restarting a container then needs no login at all. This is both more robust and
less likely to get accounts flagged than any login automation.

> Worth being explicit: mass **automated account creation** is a different thing
> from automated login to accounts you own, and is the part most likely to
> breach Ubisoft's terms and get the accounts banned. Creating them by hand
> keeps the risky step human and the repeatable step automated. One account per
> concurrent instance is required either way - the same account cannot be
> logged in twice.

## School Mode: prefer never restarting

School Mode does not persist across launches and there is no config file for it,
so the cheapest fix is to **not relaunch**. A container that stays up needs it
toggled exactly once, and with give-up resets (which need no map reload) a worker
can run for days. Treat a container restart as an exceptional event, not the
episode loop.

For the exceptional case, automate the toggle by clicking it:

- Each container has its **own Xvfb display**, so `DISPLAY=:99 xdotool` clicks
  cannot collide with or steal focus from another worker. This is why the
  container path is *easier* than the desktop - on the host we are on Wayland
  with one shared display.
- Overlay geometry is stable when the window size and `OverlayScale` are fixed,
  which they are in a container.
- This is not anti-cheat evasion: School is the *more* restrictive mode, and
  automating the toggle only ever moves the game into the state that disables
  leaderboard submission.

Checkpoint/restore (CRIU) to snapshot a "already in School Mode" process is
appealing but fragile with a GPU and an X connection in the picture. Not
recommended over just keeping containers alive.

## Watching all the screens

`x11vnc` per container against its Xvfb display, then a grid of `noVNC` iframes
in the control panel. Read-only by default (`-viewonly`) so a stray click
cannot fight the policy, with a per-tile toggle to take control when you want
to fix something by hand - including the School Mode toggle.

## Training topology

N workers, one learner. Each worker runs the env and pushes transitions; the
learner owns the replay buffer and the gradient steps. This is tmrl's
architecture and SAC is off-policy, so worker data does not need to come from
the current policy.

Start with N=1 until single-instance training actually learns. Reward shaping
and termination bugs are miserable to debug across eight games at once, and
none of the fleet work is invalidated by waiting.

## Order of work

1. One container that runs the game, Openplanet in School Mode, pad + broker,
   and is drivable from the host. Everything else is repetition of this.
2. Prefix-snapshot flow so a second container is a copy plus a different volume.
3. `tools/fleet.py` to launch and supervise N of them (host side already built).
4. noVNC grid in the panel.
5. Multi-worker trainer.

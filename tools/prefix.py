#!/usr/bin/env python3
"""Per-instance Wine prefixes: clone one, point it at its own port, launch it.

Each concurrent game needs its own Wine prefix, because the prefix is where
the Ubisoft Connect session lives and one account cannot be signed in twice.
Instance 0 is your existing Steam prefix and is never modified; every other
instance is a clone of it.

    tools/prefix.py status
    tools/prefix.py clone 1 2            make instance 1 and 2
    tools/prefix.py port 1               set its Openplanet listen port
    tools/prefix.py launch 1             start that instance's game

**The clone is deliberately signed out.** `clone` removes the Ubisoft Connect
session file from the copy, so the first launch asks for a login and you sign
in by hand with that instance's account. That is the one manual step and it
stays manual: scripting the Ubisoft login is the fastest way to get every
account banned, and a failed retry loop against their auth endpoint is
indistinguishable from credential stuffing. After that first login the session
persists in the prefix and restarts need nothing.

Launching does **not** go through Steam. Steam links exactly one Ubisoft
account per Steam account, so a Steam launch would sign every instance into
the same one. Running `Trackmania.exe` under Proton directly hands
authentication to the prefix's own Ubisoft Connect, which is what makes three
different accounts possible. Trackmania's Starter Access is free, so each
account is entitled to play on its own.

Clones use `cp --reflink=auto` and live on the same XFS filesystem as the
original, so a 2.3GB prefix copies in about a second and costs no extra disk
until the copies diverge.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.ports import instance_ports                      # noqa: E402
from env.prefixes import (BASE_COMPAT, CLONE_ROOT, GAME_DIR, OPENPLANET,
                          UBI_SESSION, compat_dir, exists,
                          prefix_for)                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTON = os.environ.get(
    "TMAI_PROTON",
    "/home/troggoman/.local/share/Steam/steamapps/common/Proton - Experimental")
STEAM_CLIENT = os.path.expanduser("~/.local/share/Steam")
PLUGIN_SECTION = "[Plugin_TMAITelemetry]"


def settings_ini(i: int) -> str:
    return os.path.join(prefix_for(i), OPENPLANET, "Settings.ini")


def set_port(i: int, quiet: bool = False) -> bool:
    """Write this instance's listen port into its Openplanet settings.

    Openplanet only writes a `[Plugin_*]` section once a setting has been
    changed from its default, so on a fresh prefix the section does not exist
    and has to be created. Doing it in the file rather than through the in-game
    settings UI is the difference between one command and N sessions of
    clicking.
    """
    port = instance_ports(i)["plugin"]
    path = settings_ini(i)
    if not os.path.isfile(path):
        print(f"instance {i}: no Settings.ini at {path} - launch the game "
              f"once so Openplanet creates it, then re-run this")
        return False

    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    out, in_section, wrote = [], False, False
    for line in lines:
        if line.startswith("["):
            if in_section and not wrote:
                out.append(f"S_Port={port}")
                wrote = True
            in_section = line.strip() == PLUGIN_SECTION
        if in_section and line.strip().startswith("S_Port="):
            out.append(f"S_Port={port}")
            wrote = True
            continue
        out.append(line)
    if not wrote:
        out += ["", PLUGIN_SECTION, f"S_Port={port}"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    if not quiet:
        print(f"instance {i}: Openplanet listen port -> {port}")
    return True


def clone(i: int, force: bool = False) -> bool:
    if i == 0:
        print("instance 0 is the original prefix - nothing to clone")
        return False
    dst = compat_dir(i)
    if os.path.isdir(dst) and not force:
        print(f"instance {i}: {dst} already exists (use --force to replace)")
        return False
    if not os.path.isdir(BASE_COMPAT):
        print(f"base prefix missing: {BASE_COMPAT}", file=sys.stderr)
        return False
    if os.path.isdir(dst) and force:
        shutil.rmtree(dst)

    os.makedirs(CLONE_ROOT, exist_ok=True)
    print(f"instance {i}: cloning {BASE_COMPAT} -> {dst}", flush=True)
    # --reflink=auto: instant and free on XFS, a normal copy anywhere else.
    r = subprocess.run(["cp", "-a", "--reflink=auto", BASE_COMPAT, dst])
    if r.returncode != 0:
        print(f"instance {i}: copy failed", file=sys.stderr)
        return False

    # Sign the clone out. The session in user.dat belongs to the account that
    # created the original prefix; leaving it would put two instances on one
    # account, which the game refuses, and the failure looks like a network
    # error rather than what it is.
    sess = os.path.join(prefix_for(i), UBI_SESSION)
    if os.path.isfile(sess):
        os.remove(sess)
        print(f"instance {i}: signed out (removed user.dat) - "
              f"first launch will ask for a login")

    # A stale lock from the source prefix stops Proton starting.
    for junk in ("pfx.lock",):
        p = os.path.join(dst, junk)
        if os.path.exists(p):
            os.remove(p)

    set_port(i)
    print(f"instance {i}: ready. Next: tools/prefix.py launch {i}, "
          f"sign in as this instance's account, set the plate to TAS.")
    return True


def launch(i: int, extra: list[str] | None = None) -> int:
    compat = compat_dir(i)
    if not os.path.isdir(compat):
        print(f"instance {i}: no prefix - run `tools/prefix.py clone {i}` first",
              file=sys.stderr)
        return 1
    proton = os.path.join(PROTON, "proton")
    if not os.path.isfile(proton):
        print(f"no Proton at {proton} (set $TMAI_PROTON)", file=sys.stderr)
        return 1
    exe = os.path.join(GAME_DIR, "Trackmania.exe")
    if not os.path.isfile(exe):
        print(f"no game at {exe}", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env["STEAM_COMPAT_DATA_PATH"] = compat
    env["STEAM_COMPAT_CLIENT_INSTALL_PATH"] = STEAM_CLIENT
    # Every instance gets its own shader cache directory. Sharing one is a
    # write conflict between processes, and the symptom is a stutter that
    # looks like the control loop overrunning.
    env["DXVK_STATE_CACHE_PATH"] = os.path.join(compat, "dxvk-cache")
    env["TMAI_INSTANCE"] = str(i)
    os.makedirs(env["DXVK_STATE_CACHE_PATH"], exist_ok=True)

    ports = instance_ports(i)
    log = os.path.join(ROOT, "logs", f"game{i}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    print(f"instance {i}: launching (plugin should listen on "
          f"{ports['plugin']}) -> {log}", flush=True)
    with open(log, "a", buffering=1) as f:
        f.write(f"\n--- instance {i} launch ---\n")
        p = subprocess.Popen([proton, "waitforexitandrun", exe]
                             + list(extra or []),
                             env=env, cwd=GAME_DIR, stdout=f,
                             stderr=subprocess.STDOUT)
    print(f"instance {i}: pid {p.pid}")
    return 0


UBI_EXE = ("drive_c/Program Files (x86)/Ubisoft/Ubisoft Game Launcher/"
           "UbisoftGameLauncher.exe")


def login(i: int, flags: list[str] | None = None,
          display: str | None = None) -> int:
    """Open just Ubisoft Connect in this prefix, so the sign-in can be done
    without the game's DRM wrapper spawning it for us.

    The reason this exists: the login page is CEF, the launcher log says
    `Using CEF with native rendering`, and under XWayland that paints a black
    window with a live cursor behind it. Starting the launcher ourselves is
    the only place we can pass Chromium switches at all - the game's wrapper
    spawns it with a fixed command line we cannot touch.
    """
    compat = compat_dir(i)
    exe = os.path.join(prefix_for(i), UBI_EXE)
    if not os.path.isfile(exe):
        print(f"instance {i}: no Ubisoft Connect at {exe}", file=sys.stderr)
        return 1
    proton = os.path.join(PROTON, "proton")
    env = dict(os.environ)
    env["STEAM_COMPAT_DATA_PATH"] = compat
    env["STEAM_COMPAT_CLIENT_INSTALL_PATH"] = STEAM_CLIENT
    if display:
        # A plain X server with no GPU and no compositor. CEF's "native
        # rendering" path is what paints black on XWayland; on a bare X
        # display there is no hardware path for it to fail into. This is the
        # container plan from FLEET.md, tested without the container.
        env["DISPLAY"] = display
        env.pop("WAYLAND_DISPLAY", None)
        env["SDL_VIDEODRIVER"] = "x11"
    argv = [proton, "run", exe] + list(
        flags if flags is not None
        # --disable-gpu is the one that matters; the rest are the usual
        # companions for making Chromium paint on a compositor it dislikes.
        else ["--disable-gpu", "--disable-gpu-compositing",
              "--disable-software-rasterizer", "--in-process-gpu"])
    log = os.path.join(ROOT, "logs", f"login{i}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    print(f"instance {i}: opening Ubisoft Connect -> {log}", flush=True)
    with open(log, "a", buffering=1) as f:
        f.write(f"\n--- instance {i} login ---\n")
        p = subprocess.Popen(argv, env=env, cwd=os.path.dirname(exe),
                             stdout=f, stderr=subprocess.STDOUT)
    print(f"instance {i}: pid {p.pid}. Sign in as this instance's account, "
          f"then close it and run `launch {i}`.")
    return 0


def status(n: int) -> int:
    print(f"{'inst':>4}  {'prefix':<46} {'plugin':>7}  {'signed in':>9}  maps")
    for i in range(n):
        d = compat_dir(i)
        here = exists(i)
        port = instance_ports(i)["plugin"]
        signed = "-"
        if here:
            signed = "yes" if os.path.isfile(
                os.path.join(prefix_for(i), UBI_SESSION)) else "NO"
        maps = "-"
        if here:
            md = os.path.join(prefix_for(i),
                              "drive_c/users/steamuser/Documents/Trackmania/Maps")
            maps = str(sum(len([f for f in files if f.lower().endswith(".map.gbx")])
                           for _, _, files in os.walk(md))) if os.path.isdir(md) else "0"
        shown = d if here else f"{d}  (missing)"
        print(f"{i:>4}  {shown[-46:]:<46} {port:>7}  {signed:>9}  {maps}")
    print(f"\nclone root: {CLONE_ROOT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")
    p.add_argument("-n", "--instances", type=int, default=4)

    p = sub.add_parser("clone", help="copy the base prefix for these instances")
    p.add_argument("instances", type=int, nargs="+")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("port", help="set the Openplanet listen port")
    p.add_argument("instances", type=int, nargs="+")

    p = sub.add_parser("login", help="open Ubisoft Connect alone, to sign in")
    p.add_argument("instance", type=int)
    p.add_argument("--no-flags", action="store_true",
                   help="do not pass any Chromium switches")
    p.add_argument("--display", help="run on this X display, e.g. :99 - "
                                     "a bare Xvfb, viewed over VNC")

    p = sub.add_parser("launch")
    p.add_argument("instance", type=int)

    a = ap.parse_args()
    if a.cmd == "status":
        return status(a.instances)
    if a.cmd == "clone":
        ok = all(clone(i, a.force) for i in a.instances)
        return 0 if ok else 1
    if a.cmd == "port":
        return 0 if all(set_port(i) for i in a.instances) else 1
    if a.cmd == "login":
        return login(a.instance, [] if a.no_flags else None,
                     display=a.display)
    if a.cmd == "launch":
        return launch(a.instance)
    return 1


if __name__ == "__main__":
    sys.exit(main())

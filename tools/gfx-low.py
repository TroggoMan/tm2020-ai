#!/usr/bin/env python3
"""Set every instance's graphics to the lowest the game itself offers.

    tools/gfx-low.py                 main install + every fleet instance
    tools/gfx-low.py 1 2             only tmai01 and tmai02
    tools/gfx-low.py --res 800x450   also drop the window size
    tools/gfx-low.py --restore       put back the backup this made

Nothing here is invented. TM2020's own Default.json carries TWO display
profiles - `Display` (what you play with) and `DisplaySafe` (what safe mode
uses), and DisplaySafe already IS the lowest setting of every knob, with the
exact enum spellings the game accepts: Shadows "none", TexturesQuality
"very_low", ShaderQuality "very_fast", FilterAnisoQ "bilinear",
ReflectEverywhere "none", FxBloomHdr/FxMotionBlur/FxBlur off, Decals off.
So this copies DisplaySafe over Display instead of guessing strings that a
rejected value would silently reset.

Four deliberate departures from safe mode, because we want speed, not safety:
  DisplayMode      stays `windowed`   - headless-main drives a window on Xvfb
  ScreenSizeWin    left alone unless --res, so the grid view keeps working
  MultiThread/ThreadCountMax  stay on/4 - safe mode single-threads the engine,
                   which is a compatibility workaround, not a saving
  Automatic_Enabled -> false - the engine otherwise raises quality back up on
                   its own to chase Automatic_MinFps, undoing all of this
  MaxFps           -> the frame cap (see tools/frame-cap.sh)

THE GAME MUST BE CLOSED for that instance. Trackmania rewrites Default.json
from memory when it exits, so an edit made while it runs is clobbered - the
same trap tools/op-mode.sh documents for Openplanet's Settings.ini. This
refuses to touch a running instance rather than write a file that will vanish.
"""
import argparse, json, os, pwd, shutil, subprocess, sys, time

MAIN = ("/mnt/4TB/SteamLibrary/steamapps/compatdata/2225070/pfx/drive_c/users"
        "/steamuser/Documents/Trackmania/Config/Default.json")
FLEET_HOME = "/mnt/games/tm2020-ai-users"
MAX_FPS = int(os.environ.get("TMAI_FPS_CAP", "60") or 60)


def sh(argv, **kw):
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def as_root_read(path):
    r = sh(["sudo", "cat", path])
    return r.stdout if r.returncode == 0 else None


def as_root_write(path, text, owner):
    r = sh(["sudo", "tee", path], input=text)
    if r.returncode == 0 and owner:
        sh(["sudo", "chown", f"{owner}:{owner}", path])
    return r.returncode == 0


def game_running(user):
    return sh(["pgrep", "-u", user, "-f", "Trackmania.exe"]).returncode == 0


def apply(path, owner, res, restore):
    root = owner is not None
    bak = path + ".bak-gfxlow"
    if restore:
        if root:
            ok = sh(["sudo", "test", "-f", bak]).returncode == 0
            if ok:
                sh(["sudo", "cp", bak, path]); return "restored"
            return "no backup"
        if os.path.exists(bak):
            shutil.copy2(bak, path); return "restored"
        return "no backup"

    raw = as_root_read(path) if root else (open(path).read()
                                           if os.path.exists(path) else None)
    if raw is None:
        return "no config yet - launch the game once"
    cfg = json.loads(raw)
    if "DisplaySafe" not in cfg or "Display" not in cfg:
        return "unexpected config shape - not touched"

    keep = {k: cfg["Display"][k] for k in
            ("DisplayMode", "ScreenSizeWin", "MultiThread", "ThreadCountMax",
             "EmulateCursorGDI", "DisableZBufferRange",
             "DisableWindowedAntiAlias", "GpuSyncTimeOut")
            if k in cfg["Display"]}
    cfg["Display"] = dict(cfg["DisplaySafe"])
    cfg["Display"].update(keep)
    cfg["Display"]["MaxFps"] = MAX_FPS
    cfg["Display"]["Automatic_Enabled"] = False
    cfg["Display"]["Customize"] = True          # or the Preset overrides us
    if res:
        cfg["Display"]["ScreenSizeWin"] = res
    # Cheap wins that are plain booleans, so no enum to get wrong.
    cfg["IsSkipRollingDemo"] = True             # no attract demo on the menu
    cfg["SkipIntro"] = True
    cfg["AudioEnabled"] = False                 # kills the mixer threads
    # NOT touched: TmCarQuality, TmCarParticlesQuality, PlayerShadow,
    # PlayerOcclusion, TmBackgroundQuality. Those live outside the Display
    # block, so DisplaySafe gives no known-good spelling for them and a value
    # the game rejects is silently reset. Drop them in the in-game menu once
    # per instance if you want them; they survive in this same file.
    out = json.dumps(cfg, indent=1)

    if root:
        sh(["sudo", "cp", "-n", path, bak])
        return "lowest" if as_root_write(path, out, owner) else "write failed"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    open(path, "w").write(out)
    return "lowest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instances", nargs="*", type=int)
    ap.add_argument("--res", default=None, help='e.g. "800x450"')
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()

    targets = []
    if not a.instances:
        targets.append(("troggoman (main)", MAIN, None))
        for d in sorted(os.listdir(FLEET_HOME)):
            if d.startswith("tmai"):
                a.instances.append(int(d[4:]))
    for n in sorted(set(a.instances)):
        u = f"tmai{n:02d}"
        try:
            home = pwd.getpwnam(u).pw_dir
        except KeyError:
            print(f"{u}: no such user"); continue
        targets.append((u, f"{home}/.local/share/Steam/steamapps/compatdata"
                           f"/2225070/pfx/drive_c/users/steamuser/Documents"
                           f"/Trackmania/Config/Default.json", u))

    for name, path, owner in targets:
        user = owner or "troggoman"
        if game_running(user):
            print(f"{name}: GAME IS RUNNING - close it first, the game "
                  f"rewrites this file on exit")
            continue
        print(f"{name}: {apply(path, owner, a.res, a.restore)}")
    print(f"\nMaxFps set to {MAX_FPS} (TMAI_FPS_CAP). Takes effect next launch.")


if __name__ == "__main__":
    sys.exit(main())

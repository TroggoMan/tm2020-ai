#!/usr/bin/env python3
"""Tile every instance's screen into one video and push it to Twitch/YouTube.

    tools/stream.py --preview                     write a local file, no key
    tools/stream.py --twitch  live_xxx            stream to Twitch
    tools/stream.py --youtube xxxx-xxxx-xxxx      stream to YouTube

One ffmpeg process does all of it: grab each instance's X display, tile them
with `xstack`, and encode once. No OBS, no compositor, no desktop session -
which is the point, because the whole thing has to run headless on a server.

**It reads the Xvfb displays, not the desktop.** Each instance already runs on
its own `:10N` (see tools/steam-instance --vnc), which is also what stops the
window manager throttling an unfocused game. So the capture is a side effect
of a setup we needed anyway, and streaming costs the games nothing: with
h264_nvenc the encode happens on the GPU's dedicated encoder block, not on the
shaders the games are using and not on the CPU the learner is using.

Every frame carries **TAS**. That is a hard project constraint, not decoration:
footage of an AI driving must be self-evidently tool-assisted, and a burned-in
overlay cannot be lost the way a nameplate can.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWITCH = "rtmp://live.twitch.tv/app/{}"
YOUTUBE = "rtmp://a.rtmp.youtube.com/live2/{}"


def display_for(instance: int) -> str:
    """Same mapping tools/steam-instance uses."""
    return f":{99 + instance}"


def grid(n: int) -> tuple[int, int]:
    """Columns x rows. Four instances is the interesting case and wants 2x2."""
    if n <= 1:
        return 1, 1
    if n <= 2:
        return 2, 1
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    return 3, 3


def build(args) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", args.loglevel]

    # --single-display: grab ONE X display (the resizable browser grid at
    # /grid, run fullscreen on its own Xvfb) instead of tiling N game
    # displays here. The layout you drag in the browser is what streams.
    if args.single_display:
        cmd += ["-f", "x11grab", "-framerate", str(args.fps),
                "-video_size", f"{args.width}x{args.height}",
                "-i", f"{args.single_display}.0+0,0"]
        parts = [f"[0:v]scale={args.width}:{args.height},setsar=1[grid]"]
        return _finish(cmd, parts, args)

    cols, rows = grid(args.instances)
    tile_w, tile_h = args.width // cols, args.height // rows

    for i in range(args.instances):
        cmd += ["-f", "x11grab", "-framerate", str(args.fps),
                "-video_size", f"{args.capture_width}x{args.capture_height}",
                "-i", f"{display_for(i)}.0+0,0"]

    # Scale each input to its tile, then stack. xstack needs every input to be
    # exactly its tile size or it refuses, so the scale is not optional.
    parts = [f"[{i}:v]scale={tile_w}:{tile_h},setsar=1[t{i}]"
             for i in range(args.instances)]
    if args.instances == 1:
        layout = "[t0]copy[grid]"
    else:
        pos = []
        for i in range(args.instances):
            c, r = i % cols, i // cols
            pos.append(f"{c * tile_w}_{r * tile_h}")
        ins = "".join(f"[t{i}]" for i in range(args.instances))
        layout = (f"{ins}xstack=inputs={args.instances}"
                  f":layout={'|'.join(pos)}[grid]")
    parts.append(layout)
    return _finish(cmd, parts, args)


def _finish(cmd: list[str], parts: list[str], args) -> list[str]:
    # The TAS stamp. drawtext needs a font file; fall back to a plain box if
    # none is found rather than failing the whole stream over a label.
    font = next((f for f in (
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ) if os.path.isfile(f)), None)
    if font:
        parts.append(
            f"[grid]drawtext=fontfile={font}:text='TAS':fontcolor=white@0.85:"
            f"fontsize={max(18, args.height // 30)}:box=1:boxcolor=black@0.5:"
            f"boxborderw=8:x=w-tw-24:y=24[out]")
        last = "[out]"
    else:
        print("no DejaVu font found - streaming without the TAS stamp is NOT "
              "acceptable for this project; install ttf-dejavu", file=sys.stderr)
        return []

    cmd += ["-filter_complex", ";".join(parts), "-map", last]

    if args.encoder == "auto":
        enc = ("h264_nvenc" if _has_encoder("h264_nvenc") else "libx264")
    else:
        enc = args.encoder
    cmd += ["-c:v", enc]
    if enc == "h264_nvenc":
        # p4/ll: low latency preset. The 2070's encoder block is idle while
        # the games use the shaders, so this is close to free.
        cmd += ["-preset", "p4", "-tune", "ll", "-rc", "cbr"]
    else:
        cmd += ["-preset", "veryfast", "-tune", "zerolatency"]
    cmd += ["-b:v", args.bitrate, "-maxrate", args.bitrate,
            "-bufsize", args.bitrate, "-pix_fmt", "yuv420p",
            "-g", str(args.fps * 2), "-f", "flv" if args.url else "matroska"]

    # No audio source. RTMP services want an audio track, so give them silence
    # rather than have the ingest drop the stream.
    if args.url:
        cmd = cmd[:1] + ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"] + cmd[1:]
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]

    cmd.append(args.url or args.out)
    return cmd


def _has_encoder(name: str) -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return name in out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--instances", type=int, default=4)
    ap.add_argument("--twitch", metavar="KEY")
    ap.add_argument("--youtube", metavar="KEY")
    ap.add_argument("--preview", action="store_true",
                    help="write logs/stream.mkv instead of going live")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "stream.mkv"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--capture-width", type=int, default=1600)
    ap.add_argument("--capture-height", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", default="4500k")
    ap.add_argument("--encoder", default="auto",
                    choices=("auto", "h264_nvenc", "libx264"))
    ap.add_argument("--loglevel", default="warning")
    ap.add_argument("--single-display", metavar=":N",
                    help="capture ONE X display (the /grid browser run "
                         "fullscreen on its own Xvfb) instead of tiling the "
                         "game displays here")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not installed", file=sys.stderr)
        return 1
    a.url = (TWITCH.format(a.twitch) if a.twitch
             else YOUTUBE.format(a.youtube) if a.youtube else None)
    if not a.url and not a.preview:
        print("give --twitch KEY, --youtube KEY, or --preview", file=sys.stderr)
        return 1

    if a.single_display:
        if not re.fullmatch(r":\d+(\.\d+)?", a.single_display):
            print(f"bad display {a.single_display!r}", file=sys.stderr)
            return 1
        sock = f"/tmp/.X11-unix/X{a.single_display[1:].split('.')[0]}"
        if not os.path.exists(sock) and not a.dry_run:
            print(f"no X display {a.single_display} - run the /grid browser on "
                  f"its own Xvfb first", file=sys.stderr)
            return 1
    else:
        missing = [display_for(i) for i in range(a.instances)
                   if not os.path.exists(f"/tmp/.X11-unix/X{display_for(i)[1:]}")]
        if missing:
            print(f"no X display for {', '.join(missing)} - start each instance "
                  f"with `tools/steam-instance <n> --vnc` first", file=sys.stderr)
            if not a.dry_run:
                return 1

    cmd = build(a)
    if not cmd:
        return 1
    if a.dry_run:
        # Never print a stream key.
        print(" ".join(c if not (a.url and a.url in c) else "<rtmp url hidden>"
                       for c in cmd))
        return 0
    print(f"streaming {a.instances} instance(s) as {grid(a.instances)[0]}x"
          f"{grid(a.instances)[1]} at {a.width}x{a.height}{a.fps}fps"
          + (f" -> {'twitch' if a.twitch else 'youtube'}" if a.url
             else f" -> {a.out}"), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())

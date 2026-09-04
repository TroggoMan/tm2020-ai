#!/usr/bin/env python3
"""Local control panel for the TM2020 AI driver.

Serves a dashboard on http://127.0.0.1:8080 that shows live telemetry and
drives everything else in the project: restart the map, load a TMX map, take
manual control of the pad, record a reference line, start and stop training.

Stdlib only, deliberately - it has to work whether or not the torch venv is
ready.

    python3 web/server.py

Needs the broker running (telemetry/broker.py); it connects there rather than
to the plugin, so the trainer can run at the same time.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

# The panel must keep working before the torch venv exists, so it only ever
# imports the two pure-stdlib modules from env/ - never tm_env, which needs
# numpy and gymnasium.
from env import config as tuning          # noqa: E402
from env import surfaces                  # noqa: E402
from env.ports import instance_ports      # noqa: E402

BROKER = ("127.0.0.1", 8767)
PAD = ("127.0.0.1", 8765)
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")


def venv_ready() -> bool:
    """The interpreter existing isn't enough - torch is a long install, and a
    Start button that works before it lands just produces a confusing crash."""
    if not os.path.exists(VENV_PY):
        return False
    site = os.path.join(ROOT, ".venv", "lib")
    for root, dirs, _ in os.walk(site):
        if "stable_baselines3" in dirs:
            return True
        if root.count(os.sep) - site.count(os.sep) > 2:
            dirs.clear()
    return False


def port_open(addr) -> bool:
    try:
        with socket.create_connection(addr, timeout=0.3):
            return True
    except OSError:
        return False


# -- live instance grid ---------------------------------------------------
# One MJPEG stream per X display, tiled in a resizable browser page. No
# noVNC / websockify: ffmpeg's mpjpeg muxer already emits exactly the
# multipart/x-mixed-replace an <img> renders natively.
FFMPEG = shutil.which("ffmpeg")
# Cap concurrent MJPEG streams. Each holds a thread + an ffmpeg encode; a grid
# left open across reconnects, or a client that vanishes without closing the
# socket, used to pile these up until the panel stopped responding.
_MJPEG_MAX = 6
_mjpeg_lock = threading.Lock()
_mjpeg_n = 0


def x_displays(games_only: bool = True) -> list[dict]:
    """X displays that are running a Trackmania game window.

    games_only (the default) drops the desktop (:0), a bare Steam client, and
    anything else - the grid is for watching the cars, not the launcher.
    Pass all=1 on the request to see every display instead.
    """
    out = []
    try:
        socks = sorted(os.listdir("/tmp/.X11-unix"))
    except OSError:
        return out
    for s in socks:
        if not s.startswith("X"):
            continue
        disp = ":" + s[1:]
        label, geom, is_game = "", "", False
        try:
            env = {**os.environ, "DISPLAY": disp}
            wid = subprocess.run(["xdotool", "search", "--onlyvisible",
                                  "--name", "."], capture_output=True,
                                 text=True, env=env, timeout=2).stdout.split()
            best = None   # ((score, area), name, geom)
            for w in wid:
                name = subprocess.run(["xdotool", "getwindowname", w],
                                      capture_output=True, text=True, env=env,
                                      timeout=2).stdout.strip()
                g = subprocess.run(["xdotool", "getwindowgeometry", "--shell", w],
                                   capture_output=True, text=True, env=env,
                                   timeout=2).stdout
                mw = re.search(r"WIDTH=(\d+)", g)
                mh = re.search(r"HEIGHT=(\d+)", g)
                if not (mw and mh):
                    continue
                area = int(mw.group(1)) * int(mh.group(1))
                if area < 40000:            # skip tiny utility windows
                    continue
                game = "trackmania" in name.lower()
                score = 2 if game else \
                        1 if name and name.lower() != "openbox" else 0
                key = (score, area)
                if best is None or key > best[0]:
                    best = (key, name, f"{mw.group(1)}x{mh.group(1)}")
                    is_game = game or is_game
                if game:
                    is_game = True
            if best:
                label, geom = best[1] or "(untitled)", best[2]
        except (OSError, subprocess.SubprocessError):
            pass
        if games_only and not is_game:
            continue
        out.append({"display": disp, "label": label or "(no window)",
                    "geom": geom, "game": is_game})
    return out


def mjpeg_ffmpeg(display: str, fps: int, width: int, q: int = 7,
                 crop: str | None = None):
    """ffmpeg grabbing one X display as an MJPEG multipart stream on stdout.

    The defaults (960 wide, 12fps, q7) are sized for the GRID's small
    thumbnails, where a dozen streams share one browser tab. They look bad
    blown back up to full size in OBS, which is what a stream does - so w, fps
    and q are all query params. For a broadcast source use

        /grid/mjpeg?display=:99&w=1920&fps=30&q=2

    `crop` takes W:H:X:Y and is applied BEFORE the scale, so it selects a
    region of the display rather than of the output. It is how a square view
    is cut out of a 16:9 display without touching the game:

        /grid/mjpeg?display=:99&crop=1080:1080:420:0&w=1080&fps=30&q=2

    Cropping in ffmpeg beats cropping in OBS because what travels over the
    socket is only the region you keep - the discarded pixels are never
    encoded. It does NOT change what the game renders, so it trims field of
    view; to get a genuinely square RENDER, restart the game square
    (GAME_W=GAME_H in tools/headless-main.sh) and leave crop off.

    scale=-2 keeps the aspect and rounds to an even height, which mpjpeg needs.
    q is ffmpeg's mjpeg quantiser: 2 is near-lossless, 31 is awful. It costs
    bandwidth, but this only ever travels over loopback.
    """
    vf = f"crop={crop},scale={width}:-2" if crop else f"scale={width}:-2"
    # nice +10: this is a NICE-TO-HAVE next to the training control loop, which
    # has a hard 50ms budget per step and no way to catch up if it misses one.
    # The capture stalling for a few milliseconds costs nobody anything; the
    # control loop overrunning records a transition whose dt is a lie.
    return ["nice", "-n", "10", FFMPEG, "-hide_banner", "-loglevel", "error",
            # -draw_mouse 0: leave the pointer OUT of the capture. On a
            # headless display the pointer just parks wherever the last click
            # left it - usually dead centre of the game view - and there is no
            # user moving it away. Hiding it here rather than warping it to a
            # corner keeps it fully usable over VNC, where you still need it to
            # click through menus and build the splitscreen lobby.
            "-f", "x11grab", "-draw_mouse", "0",
            "-framerate", str(fps), "-i", display,
            "-vf", vf, "-q:v", str(q),
            "-f", "mpjpeg", "-"]


def pad_state(seat: int = 0) -> dict | None:
    """What the virtual pad is currently holding.

    This is our side of the wire. The plugin separately reports what the game
    actually received, and the two disagreeing is the clearest possible signal
    that the pad has stopped reaching the game - worth more than the liveness
    check this replaces, for the same one connection.
    """
    from env.ports import seat_pad_addr
    addr = PAD if not seat else seat_pad_addr(seat)
    try:
        with socket.create_connection(addr, timeout=0.4) as s:
            s.sendall(b"state\n")
            return json.loads(s.recv(512).decode().strip())
    except (OSError, json.JSONDecodeError):
        return None


def _render_markdown(title: str, text: str) -> str:
    """Headings, code blocks, tables and paragraphs. Deliberately not a full
    markdown implementation - a dependency-free panel is worth more than
    perfect rendering of a file you can also just open in an editor."""
    import html as _html
    import re as _re

    out, in_code, in_table = [], False, False
    for raw in text.splitlines():
        if raw.startswith("```"):
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(_html.escape(raw))
            continue

        line = _html.escape(raw)
        # inline: `code`, **bold**, *italic*
        line = _re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
        line = _re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", line)

        is_row = line.strip().startswith("|")
        if is_row and not in_table:
            out.append("<table>")
            in_table = True
        if not is_row and in_table:
            out.append("</table>")
            in_table = False
        if is_row:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(_re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            tag = "th" if len(out) and out[-1] == "<table>" else "td"
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells)
                       + "</tr>")
            continue

        stripped = line.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level}>{stripped.lstrip('#').strip()}</h{level}>")
        elif stripped.startswith("> "):
            out.append(f"<blockquote>{stripped[2:]}</blockquote>")
        elif stripped.startswith(("* ", "- ")):
            out.append(f"<li>{stripped[2:]}</li>")
        elif not stripped:
            out.append("")
        elif stripped.startswith("    "):
            out.append(f"<pre><code>{stripped}</code></pre>")
        else:
            out.append(f"<p>{line}</p>")
    if in_table:
        out.append("</table>")
    if in_code:
        out.append("</code></pre>")

    body = "\n".join(out)
    return f"""<!doctype html><meta charset="utf-8">
<title>{_html.escape(title)}</title>
<style>
 body{{background:#12141a;color:#d7dae0;font:15px/1.6 system-ui,sans-serif;
      max-width:52em;margin:0 auto;padding:2.5em 1.5em 6em}}
 h1,h2,h3{{color:#fff;line-height:1.25;margin:1.8em 0 .5em}}
 h1{{font-size:1.9em}} h2{{font-size:1.35em;border-bottom:1px solid #2a2e38;
     padding-bottom:.25em}} h3{{font-size:1.1em}}
 code{{background:#1c1f27;padding:.1em .35em;border-radius:3px;
       font:13px/1.5 ui-monospace,monospace;color:#9fd0a8}}
 pre{{background:#1c1f27;padding:.9em 1.1em;border-radius:6px;overflow-x:auto}}
 pre code{{background:none;padding:0;color:#c8cdd6}}
 blockquote{{border-left:3px solid #3a4150;margin:1em 0;padding:.2em 0 .2em 1em;
             color:#a2a8b4}}
 table{{border-collapse:collapse;margin:1em 0;width:100%}}
 th,td{{border:1px solid #2a2e38;padding:.4em .7em;text-align:left}}
 th{{background:#1c1f27;color:#fff}}
 li{{margin:.3em 0}} strong{{color:#fff}}
 a{{color:#7fb3ff}}
</style>
{body}"""


class Link:
    """Persistent connection to the broker: telemetry in, commands out.

    Also keeps its own checkpoint count. The plugin cannot report one -
    RaceWaypointTimes reads 0 in Time Attack - so the count is derived here by
    proximity to the map's checkpoint landmarks, the same way the training env
    does it. Doing it in the panel too means the counter works whether or not a
    trainer is running.
    """

    def __init__(self):
        self.last_map: str | None = None
        self.sock: socket.socket | None = None
        self.latest: dict | None = None
        self.last_at = 0.0
        self.replies: list[dict] = []
        self.lock = threading.Lock()
        self.lm_map: str | None = None
        self.cps: list[list[float]] = []
        self.finish: list[float] | None = None
        self.spawn: list[float] | None = None
        self.cp_seen = 0
        self.last_t = 0
        threading.Thread(target=self._run, daemon=True).start()

    def ensure_landmarks(self):
        """Fetch the checkpoint list once per map.

        Called from the HTTP thread, never the reader thread: command() waits
        for a reply that the reader thread is the one delivering, so asking
        from inside the reader would deadlock.
        """
        with self.lock:
            uid = (self.latest or {}).get("map") or self.last_map
        if not uid or uid == self.lm_map:
            return
        reply = self.command("landmarks", wait=3.0)
        # Set this either way, so an older plugin that doesn't know the command
        # costs one timeout per map rather than one per status poll.
        self.lm_map = uid
        items = (reply or {}).get("items") or []
        cps = sorted((i for i in items if i.get("kind") == "checkpoint"),
                     key=lambda i: i.get("order", 0))
        self.cps = [i["pos"] for i in cps]
        self.finish = next((i["pos"] for i in items
                            if i.get("kind") == "finish"), None)
        self.spawn = next((i["pos"] for i in items
                           if i.get("kind") == "spawn"), None)
        self.cp_seen = 0
        self._maybe_cache_geometry(uid)

    def _maybe_cache_geometry(self, uid: str):
        """Cache the occupancy grid + road trace for a newly seen map, so it
        exists before any trainer runs. The trainer builds these lazily on its
        first reset, which silently produced nothing when the map was still
        loading and left the whole run with no lidar and no track edges. This
        runs tools/dump_map.py once per uid, in the background."""
        if not uid or os.path.exists(os.path.join(ROOT, "maps", f"{uid}.json")):
            return
        if uid in getattr(self, "_geom_started", set()):
            return
        self.__dict__.setdefault("_geom_started", set()).add(uid)

        def _go():
            try:
                subprocess.run([VENV_PY, "tools/dump_map.py"], cwd=ROOT,
                               capture_output=True, text=True, timeout=90)
            except Exception:                                  # noqa: BLE001
                pass
        threading.Thread(target=_go, daemon=True).start()
        print(f"map {uid}: caching occupancy + roadtrace in the background "
              f"(tools/dump_map.py)", flush=True)

    def _count_cp(self, rec: dict):
        if not self.cps:
            return
        t = rec.get("race_time") or 0
        if t < self.last_t - 200:      # clock went backwards: a new run started
            self.cp_seen = 0
        self.last_t = t
        if self.cp_seen >= len(self.cps):
            return
        p = rec.get("pos") or [0.0, 0.0, 0.0]
        c = self.cps[self.cp_seen]
        d2 = sum((p[i] - c[i]) ** 2 for i in range(3))
        if d2 < 400.0:                 # 20m, matching the env's cp_radius
            self.cp_seen += 1

    def _run(self):
        buf = b""
        while True:
            if self.sock is None:
                try:
                    self.sock = socket.create_connection(BROKER, timeout=3)
                    self.sock.settimeout(1.0)
                    buf = b""
                except OSError:
                    time.sleep(2.0)
                    continue
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                data = b""
            if not data:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
                time.sleep(1.0)
                continue
            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with self.lock:
                    if "car" in rec:
                        self.latest = rec
                        self.last_at = time.time()
                        # Remember the last uid we were actually told.
                        #
                        # In splitscreen most records are heartbeats with no
                        # map field, so reading the uid straight off `latest`
                        # yields None most of the time - and everything keyed
                        # on it then falls back to "default": the surface list
                        # comes up empty, the config shown is the wrong one.
                        # The map does not change while you are driving it, so
                        # the last one we were told is the right answer.
                        if rec.get("map"):
                            self.last_map = rec["map"]
                        self._count_cp(rec)
                    else:
                        self.replies.append(rec)

    def command(self, cmd: str, wait: float = 2.0):
        if self.sock is None:
            return {"ok": False, "err": "broker not connected"}
        with self.lock:
            self.replies.clear()
        try:
            self.sock.sendall((cmd + "\n").encode())
        except OSError as ex:
            return {"ok": False, "err": str(ex)}
        deadline = time.time() + wait
        while time.time() < deadline:
            with self.lock:
                if self.replies:
                    return self.replies.pop(0)
            time.sleep(0.01)
        return {"ok": False, "err": "no reply"}


class Job:
    """One long-running process, owned by whoever is asking.

    Tracked by PID FILE rather than only by the Popen handle. A handle lives
    in this server process, so restarting the panel - or starting the trainer
    from a terminal - left a running job that the UI could see in the log but
    could not stop, and the Start button refused because "already running"
    was false. Adopting by pid means the panel can always stop what is
    actually running, whoever launched it.
    """

    def __init__(self, name: str, logfile: str, match: str = ""):
        self.name = name
        self.logfile = logfile
        # A distinctive fragment of the command line, so a job started outside
        # this process can still be recognised.
        self.match = match
        self.proc: subprocess.Popen | None = None
        self.pidfile = os.path.join(ROOT, "logs", f"{name}.pid")

    # -- adoption ---------------------------------------------------------

    def _alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        # A recycled pid belonging to something else must not be killed.
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                return self.match.encode() in f.read()
        except OSError:
            return False

    def _recorded_pid(self) -> int | None:
        try:
            with open(self.pidfile) as f:
                pid = int(f.read().strip())
        except (OSError, ValueError):
            return None
        return pid if self._alive(pid) else None

    def _scan(self) -> int | None:
        """Find a matching process even without a pid file."""
        if not self.match:
            return None
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as f:
                    if self.match.encode() in f.read():
                        return int(entry)
            except OSError:
                continue
        return None

    def pid(self) -> int | None:
        if self.proc is not None and self.proc.poll() is None:
            return self.proc.pid
        return self._recorded_pid() or self._scan()

    @property
    def running(self) -> bool:
        return self.pid() is not None

    def start(self, argv: list[str]) -> dict:
        existing = self.pid()
        if existing:
            return {"ok": False, "err": f"{self.name} already running "
                                        f"(pid {existing})"}
        os.makedirs(os.path.dirname(self.logfile), exist_ok=True)
        log = open(self.logfile, "w")
        self.proc = subprocess.Popen(
            argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            with open(self.pidfile, "w") as f:
                f.write(str(self.proc.pid))
        except OSError:
            pass
        return {"ok": True, "pid": self.proc.pid}

    def stop(self) -> dict:
        pid = self.pid()
        if pid is None:
            return {"ok": False, "err": f"{self.name} not running"}
        # The trainer needs to release the pad and sockets, so ask nicely.
        try:
            os.killpg(os.getpgid(pid), signal.SIGINT)
        except OSError:
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                pass
        for _ in range(50):
            if not self._alive(pid):
                self._forget()
                return {"ok": True, "pid": pid}
            time.sleep(0.1)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        self._forget()
        return {"ok": True, "pid": pid, "killed": True}

    def _forget(self) -> None:
        self.proc = None
        try:
            os.remove(self.pidfile)
        except OSError:
            pass

    def tail(self, n: int = 40) -> str:
        try:
            with open(self.logfile) as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""


LINK = Link()
TRAINER = Job("trainer", os.path.join(ROOT, "logs", "train.log"),
              match="train/train_sac.py")
RECORDER = Job("recorder", os.path.join(ROOT, "logs", "record_line.log"),
               match="tools/record_line.py")
# The pad-server fleet (N pads + one broker into a splitscreen game).
# tools/fleet.py supervises and restarts its children, so the panel only has
# to own the supervisor - one pid, stopped with the same button.
FLEET = Job("fleet", os.path.join(ROOT, "logs", "fleet.log"),
            match="tools/fleet.py")

_SYS_PY = None


def system_python() -> str:
    """The interpreter fleet.py must run under: it (and its pad-server
    children) import evdev, which the torch venv does not have. Same rule as
    tools/fleet.py; cached so the button is instant on the second click."""
    global _SYS_PY
    if _SYS_PY is not None:
        return _SYS_PY
    for cand in ("/usr/bin/python3", shutil.which("python3")):
        if not cand:
            continue
        try:
            subprocess.run([cand, "-c", "import evdev"], check=True,
                           capture_output=True, timeout=10)
            _SYS_PY = cand
            return cand
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                OSError):
            continue
    _SYS_PY = ""
    return ""


CONFIG_DIR = os.path.join(ROOT, "configs")


def list_configs() -> list[str]:
    """Map identifiers, each listed ONCE. A trailing .explore / .race is a
    PROFILE, picked by the stage dropdown - not part of the map id. Listing
    `<uid>.explore` as its own selectable map is what let a save land in
    `<uid>.explore.explore.json`."""
    if not os.path.isdir(CONFIG_DIR):
        return []
    out = set()
    for f in os.listdir(CONFIG_DIR):
        if not f.endswith(".json"):
            continue
        base = f[:-5]
        for suf in (".explore", ".race"):
            if base.endswith(suf):
                base = base[:-len(suf)]
                break
        out.add(base)
    return sorted(out)


def active_profile() -> str:
    """The config profile the running trainer is actually reading.

    Explore and race keep SEPARATE files per map - `<uid>.explore.json` and
    `<uid>.json` - because they want very different numbers on the same track.
    The panel had no idea, so it always edited the race file: with an explore
    run going, every surface weight and reward tweak was written to a file
    nothing was reading, and the change simply had no effect. Ask the trainer
    what it is doing rather than guessing.
    """
    pid = TRAINER.pid()
    if pid is None:
        return ""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().split(b"\0")
    except OSError:
        return ""
    return "explore" if b"explore" in argv else ""


def config_name(uid: str, profile: str = "") -> str:
    return f"{uid}.{profile}" if profile else uid


def read_config(uid: str) -> dict:
    """Always merged over DEFAULTS, so the panel renders every knob even for a
    map whose file predates a new one."""
    path = os.path.join(CONFIG_DIR, f"{uid}.json")
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        raw = {}
    return tuning.deep_merge(tuning.DEFAULTS, raw)


def write_config(uid: str, data: dict) -> dict:
    """Atomic, because the trainer is watching this file by mtime and a
    half-written read would be a parse error mid-run."""
    if not uid or os.sep in uid or uid in (".", ".."):
        return {"ok": False, "err": "bad map id"}
    os.makedirs(CONFIG_DIR, exist_ok=True)
    path = os.path.join(CONFIG_DIR, f"{uid}.json")
    # DEFAULTS < the file already on disk < the incoming edits. Merging over
    # the EXISTING file (not just DEFAULTS) is what stops a partial save from
    # blanking keys the caller did not send - `markers`, `marks`, `hints`,
    # `route_edits` all have their own editors and must survive a plain
    # tuning save.
    try:
        with open(path) as _ef:
            existing = json.load(_ef)
    except (OSError, json.JSONDecodeError):
        existing = {}
    merged = tuning.deep_merge(
        tuning.deep_merge(tuning.DEFAULTS, existing), data)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=2)
        os.replace(tmp, path)
    except OSError as ex:
        return {"ok": False, "err": str(ex)}
    return {"ok": True, "map": uid}


def list_lines() -> list[str]:
    d = os.path.join(ROOT, "lines")
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(".json"))


def line_map(name: str) -> str | None:
    """Which map a reference line was recorded on, or None if unknown.

    Read directly rather than through Centerline, which needs numpy - the
    panel has to keep working before the venv exists.
    """
    try:
        with open(os.path.join(ROOT, "lines", name)) as f:
            return json.load(f).get("map")
    except (OSError, json.JSONDecodeError):
        return None


def _load_route_edits(uid: str) -> list:
    p = os.path.join(ROOT, "maps", f"{uid}.route_edits.json")
    try:
        with open(p) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for pat in doc.get("patches", []):
        try:
            out.append({"a": [float(pat["a"][0]), float(pat["a"][1])],
                        "b": [float(pat["b"][0]), float(pat["b"][1])],
                        "points": [[float(x), float(z)] for x, z in pat["points"]],
                        "kind": "jump" if pat.get("kind") == "jump" else "road"})
        except (KeyError, TypeError, ValueError):
            continue
    return [p for p in out if p["points"]]


def _dekink(pts: list, zi: int = 1, cos_thresh: float = -0.25, passes: int = 5):
    """Drop points where the line folds back on itself - a splice artefact of
    patches whose ends do not line up. Mirrors env.routemodel._dekink so the
    panel draws the same line the trainer uses. zi is the index of the second
    horizontal axis (1 for [x,z], 2 for [x,y,z]). Pure stdlib."""
    for _ in range(passes):
        if len(pts) < 4:
            break
        keep = [pts[0]]
        dropped = 0
        for i in range(1, len(pts) - 1):
            p0, p1, p2 = keep[-1], pts[i], pts[i + 1]
            ax, az = p1[0] - p0[0], p1[zi] - p0[zi]
            bx, bz = p2[0] - p1[0], p2[zi] - p1[zi]
            na = (ax * ax + az * az) ** 0.5
            nb = (bx * bx + bz * bz) ** 0.5
            if na > 1e-6 and nb > 1e-6 and \
                    (ax * bx + az * bz) / (na * nb) < cos_thresh:
                dropped += 1
                continue
            keep.append(p1)
        keep.append(pts[-1])
        pts = keep
        if not dropped:
            break
    return pts


_DECK_COLS: dict = {}      # uid -> {(cx,cz): [world_y deck floors]}, cached


def _snap_line_to_deck(uid: str, pts: list) -> list:
    """Pull each [x,y,z] point's Y onto the real road deck from the occupancy
    grid, so the panel draws (and measures markers on) the same line the
    trainer uses. Mirrors env.routemodel._snap_to_deck. Pure stdlib."""
    if not pts:
        return pts
    cols = _DECK_COLS.get(uid)
    if cols is None:
        cols = {}
        try:
            with open(os.path.join(ROOT, "maps", f"{uid}.json")) as f:
                g = json.load(f)
            a = g.get("cells") or []
            by = (g.get("block_size") or [32, 8, 32])[1]
            bh = g.get("base_height", 8)
            for i in range(0, len(a) - 2, 3):
                cx, cy, cz = a[i], a[i + 1], a[i + 2]
                wy = (cy - bh) * by
                if wy > 4.0:
                    cols.setdefault((cx, cz), []).append(wy + 2.0)
            for k in cols:
                cols[k].sort()
        except (OSError, ValueError, KeyError, TypeError):
            cols = {}
        _DECK_COLS[uid] = cols
    if not cols:
        return pts
    try:
        with open(os.path.join(ROOT, "maps", f"{uid}.json")) as f:
            g = json.load(f)
        bx, _, bz = g.get("block_size") or [32, 8, 32]
    except (OSError, ValueError):
        bx, bz = 32, 32

    def cell(p):
        return (int(p[0] // bx), int(p[2] // bz))

    ys = [p[1] for p in pts]
    snapped = [None] * len(pts)
    prev = None
    for i, p in enumerate(pts):
        decks = cols.get(cell(p))
        if not decks:
            continue
        best = min(decks, key=lambda d: abs(d - ys[i]))
        if prev is not None and abs(best - prev) > 30.0:
            near = [d for d in decks if abs(d - prev) <= 30.0]
            if near:
                best = min(near, key=lambda d: abs(d - ys[i]))
        snapped[i] = best
        prev = best
    have = [i for i, v in enumerate(snapped) if v is not None]
    if len(have) < 2:
        return pts
    out = [list(p) for p in pts]
    for i in range(len(out)):
        if snapped[i] is not None:
            out[i][1] = snapped[i]
        elif i < have[0] or i > have[-1]:
            out[i][1] = ys[i]
        else:
            lo = max(h for h in have if h <= i)
            hi = min(h for h in have if h >= i)
            t = 0 if hi == lo else (i - lo) / (hi - lo)
            out[i][1] = snapped[lo] + (snapped[hi] - snapped[lo]) * t
    return out


def _splice_patches(pts: list, patches: list):
    """pts: [[x,z],...]. Replace the span between each patch's two anchors (the
    nearest line points) with its polyline. Pure stdlib - the panel has no
    numpy. Returns (new_pts, jump_spans_world)."""
    def nearest(p):
        bi, bd = 0, None
        for i, q in enumerate(pts):
            d = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
            if bd is None or d < bd:
                bd, bi = d, i
        return bi
    jumps = []
    for pat in patches:
        if not pts:
            break
        ia, ib = nearest(pat["a"]), nearest(pat["b"])
        lo, hi = sorted((ia, ib))
        seg = pat["points"][::-1] if ia > ib else pat["points"]
        pts = pts[:lo + 1] + [list(q) for q in seg] + pts[hi:]
        if pat["kind"] == "jump" and seg:
            jumps.append([seg[0][0], seg[0][1], seg[-1][0], seg[-1][1]])
    return _dekink(pts, zi=1), jumps


def spliced_line_3d(uid: str):
    """The 3D line the TRAINER will use, and its cumulative distance.

    Markers are stored as a distance, and the reward compares that against
    progress along the line AFTER route edits are spliced in. So the distance
    has to be measured on the SPLICED geometry: a hand-drawn straight line
    across a section is shorter than the wander it replaces, so measuring on
    the raw roadtrace puts every marker past that point in the wrong place.

    Mirrors env.routemodel._apply_patches exactly, including lerping y between
    the two anchor points - the patch itself is 2D (drawn top-down), so any
    other choice of height would disagree with the trainer's 3D distance.
    Pure stdlib; the panel has no numpy.

    Returns (points3, cumulative) or (None, None).
    """
    try:
        with open(os.path.join(ROOT, "maps", f"{uid}.roadtrace.json")) as fh:
            pts = [list(p[:3]) for p in (json.load(fh).get("points") or [])]
    except (OSError, ValueError, TypeError, IndexError):
        return None, None
    if not pts:
        return None, None

    def nearest_xz(p):
        bi, bd = 0, None
        for i, q in enumerate(pts):
            d = (q[0] - p[0]) ** 2 + (q[2] - p[1]) ** 2
            if bd is None or d < bd:
                bd, bi = d, i
        return bi

    for pat in _load_route_edits(uid):
        try:
            ia, ib = nearest_xz(pat["a"]), nearest_xz(pat["b"])
        except (KeyError, TypeError, IndexError):
            continue
        lo, hi = sorted((ia, ib))
        pp = pat.get("points") or []
        if ia > ib:
            pp = pp[::-1]
        if not pp:
            continue
        y0, y1 = pts[lo][1], pts[hi][1]
        m = len(pp)
        seg = []
        for k, q in enumerate(pp):
            t = k / max(m - 1, 1)
            seg.append([float(q[0]), y0 + (y1 - y0) * t, float(q[1])])
        pts = pts[:lo + 1] + seg + pts[hi:]

    pts = _dekink(pts, zi=2)     # same fold-removal the trainer applies
    pts = _snap_line_to_deck(uid, pts)   # ride the real road deck (see env.routemodel)

    cum = [0.0]
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        cum.append(cum[-1] + (dx * dx + dy * dy + dz * dz) ** 0.5)
    return pts, cum


def route_payload(uid: str) -> dict:
    """Everything the panel's route editor needs. Pure stdlib: reads the line
    from the pre-built maps/<uid>.roadtrace.json / .learned.json caches (the
    trainer regenerates them), the occupancy backdrop from maps/<uid>.json, the
    landmarks from the live link, and the user's saved patches - spliced in
    here so the preview matches what env.routemodel will produce."""
    out = {"ok": True, "map": uid, "source": "none", "points": [],
           "half_width": [], "cells": [], "checkpoints": [], "spawn": None,
           "finish": None, "order": [], "jumps": [], "patches": []}
    out["patches"] = _load_route_edits(uid)

    def _read(name):
        try:
            with open(os.path.join(ROOT, "maps", name)) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    grid = _read(f"{uid}.json")
    if grid and grid.get("cells"):
        a = grid["cells"]
        xz = {(a[i], a[i + 2]) for i in range(0, len(a) - 2, 3)}
        out["cells"] = [[x, z] for x, z in sorted(xz)][:6000]

    base = _read(f"{uid}.roadtrace.json")
    src = "roadtrace"
    if base is None:
        base = _read(f"{uid}.learned.json")
        src = "learned"
    if base and base.get("points"):
        pts = [[round(p[0], 1), round(p[2], 1)] for p in base["points"]]
        out["order"] = base.get("order", [])
        hw = base.get("half_width") or []
    else:
        # provisional: straight lines spawn -> checkpoints -> finish
        pts, hw, src = [], [], "provisional"

    with LINK.lock:
        live = (LINK.latest or {}).get("map") or LINK.last_map
        cps = list(LINK.cps or [])
        finish = list(LINK.finish) if LINK.finish else None
        spawn = list(LINK.spawn) if LINK.spawn else None
    if live == uid:
        if cps:
            out["checkpoints"] = [[c[0], c[2]] for c in cps]
        if finish:
            out["finish"] = [finish[0], finish[2]]
        if spawn:
            out["spawn"] = [spawn[0], spawn[2]]
        if not pts and cps:
            chain = ([spawn] if spawn else []) + cps + ([finish] if finish else [])
            for i in range(len(chain) - 1):
                a, b = chain[i], chain[i + 1]
                n = max(int((((a[0] - b[0]) ** 2 +
                              (a[2] - b[2]) ** 2) ** 0.5) / 4), 1)
                for k in range(n):
                    t = k / n
                    pts.append([round(a[0] + (b[0] - a[0]) * t, 1),
                                round(a[2] + (b[2] - a[2]) * t, 1)])
            if chain:
                pts.append([round(chain[-1][0], 1), round(chain[-1][2], 1)])
    elif not pts:
        out["note"] = ("map not loaded and no cached trace - load it in-game, "
                       "or run tools/build_route.py " + uid)

    if out["patches"] and pts:
        pts, jumps = _splice_patches(pts, out["patches"])
        out["jumps"] = jumps
        src += f"+edits({len(out['patches'])})"

    out["points"] = pts
    out["half_width"] = hw[:len(pts)] if hw else []
    out["source"] = src if pts else "none"

    # Reward markers, converted from "distance along the line" to map XZ so the
    # panel can draw them. They are stored as a distance because that is what
    # the reward compares against (max_s), but a distance is not something you
    # can see on a map - which is the whole reason for showing them here.
    #
    # Measured along the SPLICED points, i.e. the same geometry the panel is
    # about to draw, so a marker never appears off the line it belongs to.
    out["markers"] = []
    # Named marks (mark_from / mark_to for hints) - display only. Stored as a
    # world [x,y,z], so no line projection needed, just drop the Y.
    out["marks"] = []
    _cfg = {}
    for _p in (config_name(uid, active_profile()) + ".json",
               f"{uid}.explore.json", f"{uid}.json"):
        try:
            with open(os.path.join(ROOT, "configs", _p)) as fh:
                _cfg = json.load(fh)
            break
        except (OSError, ValueError):
            continue
    marks = (_cfg.get("markers") or [])
    for nm, p in (_cfg.get("marks") or {}).items():
        try:
            out["marks"].append({"name": nm,
                                 "xz": [round(float(p[0]), 1),
                                        round(float(p[2]), 1)]})
        except (TypeError, ValueError, IndexError):
            pass
    if marks and pts:
        # Distance must be measured the way the REWARD measures it: Centerline
        # accumulates 3D segment lengths, so a 2D (x,z) sum drifts on every
        # slope and the diamond lands further along the line than the bonus
        # actually fires. Only 3.5m over this whole track, but the drawing
        # exists to be trusted, and 2D would be wrong by more on a hillier map.
        src3, cum = spliced_line_3d(uid)
        if not src3:
            src3, cum = [], [0.0]
        total = cum[-1] or 1.0
        for m in marks:
            try:
                at = float(m["s"])
            except (TypeError, ValueError, KeyError):
                continue
            at = max(0.0, min(at, total))
            # INTERPOLATE along the segment; do not snap to the nearest point.
            # A hand-drawn patch can be two points spanning 126m, so snapping
            # put a marker at the far END of the straight - 60m from where its
            # bonus actually fires. The reward compares `s` numerically and was
            # always right; it was only the drawing that lied, which is worse,
            # because the drawing is what you place them with.
            j = next((k for k, c in enumerate(cum) if c >= at), len(cum) - 1)
            j = max(1, min(j, len(pts) - 1))
            c0, c1 = cum[j - 1], cum[j]
            t = 0.0 if c1 <= c0 else (at - c0) / (c1 - c0)
            t = max(0.0, min(1.0, t))
            p0, p1 = pts[j - 1], pts[j]
            out["markers"].append({
                "s": round(at, 1),
                "bonus": float(m.get("bonus", 0.0)),
                "xz": [round(p0[0] + (p1[0] - p0[0]) * t, 1),
                       round(p0[1] + (p1[1] - p0[1]) * t, 1)],
            })
    return out


def route3d_payload(uid: str) -> dict:
    """Everything the /route3d viewer needs: the occupancy blocks in world
    space, the deck-snapped reference line, landmarks, jump spans and marks.
    Repackages what route_payload / spliced_line_3d already compute."""
    out = {"ok": True, "map": uid, "block": [32, 8, 32], "base_height": 8,
           "blocks": [], "names": [], "line": [], "checkpoints": [],
           "spawn": None, "finish": None, "jumps": [], "marks": []}
    try:
        with open(os.path.join(ROOT, "maps", f"{uid}.json")) as f:
            g = json.load(f)
    except (OSError, ValueError):
        g = {}
    out["block"] = g.get("block_size") or [32, 8, 32]
    out["base_height"] = g.get("base_height", 8)
    names = g.get("names") or []
    out["names"] = names
    boxes = g.get("boxes") or []
    # boxes is flat, 8 ints/block: x,y,z,dir,sx,sy,sz,nameidx
    if boxes and len(boxes) % 8 == 0:
        for i in range(0, len(boxes), 8):
            x, y, z, d, sx, sy, sz, ni = boxes[i:i + 8]
            out["blocks"].append([x, y, z, d, sx, sy, sz, ni])

    p3, _ = spliced_line_3d(uid)
    if p3:
        out["line"] = [[round(p[0], 1), round(p[1], 1), round(p[2], 1)]
                       for p in p3]

    with LINK.lock:
        live = (LINK.latest or {}).get("map") or LINK.last_map
        cps = list(LINK.cps or [])
        finish = list(LINK.finish) if LINK.finish else None
        spawn = list(LINK.spawn) if LINK.spawn else None
    if live == uid:
        out["checkpoints"] = [[round(c[0], 1), round(c[1], 1), round(c[2], 1)]
                              for c in cps]
        if finish:
            out["finish"] = [round(finish[0], 1), round(finish[1], 1),
                             round(finish[2], 1)]
        if spawn:
            out["spawn"] = [round(spawn[0], 1), round(spawn[1], 1),
                            round(spawn[2], 1)]

    for pat in _load_route_edits(uid):
        if pat.get("kind") == "jump" and pat.get("points"):
            pp = pat["points"]
            out["jumps"].append([pp[0][0], pp[0][1], pp[-1][0], pp[-1][1]])

    _cfg = {}
    for _p in (config_name(uid, active_profile()) + ".json",
               f"{uid}.explore.json", f"{uid}.json"):
        try:
            with open(os.path.join(ROOT, "configs", _p)) as fh:
                _cfg = json.load(fh)
            break
        except (OSError, ValueError):
            continue
    for nm, p in (_cfg.get("marks") or {}).items():
        try:
            out["marks"].append({"name": nm, "pos": [float(p[0]), float(p[1]),
                                                     float(p[2])]})
        except (TypeError, ValueError, IndexError):
            pass
    return out


def line_distance(name: str, pos) -> float | None:
    """Closest approach of a reference line to a world position.

    The map uid only helps for lines recorded since it was added, and every
    line already on disk predates it. Geometry works for all of them: if the
    car is 700m from the nearest point of a line, that line is not for this
    track, whatever its metadata says.
    """
    if not pos:
        return None
    try:
        with open(os.path.join(ROOT, "lines", name)) as f:
            pts = json.load(f).get("points") or []
    except (OSError, json.JSONDecodeError):
        return None
    if not pts:
        return None
    best = min((p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2 + (p[2] - pos[2]) ** 2
               for p in pts)
    return best ** 0.5


# Beyond this the line is certainly for another track. Well past any sane
# off-line limit, so it never fires on a line that merely needs retuning.
FAR_FROM_LINE = 200.0


def lines_with_maps() -> list[dict]:
    with LINK.lock:
        latest = LINK.latest or {}
    live, pos = latest.get("map"), (latest.get("pos") if latest.get("car") else None)
    out = []
    for name in list_lines():
        uid = line_map(name)
        dist = line_distance(name, pos)
        if uid and live:
            matches = uid == live
        elif dist is not None:
            matches = dist <= FAR_FROM_LINE
        else:
            matches = None      # no car on track yet; nothing to compare
        out.append({"name": name, "map": uid, "matches": matches,
                    "distance": round(dist) if dist is not None else None})
    return out


def list_archive(name: str = "sac_tm") -> list[dict]:
    """Rollback points. Newest first - that is the order you want them in when
    the last hour of training has made things worse."""
    d = os.path.join(ROOT, "models", "archive", name)
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if not fn.endswith(".zip"):
            continue
        full = os.path.join(d, fn)
        try:
            st = os.stat(full)
        except OSError:
            continue
        out.append({"name": fn[:-4], "path": full, "mtime": st.st_mtime,
                    "size": st.st_size})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def list_models() -> list[str]:
    d = os.path.join(ROOT, "models")
    if not os.path.isdir(d):
        return []
    return sorted(f[:-4] for f in os.listdir(d)
                  if f.endswith(".zip") and not f.endswith("_best.zip"))


def tail_jsonl(path: str, n: int = 25) -> list[dict]:
    try:
        with open(path) as f:
            lines = f.readlines()[-n:]
    except OSError:
        return []
    out = []
    for raw in lines:
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


_runstats_cache = {"key": None, "val": {}}


def run_stats() -> dict:
    """Whole-run totals from the why log, not just the recent tail.

    /api/why returns the last 25 entries, which is right for the WHY overlay
    but makes the bar's per-checkpoint cells misleading: a finish 35 episodes
    ago shows as "CP5 0/20" while "best lap" still (correctly) reports it. Two
    true numbers that look like a contradiction.

    The log is rotated at the start of every run, so scanning the whole file
    IS "this run" - no episode-range bookkeeping needed. Cached on (size,
    mtime) because the overlays poll every few seconds and this would
    otherwise re-read a growing file each time.
    """
    path = os.path.join(ROOT, "logs", "why.jsonl")
    try:
        st = os.stat(path)
        key = (st.st_size, st.st_mtime)
    except OSError:
        return {"episodes": 0, "reached": {}, "finishes": 0, "best_ms": None}
    if _runstats_cache["key"] == key:
        return _runstats_cache["val"]

    episodes = finishes = 0
    best_ms = None
    cp_total = 0
    reached: dict[int, int] = {}
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episodes += 1
                cp_total = r.get("cp_total") or cp_total
                cp = int(r.get("cp") or 0)
                # Cumulative: an episode that reached CP4 also reached 1..3.
                for n in range(1, cp + 1):
                    reached[n] = reached.get(n, 0) + 1
                if r.get("reason") == "FINISH":
                    finishes += 1
                    t = r.get("race_time")
                    if t and (best_ms is None or t < best_ms):
                        best_ms = t
    except OSError:
        pass
    val = {"episodes": episodes, "reached": reached, "finishes": finishes,
           "best_ms": best_ms, "cp_total": cp_total}
    _runstats_cache["key"] = key
    _runstats_cache["val"] = val
    return val


def list_runs() -> dict:
    """What the last few hours actually produced: real .Replay.Gbx files from
    finished PBs, plus JSON traces for the runs that got further without
    finishing."""
    base = os.path.join(ROOT, "runs")
    out = {"maps": []}
    if not os.path.isdir(base):
        return out
    for uid in sorted(os.listdir(base)):
        d = os.path.join(base, uid)
        if not os.path.isdir(d):
            continue
        entry = {"map": uid, "replays": [], "traces": 0}
        rep = os.path.join(d, "replays")
        if os.path.isdir(rep):
            names = sorted((f for f in os.listdir(rep) if f.endswith(".Gbx")),
                           reverse=True)
            entry["replays"] = names[:20]
            entry["replay_total"] = len(names)
        tr = os.path.join(d, "traces")
        if os.path.isdir(tr):
            entry["traces"] = len([f for f in os.listdir(tr)
                                   if f.endswith(".json")])
        out["maps"].append(entry)
    return out


REPLAY_DIRS = [
    "/mnt/4TB/SteamLibrary/steamapps/compatdata/2225070/pfx/drive_c/"
    "users/steamuser/Documents/Trackmania/Replays",
    os.path.expanduser("~/Downloads"),
]

TIME_RE = re.compile(r"\((\d+)[_'](\d+)[_'']{1,2}(\d+)\)")


def pretty_time(name: str) -> str | None:
    """Trackmania puts the run time in the filename, e.g. (00'24''595)."""
    m = TIME_RE.search(name)
    if not m:
        return None
    mins, secs, ms = (int(g) for g in m.groups())
    total = mins * 60 + secs + ms / 1000
    return f"{mins}:{secs:02d}.{ms:03d}" if mins else f"{total:.3f}s"


def time_seconds(name: str) -> float | None:
    m = TIME_RE.search(name)
    if not m:
        return None
    mins, secs, ms = (int(g) for g in m.groups())
    return mins * 60 + secs + ms / 1000


def find_replays(query: str = "", limit: int = 200) -> list[dict]:
    q = query.lower().strip()
    out = []
    for base in REPLAY_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for fn in files:
                low = fn.lower()
                if not (low.endswith(".replay.gbx") or low.endswith(".ghost.gbx")):
                    continue
                if q and q not in low:
                    continue
                full = os.path.join(root, fn)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                out.append({
                    "path": full,
                    "name": fn,
                    "dir": os.path.relpath(root, base),
                    "time": pretty_time(fn),
                    "seconds": time_seconds(fn),
                    "mtime": st.st_mtime,
                    "size": st.st_size,
                })
    # Newest first - the run you just drove is almost always the one you want.
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out[:limit]


def seen_materials(uid: str | None) -> list[str]:
    """Which materials this map has actually put under the wheels.

    EPlugSurfaceMaterialId has 81 entries and a Trackmania track uses about
    six of them - the rest are ShootMania and engine-test leftovers that no
    map contains. Offering all 81 in the tuning UI is offering 75 rows that
    can never fire. The environment records what it has genuinely driven on
    and writes it here, so the list is evidence rather than a guess.

    Empty until a trainer has driven the map at least once.
    """
    if not uid or os.sep in uid:
        return []
    path = os.path.join(ROOT, "maps", f"{uid}.materials.json")
    try:
        with open(path) as f:
            return list(json.load(f).get("materials") or [])
    except (OSError, json.JSONDecodeError):
        return []


def fleet_status(instances: int = 4) -> list[dict]:
    """Which instance ports are actually listening.

    The panel is where you decide how many games to drive, so it has to be
    able to say which ones are ready. A pad without a plugin is a game that
    isn't running; a plugin without a broker is a game nothing can read.
    """
    out = []
    for i in range(instances):
        p = instance_ports(i)
        row = {"instance": i, **p}
        for key in ("pad", "plugin", "broker"):
            row[key + "_up"] = port_open(("127.0.0.1", p[key]))
        row["ready"] = row["pad_up"] and row["broker_up"]
        out.append(row)
    return out


def seat_status(latest: dict | None, seats: int = 4) -> list[dict]:
    """One row per splitscreen seat: does it have a car, and does a pad reach it?

    `bound` is the thing worth showing. A seat can have a car, be in Playing,
    and still answer to nothing, because the game assigns controllers to seats
    in its own settings and nothing we do from outside changes that. The
    symptom of an unbound seat is every episode ending 'stuck', which reads
    exactly like a policy that cannot drive - so it is worth stating plainly
    rather than leaving to be inferred.
    """
    from env.ports import seat_ports
    players = {p.get("slot"): p for p in ((latest or {}).get("players") or [])}
    out = []
    for i in range(seats):
        sp = seat_ports(i)
        p = players.get(i) or {}
        row = {"seat": i, "pad": sp["pad"], "broker": sp["broker"],
               "pad_up": port_open(("127.0.0.1", sp["pad"])),
               "car": bool(p.get("car")),
               "ui": p.get("ui"),
               "dist": p.get("dist"),
               "speed": p.get("speed"),
               "in_steer": p.get("in_steer")}
        # `moving` is all a passive read can honestly say. A car that has
        # covered ground was driven by SOMETHING - which during setup is
        # usually the keyboard - so inferring "a pad reaches this seat" from
        # it claims ready when nothing of ours is connected. That is the worst
        # direction for this indicator to be wrong in, and it was: all four
        # seats read bound while every pad drove nothing.
        #
        # Whether a pad actually reaches a seat can only be established by
        # moving that pad and watching, which is what tools/calibrate_seats.py
        # does. Until it has run, this is unknown and says so.
        row["moving"] = bool(p.get("car")) and (
            abs(float(p.get("in_steer") or 0.0)) > 0.01
            or float(p.get("dist") or 0.0) > 1.0)
        row["bound"] = _calibration().get(str(i))
        out.append(row)
    return out


def _calibration() -> dict:
    """Last recorded pad -> seat mapping, if one has been taken."""
    try:
        with open(os.path.join(ROOT, "logs", "seat_calibration.json")) as f:
            return json.load(f).get("mapping", {})
    except (OSError, json.JSONDecodeError):
        return {}


def suggest_line_name(filename: str) -> str:
    """Turn 'Spring 2026 - 03_Slagathor1213_...(00'24''595).Replay.Gbx' into
    'spring-2026-03'."""
    stem = re.sub(r"\.(replay|ghost)\.gbx$", "", filename, flags=re.I)
    stem = stem.split("_")[0]
    stem = re.sub(r"\(.*?\)", "", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
    return stem or "line"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # the access log is noise for a single-user panel

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(500, b"index.html missing", "text/plain")
            return

        # Standalone WHY overlay for an OBS browser source. Its own page
        # rather than part of the panel: a browser source wants one thing, big
        # enough to read at stream resolution, on a transparent background,
        # with no controls to mis-click. Query params n / ms / terms / solid.
        # Horizontal training status bar, for a browser source UNDER the
        # game view. /why is its portrait counterpart; they show different
        # things on purpose (per-episode detail vs run-level state).
        if self.path.split("?")[0] in ("/game", "/game.html"):
            try:
                with open(os.path.join(HERE, "game.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                self._send(404, b"game.html missing", "text/plain")
            return

        if self.path.split("?")[0] in ("/bar", "/bar.html"):
            try:
                with open(os.path.join(HERE, "bar.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                self._send(500, b"bar.html missing", "text/plain")
            return

        if self.path.split("?")[0] in ("/why", "/why.html"):
            try:
                with open(os.path.join(HERE, "why.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                self._send(500, b"why.html missing", "text/plain")
            return

        if self.path in ("/route3d", "/route3d.html"):
            try:
                with open(os.path.join(HERE, "route3d.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                self._send(500, b"route3d.html missing", "text/plain")
            return

        if self.path == "/grid" or self.path == "/grid.html":
            try:
                with open(os.path.join(HERE, "grid.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                # This file changes often while it is being built; a cached
                # copy of a broken version looks exactly like "still nothing".
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                self._send(500, b"grid.html missing", "text/plain")
            return

        if self.path.startswith("/grid/displays"):
            from urllib.parse import parse_qs, urlparse
            allq = parse_qs(urlparse(self.path).query).get("all")
            self._json({"ffmpeg": bool(FFMPEG),
                        "displays": x_displays(games_only=not allq)})
            return

        if self.path.startswith("/grid/mjpeg"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            disp = (q.get("display") or [":0"])[0]
            if not re.fullmatch(r":\d+(\.\d+)?", disp):
                self._send(400, b"bad display", "text/plain")
                return
            if not FFMPEG:
                self._send(503, b"ffmpeg not installed", "text/plain")
                return
            try:
                fps = max(1, min(30, int((q.get("fps") or ["12"])[0])))
                width = max(160, min(1920, int((q.get("w") or ["960"])[0])))
                qual = max(2, min(31, int((q.get("q") or ["7"])[0])))
            except ValueError:
                fps, width, qual = 12, 960, 7
            # crop=W:H:X:Y, validated strictly - it is interpolated into an
            # ffmpeg filter string, so nothing but digits and colons gets in.
            crop = (q.get("crop") or [""])[0] or None
            if crop and not re.fullmatch(r"\d{1,5}:\d{1,5}:\d{1,5}:\d{1,5}",
                                         crop):
                self._send(400, b"crop must be W:H:X:Y", "text/plain")
                return
            global _mjpeg_n
            with _mjpeg_lock:
                if _mjpeg_n >= _MJPEG_MAX:
                    self._send(503, b"too many live streams", "text/plain")
                    return
                _mjpeg_n += 1
            proc = subprocess.Popen(mjpeg_ffmpeg(disp, fps, width, qual, crop),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=ffmpeg")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Connection", "close")
            self.end_headers()
            # A client that walks away without closing the socket leaves
            # wfile.write() to block forever once the send buffer fills - the
            # thread and its ffmpeg then leak. A write timeout turns that into
            # an OSError the finally can clean up.
            try:
                self.connection.settimeout(20)
            except OSError:
                pass
            try:
                while True:
                    chunk = proc.stdout.read(32768)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
                pass
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                with _mjpeg_lock:
                    _mjpeg_n -= 1
            return

        if self.path in ("/docs", "/tricks", "/training"):
            # The project docs, rendered well enough to read in a tab. Not a
            # markdown engine - just enough structure that TRICKS.md is
            # readable without leaving the panel, because the whole point of
            # the button is that you are looking at it while tuning.
            name = {"/tricks": "TRICKS.md", "/training": "TRAINING.md"}.get(
                self.path, "README.md")
            try:
                with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                    self._send(200, _render_markdown(name, f.read()).encode(),
                               "text/html; charset=utf-8")
            except OSError:
                self._send(404, b"not found", "text/plain")
            return

        if self.path == "/api/train/args":
            # The RUNNING trainer's actual command line, mapped back to the
            # panel's form fields - so the launch form can show what is
            # really in force, even for a run started outside the panel.
            pid = TRAINER.pid()
            if not pid:
                self._json({"running": False, "args": {}})
                return
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    argv = [a for a in f.read().split(b"\0") if a]
                argv = [a.decode("utf-8", "replace") for a in argv]
            except OSError:
                self._json({"running": False, "args": {}})
                return
            # flag -> (form key, kind). kind: v = takes a value, b = bare bool
            FLAGS = {
                "--stage": ("stage", "v"), "--steps": ("steps", "v"),
                "--name": ("name", "v"), "--resume": ("resume", "b"),
                "--control-hz": ("control_hz", "v"),
                "--gradient-steps": ("gradient_steps", "v"),
                "--promote-to": ("promote_to", "v"),
                "--init-from": ("init_from", "v"),
                "--gate-order": ("gate_order", "v"),
                "--learning-starts": ("learning_starts", "v"),
                "--bootstrap-random": ("bootstrap_random", "v"),
                "--buffer-size": ("buffer_size", "v"),
                "--archive-every": ("archive_every", "v"),
                "--instances": ("instances", "v"),
                "--bootstrap": ("bootstrap", "v"),
                "--seats": ("seats", "v"), "--handover": ("handover", "v"),
                "--handover-patience": ("handover_patience", "v"),
                "--then-race": ("then_race", "b"),
                "--curriculum": ("curriculum", "b"),
                "--auto-rollback": ("auto_rollback", "b"),
                "--max-offset": ("max_offset", "v"),
                "--stuck-speed": ("stuck_speed", "v"),
                "--stuck-seconds": ("stuck_seconds", "v"),
                "--max-episode-s": ("max_episode_s", "v"),
                "--w-weave": ("w_weave", "v"),
                "--w-reversal": ("w_reversal", "v"),
                "--cp-radius": ("cp_radius", "v"),
                "--line": ("line", "v"), "--regress-window": ("regress_window", "v"),
                "--regress-drop": ("regress_drop", "v"),
            }
            out: dict = {}
            i = 0
            while i < len(argv):
                spec = FLAGS.get(argv[i])
                if spec:
                    key, kind = spec
                    if kind == "b":
                        out[key] = True
                    elif i + 1 < len(argv):
                        out[key] = argv[i + 1]
                        i += 1
                i += 1
            if "line" in out:
                out["line"] = os.path.basename(out["line"])
            self._json({"running": True, "pid": pid, "args": out,
                        "raw": " ".join(argv[argv.index("train/train_sac.py") + 1:])
                        if "train/train_sac.py" in argv else " ".join(argv)})
            return

        if self.path.startswith("/api/model"):
            # What a model was trained AGAINST, so the panel can offer to
            # match it. Resuming with different settings is the failure the
            # .meta.json sidecar exists to catch - a model trained at 40Hz
            # with on/off pedals still LOADS at 20Hz with analogue ones, and
            # every action it has learned then means something else.
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            stage = (q.get("stage") or ["race"])[0]
            name = (q.get("name") or [""])[0] or (
                "sac_explore" if stage == "explore" else "sac_tm_v2")
            path = os.path.join(MODEL_DIR, name + ".meta.json")
            try:
                with open(path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                return self._json({"ok": True, "name": name, "meta": None})
            zp = os.path.join(MODEL_DIR, name + ".zip")
            return self._json({"ok": True, "name": name, "meta": meta,
                               "exists": os.path.isfile(zp)})

        if self.path.startswith("/api/splits"):
            # Section times for the map that is loaded. This is the number the
            # "optimise section by section" workflow runs on: a whole-lap time
            # says the run was slow, and the per-section gap says where.
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            with LINK.lock:
                latest = LINK.latest
            uid = ((q.get("map") or [None])[0] or (latest or {}).get("map")
                   or LINK.last_map)
            if not uid:
                return self._json({"ok": False, "err": "no map loaded"})
            sys.path.insert(0, os.path.join(ROOT, "tools"))
            try:
                import splits as splitlib
                rows = splitlib.load(uid)
            except SystemExit:
                return self._json({"ok": True, "map": uid, "sections": [],
                                   "episodes": 0,
                                   "note": "no split log yet - it is written "
                                           "once a run passes a checkpoint"})
            except Exception as ex:
                return self._json({"ok": False, "err": str(ex)[:200]})
            secs = splitlib.sections(rows)
            out, best_sum, complete = [], 0, True
            for i, vals in enumerate(secs):
                label = f"start->cp{i}" if i == 0 else f"cp{i - 1}->cp{i}"
                if not vals:
                    out.append({"label": label, "from_cp": i - 1, "to_cp": i,
                                "reached": 0})
                    complete = False
                    continue
                o = sorted(vals)
                best, med = o[0], o[len(o) // 2]
                best_sum += best
                out.append({"label": label, "from_cp": i - 1, "to_cp": i,
                            "best": best, "median": med, "gap": med - best,
                            "reached": len(vals)})
            fin = [int(r["race_time"]) for r in rows
                   if r.get("finished") and r.get("race_time")]
            return self._json({"ok": True, "map": uid, "episodes": len(rows),
                               "sections": out,
                               "best_lap": min(fin) if fin else None,
                               "theoretical": best_sum if complete else None})

        if self.path.startswith("/api/status"):
            LINK.ensure_landmarks()
            with LINK.lock:
                latest = LINK.latest
                age = time.time() - LINK.last_at if LINK.last_at else None
            # Which seat the controller view is looking at. In splitscreen
            # the top-level telemetry describes the CAMERA's car, so comparing
            # it against seat 0's pad reports a permanent disagreement that is
            # just the two halves being about different cars.
            from urllib.parse import parse_qs, urlparse
            seat = 0
            try:
                seat = int((parse_qs(urlparse(self.path).query).get("seat")
                            or [0])[0])
            except (TypeError, ValueError):
                seat = 0
            pad = pad_state(seat)
            self._json({
                "broker": LINK.sock is not None,
                "pad": pad is not None,
                "pad_state": pad,
                "pad_seat": seat,
                # Plugin health = telemetry actually arriving through the
                # broker, NOT a fresh socket to 8766. The plugin serves ONE
                # client (the broker has it); a probe here just lands in its
                # accept backlog unhandled and rots to CLOSE_WAIT. Polled
                # every second, that buried the plugin's listen socket under
                # thousands of dead half-connections and starved the broker's
                # own reconnects - the "telemetry flapping".
                "plugin": age is not None and age < 10,
                "telemetry_age": age,
                "latest": latest,
                "cp": LINK.cp_seen,
                "cp_total": len(LINK.cps),
                "trainer": {"running": TRAINER.running,
                            "pid": TRAINER.pid(),
                            "adopted": TRAINER.running and TRAINER.proc is None,
                            "log": TRAINER.tail(30)},
                "recorder": {"running": RECORDER.running,
                             "log": RECORDER.tail(12)},
                "fleet_job": {"running": FLEET.running,
                              "pid": FLEET.pid(),
                              "adopted": FLEET.running and FLEET.proc is None,
                              "log": FLEET.tail(12)},
                "lines": list_lines(),
                "line_info": lines_with_maps(),
                "fleet": fleet_status(),
                # The live map's own checkpoint/finish positions. Drawn
                # independently of the reference line, which may belong to a
                # different map entirely - that combination is exactly what
                # produced "a straight line with no checkpoints" on a track
                # with two corners and two gates.
                "landmarks": {"map": LINK.lm_map,
                              "checkpoints": LINK.cps,
                              "finish": LINK.finish},
                "seats": seat_status(latest),
                "venv": venv_ready(),
            })
            return

        if self.path == "/api/nn":
            # Written atomically by the probe, so a partial read is impossible;
            # a missing file just means no trainer is running.
            try:
                with open(os.path.join(ROOT, "logs", "nn_state.json"), "rb") as f:
                    self._send(200, f.read(), "application/json")
            except OSError:
                self._json({"idle": True})
            return

        if self.path.startswith("/api/why"):
            # ?n= how many episodes to return. The default stays 25 for the WHY
            # overlay, which shows a handful in detail - but the bar's rolling
            # windows are only as long as what it can see here, so asking for
            # win=100 there silently capped at 25 and the percentages were over
            # a window the label did not describe.
            from urllib.parse import parse_qs, urlparse
            try:
                n = int((parse_qs(urlparse(self.path).query).get("n")
                         or ["25"])[0])
            except ValueError:
                n = 25
            n = max(1, min(500, n))
            self._json({"entries": tail_jsonl(
                os.path.join(ROOT, "logs", "why.jsonl"), n)})
            return

        if self.path == "/api/runstats":
            self._json(run_stats())
            return

        if self.path == "/api/lineage":
            # Which model is driving, when its lineage began, and what it was
            # built on. Written by train_sac at startup.
            #
            # NOT /api/model - that name is already taken by an older route
            # (line ~1187) which matches with startswith(), so anything added
            # under it here is unreachable.
            try:
                with open(os.path.join(ROOT, "logs", "model.json")) as fh:
                    self._json(json.load(fh))
            except (OSError, ValueError):
                self._json({})
            return

        if self.path == "/api/handover":
            # How far the explore stage is from handing over to the racer.
            # Written by train.handover.HandoverWatch; absent when the run is
            # not an explore stage with --handover, which the overlay reads as
            # "no countdown to show" rather than as an error.
            #
            # Guard against a STALE file: a run without --handover never
            # rewrites it, and a crashed --handover run leaves "done": true
            # frozen. Only trust it when it is at least as new as the live
            # trainer's identity file AND that trainer is still running -
            # otherwise the overlay shows a dead run's "handing over" and
            # best lap over whatever is driving now.
            hp = os.path.join(ROOT, "logs", "handover.json")
            mp = os.path.join(ROOT, "logs", "model.json")
            try:
                fresh = os.path.getmtime(hp) >= os.path.getmtime(mp) - 5
                pid = json.load(open(mp)).get("pid")
                alive = bool(pid) and os.path.exists(f"/proc/{pid}")
                if fresh and alive:
                    with open(hp) as fh:
                        self._json(json.load(fh))
                else:
                    self._json({})
            except (OSError, ValueError):
                self._json({})
            return

        if self.path == "/api/archive":
            self._json({"models": list_models(),
                        "archive": {m: list_archive(m) for m in list_models()}})
            return

        if self.path == "/api/runs":
            self._json(list_runs())
            return

        if self.path.startswith("/api/line"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            name = (q.get("name") or [""])[0]
            lines = list_lines()
            if name not in lines:
                name = lines[0] if lines else ""
            if not name:
                self._json({"points": [], "name": None})
                return
            try:
                with open(os.path.join(ROOT, "lines", name)) as f:
                    line_doc = json.load(f)
                pts = line_doc.get("points", [])
            except (OSError, json.JSONDecodeError) as ex:
                self._json({"points": [], "err": str(ex)})
                return
            # Only x and z: the path view is top-down, and shipping y for a
            # 700-point line every poll is bytes nobody reads.
            # No recorded line for the track that is loaded? Draw the
            # PROVISIONAL one instead - spawn, through the checkpoints, to the
            # finish - which is exactly what explore mode drives. Showing
            # another map's line over this map's checkpoints is what made a
            # two-corner track render as a dead straight line with no gates.
            live_uid = (LINK.latest or {}).get("map") or LINK.last_map
            provisional = False
            line_uid = line_doc.get("map")
            wrong_map = bool(live_uid and line_uid and line_uid != live_uid)
            if wrong_map and LINK.cps:
                try:
                    from env.mapdata import Gates, provisional_line
                    items = ([{"kind": "checkpoint", "pos": c, "order": i}
                              for i, c in enumerate(LINK.cps)]
                             + ([{"kind": "finish", "pos": LINK.finish}]
                                if LINK.finish else []))
                    pl = provisional_line(Gates(items))
                    pts = pl.points.tolist()
                    provisional = True
                    wrong_map = False
                except Exception:
                    pass

            # Never draw one track's line over another track's checkpoints.
            # If the best recorded line is for a different map and we could not
            # build a provisional from the live gates, send NO points - the
            # panel shows the mismatch instead of a stray straight line.
            if wrong_map:
                pts = []

            self._json({"name": name, "lines": lines,
                        "map": live_uid if provisional else line_uid,
                        "provisional": provisional,
                        "mismatch": wrong_map,
                        "live_map": live_uid,
                        "points": [[round(p[0], 1), round(p[2], 1)] for p in pts],
                        "checkpoints": [[c[0], c[2]] for c in LINK.cps],
                        "finish": ([LINK.finish[0], LINK.finish[2]]
                                   if LINK.finish else None)})
            return

        if self.path.startswith("/api/trace"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            uid = (q.get("map") or [""])[0]
            name = (q.get("name") or [""])[0]
            base = os.path.join(ROOT, "runs", uid, "traces")
            if not uid or os.sep in uid or not os.path.isdir(base):
                self._json({"traces": [], "samples": []})
                return
            names = sorted((f for f in os.listdir(base) if f.endswith(".json")),
                           reverse=True)
            out = {"traces": names[:60], "map": uid}
            if name and name in names:
                try:
                    with open(os.path.join(base, name)) as f:
                        t = json.load(f)
                    # fields: t,x,y,z,speed,steer,gas,brake,cp -> x,z,speed
                    out["samples"] = [[s[1], s[3], s[4]] for s in t["samples"]]
                    out["meta"] = {k: t.get(k) for k in
                                   ("race_time", "distance", "finished",
                                    "checkpoints", "episode")}
                except (OSError, json.JSONDecodeError, IndexError, KeyError):
                    out["samples"] = []
            self._json(out)
            return

        if self.path.startswith("/api/config"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            with LINK.lock:
                live = (LINK.latest or {}).get("map") or LINK.last_map
            uid = (q.get("map") or [live or "default"])[0]
            # Follow the running trainer unless asked otherwise, so the knobs
            # on screen are the ones actually in force.
            profile = (q.get("profile") or [active_profile()])[0]
            self._json({
                "map": uid,
                "profile": profile,
                "stage": "explore" if profile == "explore" else "race",
                "config_file": config_name(uid, profile) + ".json",
                "live_map": live,
                "maps": list_configs(),
                "config": read_config(config_name(uid, profile)),
                "defaults": tuning.DEFAULTS,
                # Three tiers, narrowest first: what this map has actually
                # been driven on, what any Trackmania track plausibly uses,
                # and the whole engine enum.
                "seen_materials": seen_materials(uid),
                "materials": list(surfaces.TUNABLE),
                "all_materials": [m for m in surfaces.MATERIALS
                                  if not m.endswith("_Deprecated")],
                "groups": {m: surfaces.GROUPS.get(m, "other")
                           for m in surfaces.MATERIALS},
            })
            return

        if self.path.startswith("/api/route3d/traces"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            with LINK.lock:
                live = (LINK.latest or {}).get("map") or LINK.last_map
            uid = (q.get("map") or [live or ""])[0]
            n = int((q.get("n") or ["160"])[0])
            if not uid or os.sep in uid:
                self._json({"ok": False, "err": "no map"}, 400)
                return
            tdir = os.path.join(ROOT, "runs", uid, "traces")
            seats: dict = {}
            try:
                files = sorted(
                    (os.path.join(tdir, f) for f in os.listdir(tdir)
                     if f.endswith(".json")),
                    key=os.path.getmtime, reverse=True)[:max(1, n)]
            except OSError:
                files = []
            for fp in files:
                m = re.search(r"_i(\d)_", os.path.basename(fp))
                seat = int(m.group(1)) if m else 0
                try:
                    with open(fp) as fh:
                        doc = json.load(fh)
                except (OSError, ValueError):
                    continue
                fld = doc.get("fields") or []
                s = doc.get("samples") or []
                if len(s) < 2 or "x" not in fld:
                    continue
                xi, yi, zi = fld.index("x"), fld.index("y"), fld.index("z")
                stride = max(1, len(s) // 50)
                poly = [[round(r[xi], 1), round(r[yi], 1), round(r[zi], 1)]
                        for r in s[::stride]]
                seats.setdefault(str(seat), []).append(poly)
            self._json({"ok": True, "map": uid, "seats": seats})
            return

        if self.path.startswith("/api/route3d"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            with LINK.lock:
                live = (LINK.latest or {}).get("map") or LINK.last_map
            uid = (q.get("map") or [live or ""])[0]
            if not uid or os.sep in uid:
                self._json({"ok": False, "err": "no map"}, 400)
                return
            try:
                LINK.ensure_landmarks()
            except Exception:                            # noqa: BLE001
                pass
            self._json(route3d_payload(uid))
            return

        if self.path.startswith("/api/route"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            with LINK.lock:
                live = (LINK.latest or {}).get("map") or LINK.last_map
            uid = (q.get("map") or [live or ""])[0]
            if not uid or os.sep in uid:
                self._json({"ok": False, "err": "no map"}, 400)
                return
            try:
                LINK.ensure_landmarks()
            except Exception:                            # noqa: BLE001
                pass
            self._json(route_payload(uid))
            return

        if self.path.startswith("/api/replays"):
            q = ""
            if "?" in self.path:
                from urllib.parse import parse_qs, urlparse
                q = (parse_qs(urlparse(self.path).query).get("q") or [""])[0]
            found = find_replays(q)
            self._json({"replays": [
                {**r, "suggest": suggest_line_name(r["name"])} for r in found[:60]
            ], "total": len(found)})
            return

        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"ok": False, "err": "bad json"}, 400)
            return

        if self.path == "/api/cmd":
            cmd = str(body.get("cmd", "")).strip()
            if not cmd:
                self._json({"ok": False, "err": "no command"}, 400)
                return
            self._json(LINK.command(cmd))
            return

        if self.path == "/api/press":
            btn = str(body.get("button", "y")).lower()
            hold = float(body.get("hold_ms", 250))
            try:
                with socket.create_connection(PAD, timeout=2) as sk:
                    sk.sendall(f"press {btn} {hold:.0f}\n".encode())
                    sk.recv(64)
                self._json({"ok": True})
            except OSError as ex:
                self._json({"ok": False, "err": str(ex)})
            return

        if self.path == "/api/pad":
            try:
                with socket.create_connection(PAD, timeout=2) as s:
                    s.sendall(("act %.4f %.4f %.4f\n" % (
                        float(body.get("steer", 0.0)),
                        float(body.get("gas", 0.0)),
                        float(body.get("brake", 0.0)))).encode())
                    s.recv(64)
                self._json({"ok": True})
            except OSError as ex:
                self._json({"ok": False, "err": str(ex)})
            return

        if self.path == "/api/record/start":
            name = str(body.get("name", "line")).strip() or "line"
            mode = str(body.get("mode", "auto"))
            if mode not in ("auto", "live", "replay"):
                mode = "auto"
            out = os.path.join(ROOT, "lines", f"{name}.json")
            # record_line needs numpy, which only exists in the venv - the
            # panel itself runs on system python.
            if not venv_ready():
                self._json({"ok": False, "err": "venv not ready (numpy missing)"})
                return
            argv = [VENV_PY, "tools/record_line.py", out, "--mode", mode]
            if body.get("demo"):
                argv += ["--demo", os.path.join(ROOT, "demos", f"{name}.json")]
            if body.get("idle"):
                argv += ["--idle", str(float(body["idle"]))]
            if body.get("expect_time"):
                argv += ["--expect-time", str(float(body["expect_time"]))]
            self._json(RECORDER.start(argv))
            return

        if self.path == "/api/record/stop":
            self._json(RECORDER.stop())
            return

        if self.path == "/api/fleet/start":
            try:
                seats = int((body or {}).get("seats", 4))
            except (TypeError, ValueError):
                seats = 4
            seats = max(1, min(seats, 4))
            py = system_python()
            if not py:
                self._json({"ok": False, "err": "no system python with evdev - "
                            "install it: sudo pacman -S python-evdev"})
                return
            self._json(FLEET.start(
                [py, "tools/fleet.py", "--seats", str(seats)]))
            return

        if self.path == "/api/fleet/stop":
            self._json(FLEET.stop())
            return

        if self.path == "/api/train/start":
            if not venv_ready():
                self._json({"ok": False, "err": "torch/sb3 still installing"})
                return
            stage = str(body.get("stage", "race"))
            argv = [VENV_PY, "train/train_sac.py"]
            if stage == "explore":
                argv += ["--stage", "explore"]
                if body.get("gate_order"):
                    argv += ["--gate-order", str(body["gate_order"])]
            else:
                line = str(body.get("line", "")).strip()
                if not line:
                    self._json({"ok": False, "err": "pick a reference line"}, 400)
                    return
                # A line from another map makes every episode end on step one.
                # The env catches it too, but refusing here means you find out
                # before the game is handed over to a policy that cannot drive.
                uid = line_map(line)
                with LINK.lock:
                    latest = LINK.latest or {}
                live = latest.get("map")
                pos = latest.get("pos") if latest.get("car") else None
                dist = line_distance(line, pos)
                bad = (uid and live and uid != live) or \
                      (dist is not None and dist > FAR_FROM_LINE)
                if bad and not body.get("force"):
                    where = (f"the car spawns {dist:.0f}m from it"
                             if dist is not None else
                             f"it was recorded on map {uid}, but {live} is loaded")
                    self._json({"ok": False, "mismatch": True,
                                "err": f"'{line}' is not for this track - {where}. "
                                       f"Every episode would end instantly. "
                                       f"Record a line for this map, or use "
                                       f"explore mode, which needs none."})
                    return
                argv += ["--line", os.path.join(ROOT, "lines", line)]
            if int(body.get("seats") or 1) > 1:
                argv += ["--seats", str(int(body["seats"]))]
            elif body.get("instances"):
                argv += ["--instances", str(int(body["instances"]))]
            # Explore only: stop once it has found the finish and stopped
            # improving, build the racer's line from the best run, and (with
            # then_race) restart straight into the race stage.
            if stage == "explore" and int(body.get("handover") or 0) > 0:
                argv += ["--handover", str(int(body["handover"]))]
                if body.get("handover_patience"):
                    argv += ["--handover-patience",
                             str(int(body["handover_patience"]))]
                if body.get("then_race"):
                    argv += ["--then-race"]
            if body.get("curriculum"):
                argv += ["--curriculum"]
            if body.get("auto_rollback"):
                argv += ["--auto-rollback"]
            if body.get("bootstrap"):
                argv += ["--bootstrap", str(body["bootstrap"])]
            # The warm-up counts toward the step budget. If it is >= steps the
            # run pursuit-drives to the limit and stops having done ZERO
            # gradient steps - it looks like training that never learns.
            _steps = int(body.get("steps") or 0)
            _ls = int(body.get("learning_starts") or 0)
            if _steps and _ls and _ls >= _steps:
                self._json({"ok": False,
                            "err": f"warm-up length ({_ls}) must be well below "
                                   f"total steps ({_steps}) - the warm-up "
                                   f"counts toward the budget, so as set the "
                                   f"run would warm up and then stop without "
                                   f"a single gradient step. Raise steps or "
                                   f"lower the warm-up."}, 400)
                return
            if body.get("steps"):
                argv += ["--steps", str(int(body["steps"]))]
            if body.get("resume"):
                argv += ["--resume"]
            else:
                # Starting from scratch overwrites the model of that name on
                # the first save. Move the old one aside first: a policy that
                # took hours of real driving to produce should never be lost
                # to an unticked checkbox, and "it was archived" is a much
                # better outcome than "are you sure?".
                name = (str(body.get("name") or "").strip()
                        or ("sac_explore" if stage == "explore"
                            else "sac_tm_v2"))
                stamp = time.strftime("%Y%m%d-%H%M%S")
                arch = os.path.join(MODEL_DIR, "archive", name)
                for ext in (".zip", "_buffer.pkl", ".meta.json"):
                    src = os.path.join(MODEL_DIR, name + ext)
                    if not os.path.isfile(src):
                        continue
                    os.makedirs(arch, exist_ok=True)
                    dst = os.path.join(arch, f"replaced_{stamp}{ext}")
                    try:
                        os.replace(src, dst)
                        print(f"archived {name}{ext} -> {dst}", flush=True)
                    except OSError as ex:
                        return self._json(
                            {"ok": False,
                             "err": f"could not archive the existing "
                                    f"{name}{ext}: {ex}"}, 500)
            if body.get("name"):
                argv += ["--name", str(body["name"])]
            # Machine-shaped settings. These do NOT live in the per-map config
            # (they set buffer timing and the update ratio), so the panel has
            # to pass them at launch or the run silently reverts to 40Hz / auto
            # gradient steps - the combination that let slip climb.
            if body.get("control_hz") not in (None, "", 0):
                argv += ["--control-hz", str(float(body["control_hz"]))]
            if str(body.get("gradient_steps") or "").strip():
                argv += ["--gradient-steps", str(int(body["gradient_steps"]))]
            if str(body.get("promote_to") or "").strip():
                argv += ["--promote-to", str(body["promote_to"]).strip()]
            # init_from only bites on a fresh run (no --resume); harmless to
            # pass alongside resume, the trainer ignores it when it loads a
            # local checkpoint.
            if str(body.get("init_from") or "").strip():
                argv += ["--init-from", str(body["init_from"]).strip()]
            # Warm-up length / noise, buffer size, snapshot cadence - launch
            # only, same reason as control-hz above.
            if body.get("learning_starts") not in (None, "", 0):
                argv += ["--learning-starts", str(int(body["learning_starts"]))]
            if body.get("bootstrap_random") not in (None, ""):
                argv += ["--bootstrap-random", str(float(body["bootstrap_random"]))]
            if body.get("buffer_size") not in (None, "", 0):
                argv += ["--buffer-size", str(int(body["buffer_size"]))]
            if body.get("archive_every") not in (None, "", 0):
                argv += ["--archive-every", str(float(body["archive_every"]))]
            # Tuning knobs, all optional. Phase 2 moves these into a per-map
            # config the running trainer re-reads; for now they are set at
            # launch, which is why the panel disables them while it runs.
            for key, flag in (("max_offset", "--max-offset"),
                              ("stuck_speed", "--stuck-speed"),
                              ("stuck_seconds", "--stuck-seconds"),
                              ("max_episode_s", "--max-episode-s"),
                              ("w_weave", "--w-weave"),
                              ("w_reversal", "--w-reversal"),
                              ("cp_radius", "--cp-radius"),
                              ("giveup_settle_ms", "--giveup-settle-ms")):
                if body.get(key) not in (None, ""):
                    argv += [flag, str(float(body[key]))]
            self._json(TRAINER.start(argv))
            return

        if self.path == "/api/train/stop":
            self._json(TRAINER.stop())
            return

        if self.path == "/api/route/markers":
            # Reward markers, placed by clicking the route editor.
            #
            # Stored as a DISTANCE along the line, not an XZ point, because
            # that is what the reward compares against (max_s). The client
            # sends a world XZ from the click and the distance is resolved
            # HERE, against the same roadtrace cache the trainer uses - doing
            # it in the browser would mean two implementations of "how far
            # along is this", and they would drift.
            uid = (body or {}).get("map", "")
            marks_in = (body or {}).get("markers", [])
            if not uid or os.sep in uid or not isinstance(marks_in, list):
                return self._json({"ok": False, "err": "bad map/markers"}, 400)
            # The SPLICED line, so a marker sits where the reward will fire
            # after route edits, not where it would have on the raw trace.
            pts3, cum = spliced_line_3d(uid)
            if not pts3:
                return self._json({"ok": False,
                                   "err": f"no roadtrace for {uid}"}, 400)
            clean = []
            for m in marks_in[:32]:
                try:
                    bonus = float(m.get("bonus", 200.0))
                    if "s" in m and m.get("s") is not None:
                        at = float(m["s"])
                    else:
                        # nearest line point to the click, then its distance
                        cx, cz = float(m["xz"][0]), float(m["xz"][1])
                        best, bd = 0, None
                        for k, q in enumerate(pts3):
                            d = (q[0] - cx) ** 2 + (q[2] - cz) ** 2
                            if bd is None or d < bd:
                                bd, best = d, k
                        at = cum[best]
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                at = max(0.0, min(at, cum[-1]))
                entry = {"s": round(at, 1), "bonus": bonus}
                # Optional lateral limit. Absent = the marker has no width and
                # fires on progress alone; set it to require the car actually be
                # near the line when it passes.
                lim = m.get("max_offset")
                if lim is not None:
                    try:
                        entry["max_offset"] = float(lim)
                    except (TypeError, ValueError):
                        pass
                clean.append(entry)
            clean.sort(key=lambda m: m["s"])
            cfg_file = os.path.join(ROOT, "configs", f"{uid}.explore.json")
            try:
                with open(cfg_file) as fh:
                    cfg = json.load(fh)
            except (OSError, ValueError):
                cfg = {}
            cfg["markers"] = clean
            try:
                os.makedirs(os.path.dirname(cfg_file), exist_ok=True)
                tmp = cfg_file + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(cfg, fh, indent=2)
                os.replace(tmp, cfg_file)
            except OSError as ex:
                return self._json({"ok": False, "err": str(ex)}, 500)
            return self._json({"ok": True, "markers": clean})

        if self.path == "/api/route/edits":
            uid = (body or {}).get("map", "")
            patches = (body or {}).get("patches", [])
            if not uid or os.sep in uid or not isinstance(patches, list):
                return self._json({"ok": False, "err": "bad map/patches"}, 400)
            clean = []
            for p in patches[:64]:
                try:
                    a = [float(p["a"][0]), float(p["a"][1])]
                    b = [float(p["b"][0]), float(p["b"][1])]
                    pts = [[float(x), float(z)] for x, z in p["points"]][:400]
                    if not pts:
                        continue
                    clean.append({"a": a, "b": b, "points": pts,
                                  "kind": "jump" if p.get("kind") == "jump"
                                  else "road"})
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
            try:
                path = os.path.join(ROOT, "maps", f"{uid}.route_edits.json")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # Keep the LAST NON-EMPTY version as .bak. The route editor's
                # Revert button saves an empty patch list with no undo, so a
                # misclick wipes a lot of hand work; this makes it recoverable.
                if os.path.isfile(path):
                    try:
                        prev = json.load(open(path))
                        if prev.get("patches"):
                            shutil.copy2(path, path + ".bak")
                    except (OSError, ValueError):
                        pass
                with open(path, "w") as f:
                    json.dump({"map": uid, "patches": clean}, f, indent=1)
            except OSError as ex:
                return self._json({"ok": False, "err": str(ex)[:200]}, 500)
            return self._json({"ok": True, "patches": len(clean)})

        if self.path == "/api/unmark":
            name = (body or {}).get("name", "")
            with LINK.lock:
                latest = LINK.latest
            uid = ((body or {}).get("map") or (latest or {}).get("map")
                   or LINK.last_map)
            from env.config import config_path
            # Same file /api/mark wrote to - the running trainer's profile.
            path = config_path(ROOT, uid, active_profile())
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                return self._json({"ok": False, "err": "no config"}, 404)
            if data.get("marks", {}).pop(name, None) is None:
                return self._json({"ok": False, "err": "no such mark"}, 404)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return self._json({"ok": True, "marks": data.get("marks", {})})

        if self.path == "/api/calibrate":
            # Runs the real test: move each pad, see which seat answers.
            import subprocess
            try:
                r = subprocess.run(
                    [VENV_PY, "tools/calibrate_seats.py", "--json"],
                    cwd=ROOT, capture_output=True, text=True, timeout=120)
                mapping = json.loads((r.stdout or "{}").strip().splitlines()[-1])
                return self._json({"ok": True, "mapping": mapping})
            except Exception as ex:
                return self._json({"ok": False, "err": str(ex)[:200]}, 500)

        if self.path == "/api/mark":
            # Record where a car is, under a name. Stored as a world position
            # rather than an arc length - see tools/mark.py for why.
            name = (body or {}).get("name", "").strip()
            if not name:
                return self._json({"ok": False, "err": "need a name"}, 400)
            try:
                seat = int((body or {}).get("seat", 0))
            except (TypeError, ValueError):
                seat = 0
            with LINK.lock:
                latest = LINK.latest
            uid = ((body or {}).get("map")
                   or (latest.get("map") if latest else None) or LINK.last_map)
            pos = None
            # 1) An explicit point clicked on the route-editor map. Y is taken
            #    from the reference line so mark_from/mark_to resolve sanely.
            xz = (body or {}).get("xz")
            if xz and len(xz) == 2:
                try:
                    cx, cz = float(xz[0]), float(xz[1])
                    p3, _ = spliced_line_3d(uid)
                    y = 0.0
                    if p3:
                        y = min(p3, key=lambda q: (q[0] - cx) ** 2
                                + (q[2] - cz) ** 2)[1]
                    pos = [cx, y, cz]
                except (TypeError, ValueError):
                    pos = None
            # 2) Otherwise, where a car actually is. In splitscreen the
            #    top-level record is the CAMERA's car and carries no position;
            #    the seat's own car is in players[seat].
            if pos is None and latest:
                players = latest.get("players") or []
                if 0 <= seat < len(players) and players[seat].get("pos"):
                    pos = players[seat]["pos"]
                elif "pos" in latest:
                    pos = latest["pos"]
            if not pos:
                return self._json(
                    {"ok": False, "err": "no position - click the map, or load "
                     f"a map with a car on seat {seat}"}, 409)
            # Write to the config the RUNNING trainer actually reads, not a
            # hardcoded race file nothing has open.
            from env.config import config_path
            profile = active_profile()
            path = config_path(ROOT, uid, profile)
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                data = {}
            pos = [round(float(x), 2) for x in pos]
            data.setdefault("marks", {})[name] = pos
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return self._json({"ok": True, "name": name, "pos": pos,
                               "map": uid, "seat": seat,
                               "profile": profile or "race",
                               "marks": data["marks"]})

        if self.path == "/api/config":
            uid = str(body.get("map", "")).strip()
            cfg = body.get("config")
            if not isinstance(cfg, dict):
                self._json({"ok": False, "err": "no config"}, 400)
                return
            # "Save as": write to a caller-named file instead of <uid>.<profile>.
            # The name is the config basename as it appears in the map list
            # (e.g. "VO0nIz….explore" or "mytune.race"); no path separators.
            save_as = str(body.get("save_as", "")).strip()
            if save_as:
                if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", save_as) \
                        or save_as in (".", ".."):
                    self._json({"ok": False,
                                "err": "name must be letters/digits/._- only"},
                               400)
                    return
                self._json(write_config(save_as, cfg))
                return
            # No trainer restart: the env stats this file about once a second
            # and applies whatever it finds on the next step.
            # Write the profile the trainer is reading, not always the race
            # one - otherwise an explore run sees none of these edits.
            profile = body.get("profile")
            if profile is None:
                profile = active_profile()
            self._json(write_config(config_name(uid, profile), cfg))
            return

        if self.path == "/api/archive/restore":
            if TRAINER.running:
                self._json({"ok": False,
                            "err": "stop the trainer before restoring"})
                return
            name = str(body.get("model", "sac_tm"))
            snap = str(body.get("snapshot", ""))
            src = os.path.join(ROOT, "models", "archive", name, snap + ".zip")
            # Never let a crafted name walk out of the archive directory.
            if not snap or not os.path.isfile(src) or os.sep in snap:
                self._json({"ok": False, "err": "no such snapshot"}, 400)
                return
            dst = os.path.join(ROOT, "models", name + ".zip")
            try:
                # Keep whatever is currently live, so a restore is undoable.
                if os.path.exists(dst):
                    shutil.copy2(dst, dst + ".pre-restore")
                shutil.copy2(src, dst)
            except OSError as ex:
                self._json({"ok": False, "err": str(ex)})
                return
            # The replay buffer is deliberately left alone: it is off-policy
            # data and stays valid for whichever policy you roll back to.
            self._json({"ok": True, "restored": snap,
                        "note": "resume training to load it"})
            return

        self._json({"ok": False, "err": "unknown endpoint"}, 404)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"control panel on http://127.0.0.1:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())

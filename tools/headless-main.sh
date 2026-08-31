#!/usr/bin/env bash
# Run the MAIN (paid, Developer-mode) Trackmania on its own headless X display
# so its window is always focused and never frame-throttled.
#
# WHY: on the Wayland desktop, KWin throttles the frame rate of an unfocused /
# occluded window. TM2020 ramps analogue steering per frame, so a throttled
# window never finishes the ramp - the trainer log shows "the game is applying
# 13-23% of the steering we send". A bare Xvfb has no compositor and nothing
# else on it, so the game is always the focused window and the ramp completes.
#
#   tools/headless-main.sh up          start Xvfb :99 + x11vnc, launch the game,
#                                      force Developer mode, auto-load $MAP
#   tools/headless-main.sh down         stop the game + Xvfb + x11vnc
#   tools/headless-main.sh vnc          print how to view it
#   tools/headless-main.sh map "<p>"    send `playmap <p>` to the running game
#                                       (use this to find the path the game wants)
#
# `up` forces Developer mode (tools/op-mode.sh) and, once the plugin is
# listening, sends `playmap $MAP` so the track loads with no menu clicking.
# That gets you a ONE-CAR solo run hands-free. Splitscreen (4 cars) still
# needs the lobby built by hand for now - that is a separate plugin command.
# The pad servers and broker are NOT touched by this script.
set -euo pipefail

DISP=":99"
VNC_PORT=5999
COMPAT="/mnt/4TB/SteamLibrary/steamapps/compatdata/2225070"
BROKER_PORT=8767
# Path AS THE GAME SEES IT: relative to Documents/Trackmania/Maps/, NO "Maps/"
# prefix. Confirmed working. Empty = do not auto-load.
#   "My Maps/Wide left, mid right - road.Map.Gbx"   -> VO0nIz (default)
#   "My Maps/Straight Line - Road.Map.Gbx"
#   "My Maps/Booster left right - road.Map.Gbx"
# No colon in the expansion: TMAI_MAP="" (explicitly empty) must mean "do not
# auto-load" - you're picking a Campaign track by hand. Unset = the default.
MAP="${TMAI_MAP-My Maps/Wide left, mid right - road.Map.Gbx}"

# send one line to the broker, print ~2s of whatever comes back (no nc on host)
broker_send() {
  python3 - "$BROKER_PORT" "$1" <<'PY'
import socket, sys, time
port, line = int(sys.argv[1]), sys.argv[2]
try:
    s = socket.create_connection(("127.0.0.1", port), timeout=3)
except OSError as e:
    print(f"(broker unreachable: {e})"); sys.exit(1)
s.sendall((line + "\n").encode()); s.settimeout(1.5)
end, buf = time.time() + 2, b""
while time.time() < end:
    try:
        d = s.recv(65536)
    except socket.timeout:
        break
    if not d:
        break
    buf += d
sys.stdout.write(buf.decode("utf-8", "replace"))
PY
}
GAME_DIR="/mnt/4TB/SteamLibrary/steamapps/common/Trackmania"
PROTON="${TMAI_PROTON:-$HOME/.local/share/Steam/steamapps/common/Proton - Experimental}"
STEAM_CLIENT="$HOME/.local/share/Steam"
LOG="$HOME/tm2020-ai/logs/headless-main.log"

xvfb_alive() {
  # Actually reachable, not just a matching process name. A defunct Xvfb
  # still shows in pgrep but rejects connections - that is what made the
  # guard skip a real start and everything downstream XIO-110.
  DISPLAY="$DISP" timeout 3 xdpyinfo >/dev/null 2>&1
}

start_xvfb() {
  if ! xvfb_alive; then
    echo "starting Xvfb $DISP"
    pkill -f "Xvfb $DISP" 2>/dev/null || true       # clear any defunct one
    sleep 1
    setsid Xvfb "$DISP" -screen 0 1920x1080x24 -nolisten tcp >/dev/null 2>&1 &
    for _ in $(seq 1 15); do xvfb_alive && break; sleep 1; done
    xvfb_alive || { echo "Xvfb $DISP would not come up" >&2; exit 1; }
  fi
  pgrep -f "x11vnc.*$DISP" >/dev/null || {
    echo "starting x11vnc on 127.0.0.1:$VNC_PORT"
    # env -u WAYLAND_DISPLAY: x11vnc refuses to start if it sees a Wayland
    # session in the env, even pointed at a plain Xvfb. setsid: this script's
    # caller may exit; the VNC server must outlive it.
    setsid env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
      x11vnc -display "$DISP" -rfbport "$VNC_PORT" -forever -shared -nopw \
             -localhost -quiet >/dev/null 2>&1 &
    sleep 2
  }
  # A window manager is REQUIRED, not optional: on a bare Xvfb with no WM the
  # game window never receives X input focus, and TM2020 throttles its frame
  # rate when unfocused. That throttle is exactly the "the game is applying
  # 20% of the steering we send" symptom - the per-frame steering ramp never
  # completes. openbox is tiny and does nothing but hand out focus.
  pgrep -f "openbox.*DISPLAY=$DISP\|openbox" >/dev/null 2>&1 || {
    if command -v openbox >/dev/null; then
      echo "starting openbox on $DISP (focus == full frame rate == full steering)"
      DISPLAY="$DISP" setsid openbox >/dev/null 2>&1 &
      sleep 1
    else
      echo "WARNING: no openbox - without a WM the game will be frame-throttled" >&2
    fi
  }
}

case "${1:-}" in
up)
  [ -d "$COMPAT" ] || { echo "no compat dir: $COMPAT" >&2; exit 1; }
  [ -x "$PROTON/proton" ] || { echo "no Proton at $PROTON" >&2; exit 1; }
  start_xvfb

  if pgrep -f "Trackmania.exe" >/dev/null; then
    echo "a Trackmania.exe is already running - stop it first (or run 'down')" >&2
    echo "  running: $(pgrep -af Trackmania.exe | head -1)" >&2
    exit 1
  fi

  # Force Developer mode in the file BEFORE launch, so TMAITelemetry loads
  # without anyone clicking through the laggy overlay menu over VNC.
  "$(dirname "$0")/op-mode.sh" dev || {
    echo "could not set Developer mode - check op-mode.sh output above" >&2
    exit 1
  }

  echo "launching Trackmania on $DISP -> $LOG"
  mkdir -p "$(dirname "$LOG")"
  {
    echo
    echo "--- headless-main up $(date -Is) ---"
  } >>"$LOG"

  # gamescope wraps the game in a GPU-accelerated nested compositor. The game
  # renders into gamescope (fast, its own Xwayland), and only gamescope's
  # single composited frame is presented to the Xvfb SDL window - so the
  # software X server is no longer trying to service the game's full Vulkan
  # present traffic, which is what produced "XIO: fatal IO error 110" and the
  # fallback to :0. Set GAMESCOPE=0 to go back to the raw Xvfb launch.
  common_env=(
    STEAM_COMPAT_DATA_PATH="$COMPAT"
    STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM_CLIENT"
    DXVK_STATE_CACHE_PATH="$COMPAT/dxvk-cache"
  )
  proton_cmd=("$PROTON/proton" waitforexitandrun "$GAME_DIR/Trackmania.exe"
              -upc_steam_free_package_id 62710 -uplay_steam_mode)

  # --- Steam-client launch (default) --------------------------------------
  # A direct Proton launch does NOT bring Wine's controller layer up, so the
  # game sees ZERO gamepads - no driving, and 4-player splitscreen refuses to
  # start ("connect 3 more controllers"). Launching through the Steam client
  # is the only config where the virtual pads have ever worked. Set
  # STEAM_LAUNCH=0 for the old direct-Proton path (fine for solo playmap runs
  # that never touch a pad, e.g. inference-only).
  if [ "${STEAM_LAUNCH:-1}" = 1 ] && command -v steam >/dev/null; then
    echo "  via the Steam client (headless on $DISP) - controller layer"
    if ! pgrep -f 'ubuntu12_32/steam\b' >/dev/null 2>&1; then
      DISPLAY="$DISP" WAYLAND_DISPLAY= XAUTHORITY= SDL_VIDEODRIVER=x11 \
        setsid steam -silent -nochatui -nofriendsui >>"$LOG" 2>&1 &
      echo "  waiting for Steam to be ready (first run can take a few min)…"
      # steam.pipe is a NAMED PIPE (prw-), not a socket - test -p, not -S.
      # The old `-S` was never true, so this loop always burned the full
      # 80*3s before falling through. Also accept the steamwebhelper being up
      # as "ready", which is the real signal the client can take a game URL.
      for _ in $(seq 1 80); do
        { [ -p "$HOME/.steam/steam.pipe" ] || [ -S "$HOME/.steam/steam.pipe" ]; } \
          && pgrep -f 'steamwebhelper' >/dev/null 2>&1 && break
        sleep 3
      done
      sleep 8
    fi
    DISPLAY="$DISP" WAYLAND_DISPLAY= XAUTHORITY= setsid steam "steam://rungameid/2225070" >>"$LOG" 2>&1 &
    GAMEPID=$!
    echo "pid $GAMEPID (steam url launch)  (watch: tail -f $LOG)"
    echo
    echo "view it:   vncviewer 127.0.0.1:$VNC_PORT"
    if [ -n "$MAP" ]; then
      ( for _ in $(seq 1 90); do
          if broker_send ping 2>/dev/null | grep -q pong; then
            sleep 3; broker_send "playmap $MAP" >/dev/null 2>&1
            echo "sent: playmap $MAP"; exit 0
          fi
          sleep 5
        done
        echo "plugin never came up on $BROKER_PORT" >&2 ) &
    fi
    echo "next:      confirm the game shows controllers, then splitscreen.py"
    exit 0
  fi

  if [ "${GAMESCOPE:-1}" = 1 ] && command -v gamescope >/dev/null; then
    echo "  via gamescope (nested compositor on $DISP)"
    DISPLAY="$DISP" WAYLAND_DISPLAY= XAUTHORITY= SDL_VIDEODRIVER=x11 \
    env "${common_env[@]}" \
    setsid gamescope -W 1920 -H 1080 -w 1920 -h 1080 -r 60 \
      --backend sdl --force-windows-fullscreen \
      -- "${proton_cmd[@]}" >>"$LOG" 2>&1 &
  else
    echo "  raw Xvfb launch (no gamescope)"
    DISPLAY="$DISP" WAYLAND_DISPLAY= XAUTHORITY= SDL_VIDEODRIVER=x11 \
    env "${common_env[@]}" \
    setsid "${proton_cmd[@]}" >>"$LOG" 2>&1 &
  fi
  GAMEPID=$!
  echo "pid $GAMEPID  (watch: tail -f $LOG)"
  echo
  echo "view it:   vncviewer 127.0.0.1:$VNC_PORT     (or any VNC client)"

  # Auto-load the map once the plugin is listening on the broker. Solo, so
  # this is a one-car hands-free run. For 4 cars, build the splitscreen lobby
  # by hand on the VNC display after this.
  if [ -n "$MAP" ]; then
    ( echo "waiting for the plugin (broker $BROKER_PORT) so playmap can be sent…"
      for _ in $(seq 1 60); do
        if broker_send ping 2>/dev/null | grep -q pong; then
          sleep 3   # let it settle at the menu
          broker_send "playmap $MAP" >/dev/null 2>&1
          echo "sent: playmap $MAP"
          exit 0
        fi
        sleep 5
      done
      echo "plugin never came up on $BROKER_PORT - load the map by hand, or:" >&2
      echo "  tools/headless-main.sh map \"<path>\"" >&2
    ) &
  fi
  echo "next:      for 4 cars, build the splitscreen lobby on the VNC display."
  ;;
map)
  MAPARG="${2:?usage: $0 map \"<path as the game sees it>\"}"
  out=$(broker_send "playmap $MAPARG")
  echo "$out" | grep -o '"cmd":"playmap"[^}]*}' \
    || echo "(no playmap ack seen - is the game up with TMAITelemetry loaded?)"
  ;;
down)
  pkill -9 -f "2225070" 2>/dev/null && echo "stopped game (2225070)" || true
  sleep 1
  pkill -9 -f "common/Trackmania/Trackmania" 2>/dev/null || true
  pkill -9 -f "proton waitforexitandrun.*Trackmania" 2>/dev/null || true
  pkill -9 -f "gamescope .*Trackmania\|gamescope -W 1920" 2>/dev/null \
    && echo "stopped gamescope" || true
  sleep 1
  DISPLAY="$DISP" pkill -f "openbox" 2>/dev/null && echo "stopped openbox" || true
  pkill -f "x11vnc.*$DISP" 2>/dev/null && echo "stopped x11vnc" || true
  pkill -f "Xvfb $DISP" 2>/dev/null && echo "stopped Xvfb $DISP" || true
  ;;
vnc)
  echo "vncviewer 127.0.0.1:$VNC_PORT"
  ;;
*)
  echo "usage: $0 {up|down|vnc|map \"<path>\"}" >&2
  exit 1
  ;;
esac

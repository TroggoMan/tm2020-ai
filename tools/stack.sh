#!/usr/bin/env bash
# tools/stack.sh - one switch for the whole TM2020-AI stack.
#
# Composes the pieces that were separate scripts before:
#   plumbing   pad servers + broker + web panel   (was: tm2020-sacAI)
#   dev game   paid account, Developer mode, headless :99, TMAITelemetry
#              plugin, map autoloaded              (was: tools/headless-main.sh)
#   school     free account (tmai01), School mode, headless :100, the signed
#              SAC_GetData plugin + our adapter    (EXPERIMENTAL - see below)
#
# USAGE
#   tools/stack.sh up                 plumbing + dev game. The usual thing.
#   tools/stack.sh up --no-dev        plumbing only (pads, broker, panel)
#   tools/stack.sh up --school        ...also the School-mode instance
#   tools/stack.sh down               stop everything stack.sh started
#   tools/stack.sh down --game        stop only the game(s), keep plumbing up
#   tools/stack.sh status            what is up right now
#
# OPTIONS for `up`
#   --seats N      splitscreen seats / pad servers      (default 4)
#   --map "PATH"   map path as the game sees it, passed straight to `playmap`
#                  (default: the VO0nIz wide-left test map). Use --map none
#                  (or --map "") to skip autoload - you're picking a Campaign
#                  track by hand over VNC/Sunshine.
#   --no-panel     skip the web panel
#   --no-dev       skip the dev game (plumbing only, or plumbing + --school)
#   --school       also bring up the School-mode instance
#
# Every wait below says WHAT it is waiting for and WHY, and counts up in
# seconds - so a slow step (Steam cold start, first-launch shader cache build,
# 2-4 min) reads as progress, not as a hang. Nothing here creates Ubisoft
# accounts or logs in; those stay manual on purpose (a scripted login loop is
# what gets a batch of accounts banned).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/logs"
PIDFILE="$LOGDIR/stack.pids"
mkdir -p "$LOGDIR"

# ---------------------------------------------------------------- pretty output
_c() { printf '%b' "$1"; }
hd()   { _c "\n\033[1;36m==> $*\033[0m\n"; }
why()  { _c "\033[2m    why wait: $*\033[0m\n"; }
ok()   { _c "\033[1;32m    ok:\033[0m $*\n"; }
bad()  { _c "\033[1;31m    !!:\033[0m $*\n"; }
info() { printf '    %s\n' "$*"; }

track() { printf '%s:%s\n' "$1" "$2" >> "$PIDFILE"; }

# ------------------------------------------------------------------- tiny probes
port_up() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3>&-; }

panel_up() { curl -sf "http://127.0.0.1:8080/api/status" >/dev/null 2>&1; }

broker_pong() {
  python3 - <<'PY' 2>/dev/null
import socket, sys
try:
    s = socket.create_connection(("127.0.0.1", 8767), timeout=2)
    s.sendall(b"ping\n"); s.settimeout(2)
    sys.exit(0 if b"pong" in s.recv(4096) else 1)
except Exception:
    sys.exit(1)
PY
}

dpy_up() { DISPLAY="$1" timeout 3 xdpyinfo >/dev/null 2>&1; }

# wait_for <timeout_s> <label> -- <cmd...>
#   polls <cmd> once a second, prints an elapsed/limit line every 5s, and
#   returns 1 (without aborting the script) if <timeout_s> passes first.
wait_for() {
  local timeout="$1" label="$2"; shift 2
  [ "${1:-}" = "--" ] && shift
  local start=$SECONDS last=-5 now
  while :; do
    if "$@" >/dev/null 2>&1; then
      ok "$label  (after $((SECONDS - start))s)"
      return 0
    fi
    now=$((SECONDS - start))
    if [ "$now" -ge "$timeout" ]; then
      bad "$label - not ready after ${timeout}s; moving on (see logs/)"
      return 1
    fi
    if [ $((now - last)) -ge 5 ]; then
      last=$now
      _c "\033[2m    ... ${now}s / ${timeout}s  waiting on ${label}\033[0m\n"
    fi
    sleep 1
  done
}

sys_python() {   # the interpreter that can see evdev + make uinput devices
  local c
  for c in /usr/bin/python3 "$(command -v python3 || true)"; do
    [ -n "$c" ] || continue
    "$c" -c 'import evdev' 2>/dev/null && { echo "$c"; return 0; }
  done
  return 1
}

# ============================================================================ up
do_up() {
  local SEATS=4 WANT_PANEL=1 WANT_DEV=1 WANT_SCHOOL=0
  local MAP="My Maps/Wide left, mid right - road.Map.Gbx"
  while [ $# -gt 0 ]; do
    case "$1" in
      --seats)   SEATS="$2"; shift 2 ;;
      --map)     MAP="$2"; shift 2 ;;
      --no-panel) WANT_PANEL=0; shift ;;
      --no-dev)  WANT_DEV=0; shift ;;
      --school)  WANT_SCHOOL=1; shift ;;
      *) bad "unknown option for up: $1"; exit 2 ;;
    esac
  done

  if [ -e "$PIDFILE" ]; then
    bad "$PIDFILE exists - stack.sh already brought things up."
    info "run 'tools/stack.sh down' first, or 'tools/stack.sh status'."
    exit 1
  fi
  : > "$PIDFILE"

  hd "System python with evdev  (pad servers need uinput, the venv can't)"
  local SYS_PY
  if ! SYS_PY="$(sys_python)"; then
    bad "no system python with evdev. Install it:  sudo pacman -S python-evdev"
    rm -f "$PIDFILE"; exit 1
  fi
  ok "$SYS_PY"

  # -- web panel -------------------------------------------------------------
  if [ "$WANT_PANEL" = 1 ]; then
    hd "Web panel   ->  http://127.0.0.1:8080"
    why "one place to watch telemetry, load maps, and start/stop training;"
    why "also serves the instance grid at /grid."
    if panel_up; then
      ok "already up"
    else
      python3 web/server.py 8080 >"$LOGDIR/panel.log" 2>&1 &
      track panel $!
      wait_for 25 "panel /api/status" -- panel_up
    fi
  fi

  # -- pads + broker ------------------------------------------------------------
  hd "Pad servers ($SEATS) + telemetry broker   (tools/fleet.py --seats $SEATS)"
  why "each splitscreen seat needs its OWN uinput gamepad so the game can tell"
  why "them apart. The broker holds the plugin's single client slot and"
  why "re-serves it, so the panel and the trainer can both read telemetry."
  if port_up 8765 && port_up 8767; then
    ok "pads + broker already listening - leaving them"
  else
    python3 tools/fleet.py --seats "$SEATS" >"$LOGDIR/fleet.log" 2>&1 &
    track fleet $!
    local i p
    for ((i = 0; i < SEATS; i++)); do
      p=$((8765 + 10 * i))
      wait_for 25 "seat $i pad :$p" -- port_up "$p"
    done
    wait_for 20 "broker :8767" -- port_up 8767
  fi

  # -- dev game --------------------------------------------------------------
  if [ "$WANT_DEV" = 1 ]; then
    hd "Trackmania - dev account, Developer mode, headless on :99"
    why "Developer mode loads the unsigned TMAITelemetry plugin (needs the paid"
    why "account). Headless :99 + openbox keeps the window focused so the"
    why "per-frame steering ramp actually completes. Steam cold start plus the"
    why "first-launch shader cache build can take 2-4 minutes - the counter"
    why "below is not stuck, the game just takes that long to reach the menu."
    if pgrep -f 'Trackmania\.exe' >/dev/null 2>&1; then
      bad "a Trackmania.exe is already running - not launching another."
      info "if it's the one you were checking things in, close it and re-run 'up'."
    else
      # --map none / --map "" -> skip playmap autoload (Campaign track picked by hand)
      case "$MAP" in none|"") MAP=""; info "autoload OFF - pick the track by hand on :99" ;; esac
      TMAI_MAP="$MAP" tools/headless-main.sh up 2>&1 | sed 's/^/    | /'
      if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        bad "headless-main.sh up returned an error (see above / logs/headless-main.log)"
      else
        track game-dev "$(pgrep -n -f 'steam://rungameid/2225070' || echo 0)"
        wait_for 300 "TMAITelemetry plugin :8766" -- port_up 8766
        wait_for 60  "broker sees the game (ping -> pong)" -- broker_pong
        info "map autoload is handled by headless-main:  playmap \"$MAP\""
        info "view it:  vncviewer 127.0.0.1:5999"
      fi
    fi
  fi

  # -- school instance ----------------------------------------------------------
  if [ "$WANT_SCHOOL" = 1 ]; then
    school_up "$WANT_DEV"
  fi

  # -- summary ----------------------------------------------------------------
  hd "Stack is up"
  [ "$WANT_PANEL" = 1 ] && info "panel   http://127.0.0.1:8080      (grid: /grid)"
  [ "$WANT_DEV" = 1 ]   && info "dev :99  vncviewer 127.0.0.1:5999"
  [ "$WANT_SCHOOL" = 1 ] && info "school :100  vncviewer 127.0.0.1:5900"
  info ""
  info "start the explorer when you're ready:"
  info "  .venv/bin/python train/train_sac.py --stage explore --seats $SEATS \\"
  info "     --bootstrap straight --name explore_\$(date +%m%d)"
  info ""
  info "stop it all:  tools/stack.sh down        (game only: tools/stack.sh down --game)"
}

# ------------------------------------------------------------------ school_up
# EXPERIMENTAL. Best-effort: checks its prerequisites, and if one is missing it
# prints exactly what to do by hand and returns without killing the rest of the
# stack.
school_up() {
  local dev_is_up="${1:-0}"
  local U=tmai01 DISP=:100 VNC=5901 PLUGPORT=8776   # :100 / 5901 = steam-instance's convention (5900 + N)
  hd "School-mode instance - $U on $DISP   (EXPERIMENTAL)"
  why "a free account can run the SIGNED SAC_GetData data plugin but not the"
  why "unsigned TMAITelemetry. The adapter reads SAC_GetData's binary stream"
  why "on :9000 and re-serves it in our telemetry schema on :$PLUGPORT."

  getent passwd "$U" >/dev/null || { bad "no user $U - skipping school"; return 0; }
  local UHOME; UHOME="$(getent passwd "$U" | cut -d: -f6)"

  if [ ! -d "$UHOME/.steam" ] && [ ! -d "$UHOME/.local/share/Steam" ]; then
    bad "$U has no Steam install yet. One-time setup, by hand:"
    info "  tools/steam-instance 1 --vnc     # opens Steam as $U on $DISP"
    info "  then: create the Steam account, link it to a Ubisoft account on the"
    info "        web, install Trackmania (free), set the in-game plate to TAS,"
    info "        and enable the SAC_GetData plugin in Openplanet (School mode)."
    return 0
  fi

  if port_up 9000 && [ "$dev_is_up" = 1 ]; then
    bad "port 9000 is already taken by the dev game's SAC_GetData."
    info "two games on ONE host collide on 9000. Run the school boy on another"
    info "machine, or bring the stack up with --no-dev. Skipping school."
    return 0
  fi

  # its own X + VNC so it can't steal the dev game's focus
  if ! dpy_up "$DISP"; then
    hd "Xvfb $DISP  (school instance display)"
    why "a separate X server so the two games never fight over input focus."
    setsid Xvfb "$DISP" -screen 0 1600x1000x24 -nolisten tcp >/dev/null 2>&1 &
    track xvfb-school $!
    wait_for 15 "Xvfb $DISP" -- dpy_up "$DISP"
  fi
  if ! pgrep -f "x11vnc.*$DISP" >/dev/null 2>&1; then
    setsid env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 \
      x11vnc -display "$DISP" -rfbport "$VNC" -forever -shared -nopw \
             -localhost -quiet >/dev/null 2>&1 &
    track vnc-school $!
  fi
  if command -v openbox >/dev/null 2>&1; then
    sudo -u "$U" env DISPLAY="$DISP" setsid openbox >/dev/null 2>&1 &
    track obox-school $!
  fi

  hd "Steam + Trackmania as $U   (sudo may prompt once)"
  why "the free account signs in off its own Steam ticket; the client is a"
  why "native binary so its login UI actually paints (unlike Wine's)."
  sudo -u "$U" env DISPLAY="$DISP" WAYLAND_DISPLAY= SDL_VIDEODRIVER=x11 \
    setsid steam -silent -nochatui -nofriendsui >"$LOGDIR/school-steam.log" 2>&1 &
  track steam-school $!
  wait_for 200 "school Steam ready" -- test -S "$UHOME/.steam/steam.pipe" || {
    bad "school Steam never came up - see logs/school-steam.log"; return 0; }
  sudo -u "$U" env DISPLAY="$DISP" WAYLAND_DISPLAY= \
    setsid steam "steam://rungameid/2225070" >>"$LOGDIR/school-steam.log" 2>&1 &
  wait_for 300 "school game - SAC_GetData :9000" -- port_up 9000 || {
    bad "SAC_GetData never opened :9000 - enable the plugin in $U's Openplanet"
    info "  (Openplanet > Plugins > SAC_GetData), then re-run with --school."
    return 0; }

  hd "SAC_GetData adapter   :9000  ->  :$PLUGPORT"
  why "presents the free account's telemetry on a plugin-shaped port so the"
  why "trainer treats it like any other instance."
  "$ROOT/.venv/bin/python" telemetry/sac_getdata_adapter.py --serve-port "$PLUGPORT" \
    >"$LOGDIR/sac-adapter.log" 2>&1 &
  track adapter $!
  wait_for 20 "adapter :$PLUGPORT" -- port_up "$PLUGPORT"
}

# ============================================================================ down
do_down() {
  local GAME_ONLY=0
  [ "${1:-}" = "--game" ] && GAME_ONLY=1

  hd "Stopping the game(s)"
  tools/headless-main.sh down 2>&1 | sed 's/^/    | /' || true
  pkill -9 -f 'steam://rungameid/2225070' 2>/dev/null || true
  pkill -9 -f 'SteamLaunch AppId=2225070' 2>/dev/null || true
  pkill -9 -f '2225070'                   2>/dev/null || true
  pkill -9 -f 'common/Trackmania/Trackmania' 2>/dev/null || true

  if [ "$GAME_ONLY" = 1 ]; then
    # drop only the game/display rows from the pidfile, keep plumbing tracked
    if [ -f "$PIDFILE" ]; then
      grep -vE '^(game-|xvfb-|vnc-|obox-|steam-school|adapter)' "$PIDFILE" \
        > "$PIDFILE.tmp" 2>/dev/null || true
      mv -f "$PIDFILE.tmp" "$PIDFILE"
    fi
    pkill -f 'sac_getdata_adapter.py' 2>/dev/null && info "stopped adapter" || true
    ok "game stopped. Plumbing (pads, broker, panel) left up."
    return 0
  fi

  hd "Stopping tracked processes   (SIGTERM first - the trainer saves on TERM)"
  if [ -f "$PIDFILE" ]; then
    # reverse order: last started, first stopped
    tac "$PIDFILE" | while IFS=: read -r role pid; do
      [ -n "${pid:-}" ] || continue
      [ "$pid" = 0 ] && continue
      kill -0 "$pid" 2>/dev/null || continue
      kill -TERM "$pid" 2>/dev/null && info "TERM  $role ($pid)"
    done
    sleep 3
    while IFS=: read -r role pid; do
      [ -n "${pid:-}" ] || continue
      [ "$pid" = 0 ] && continue
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null && info "KILL  $role ($pid)"
      fi
    done < "$PIDFILE"
    rm -f "$PIDFILE"
  else
    info "no $PIDFILE - falling back to pattern cleanup only"
  fi

  hd "Backstop cleanup   (anything that outlived its tracked pid)"
  local pat
  for pat in 'virtual_pad_server.py' 'telemetry/broker.py' 'tools/fleet.py' \
             'web/server.py' 'sac_getdata_adapter.py'; do
    pkill -f "$pat" 2>/dev/null && info "pkill $pat" || true
  done
  pkill -f 'x11vnc.*:100' 2>/dev/null || true
  pkill -f 'Xvfb :100'    2>/dev/null || true
  pkill -u tmai01 -f 'ubuntu12_32/steam' 2>/dev/null && info "stopped tmai01 Steam" || true
  ok "stack down."
}

# ========================================================================== status
do_status() {
  hd "TM2020-AI stack status"
  printf '    %-9s ' "panel";  panel_up && echo "http://127.0.0.1:8080  UP" || echo "down"
  local i p
  for ((i = 0; i < 4; i++)); do
    p=$((8765 + 10 * i))
    printf '    %-9s ' "pad $i"; port_up "$p" && echo ":$p up" || echo ":$p down"
  done
  printf '    %-9s ' "broker";  port_up 8767 && { broker_pong && echo ":8767 up (game connected)" || echo ":8767 up (no game)"; } || echo ":8767 down"
  printf '    %-9s ' "plugin";  port_up 8766 && echo ":8766 up (TMAITelemetry)" || echo ":8766 down"
  printf '    %-9s ' "sac";     port_up 9000 && echo ":9000 up (SAC_GetData)" || echo ":9000 down"
  printf '    %-9s ' "adapter"; port_up 8776 && echo ":8776 up" || echo ":8776 down"
  printf '    %-9s ' "game";    pgrep -f 'Trackmania\.exe' >/dev/null 2>&1 && echo "running" || echo "not running"
  printf '    %-9s ' "trainer"; pgrep -f 'train/train_sac.py' >/dev/null 2>&1 && echo "running" || echo "not running"
  printf '    %-9s ' "disp 99"; dpy_up :99  && echo "up" || echo "down"
  printf '    %-9s ' "disp 100"; dpy_up :100 && echo "up" || echo "down"
  if [ -f "$PIDFILE" ]; then
    echo; info "tracked ($PIDFILE):"
    sed 's/^/      /' "$PIDFILE"
  fi
}

# =============================================================================== main
case "${1:-}" in
  up)     shift; do_up "$@" ;;
  down)   shift; do_down "$@" ;;
  status) shift; do_status ;;
  ""|-h|--help) sed -n '2,31p' "$0" | sed 's/^# \?//;s/^#$//' ;;
  *) bad "unknown command: $1"; sed -n '2,31p' "$0" | sed 's/^# \?//;s/^#$//'; exit 2 ;;
esac

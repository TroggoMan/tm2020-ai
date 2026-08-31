#!/usr/bin/env bash
# Start / stop a GPU-accelerated headless X screen on :99 (RTX 4080), so
# TM2020 runs off the desktop at full frame rate instead of ~50fps on Xvfb.
# See deploy/xorg-headless-nvidia.conf for the why.
#
#   tools/xorg-headless.sh up      start Xorg :99 on the GPU (asks for sudo)
#   tools/xorg-headless.sh down    stop it
#   tools/xorg-headless.sh status
#
# After 'up', headless-main.sh sees :99 is already alive (its xdpyinfo probe)
# and skips starting Xvfb - the game then renders on the GPU screen. openbox
# + x11vnc still get started by headless-main for focus + viewing.
set -euo pipefail

DISP="${TMAI_XORG_DISP:-:99}"
CONF="$(cd "$(dirname "$0")/.." && pwd)/deploy/xorg-headless-nvidia.conf"
LOG="$HOME/tm2020-ai/logs/xorg-headless.log"

alive() { DISPLAY="$DISP" timeout 3 xdpyinfo >/dev/null 2>&1; }

case "${1:-}" in
up)
  [ -f "$CONF" ] || { echo "no config at $CONF" >&2; exit 1; }
  if alive; then echo ":99 already up"; exit 0; fi
  mkdir -p "$(dirname "$LOG")"
  echo "starting Xorg $DISP on the GPU (sudo) -> $LOG"
  # -novtswitch / -sharevts so it does not fight the Wayland session for the
  # VT; it renders to the GPU without owning a console.
  sudo setsid Xorg "$DISP" -config "$CONF" -novtswitch -sharevts \
       -nolisten tcp -noreset >"$LOG" 2>&1 &
  for _ in $(seq 1 20); do alive && break; sleep 1; done
  if alive; then
    echo "Xorg $DISP up. glxinfo renderer:"
    DISPLAY="$DISP" glxinfo 2>/dev/null | grep -i 'renderer string' || true
  else
    echo "Xorg $DISP did not come up - check $LOG" >&2
    tail -20 "$LOG" >&2 || true
    exit 1
  fi
  ;;
down)
  sudo pkill -f "Xorg $DISP" 2>/dev/null && echo "stopped Xorg $DISP" || \
    echo "no Xorg $DISP running"
  ;;
status)
  if alive; then
    echo ":99 is up"
    DISPLAY="$DISP" glxinfo 2>/dev/null | grep -iE 'renderer string|OpenGL vendor' || true
  else
    echo ":99 is down"
  fi
  ;;
*)
  echo "usage: $0 {up|down|status}" >&2; exit 1 ;;
esac

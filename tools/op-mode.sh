#!/usr/bin/env bash
# Set Openplanet's signature mode by editing Settings.ini, so you never have to
# fight the laggy in-overlay menu over VNC again.
#
#   tools/op-mode.sh dev        Developer mode - unsigned plugins load
#                               (TMAITelemetry needs this; needs a paid account)
#   tools/op-mode.sh signed     stock signed-only mode (what a free account gets
#                               as "School mode" is just this + school-signed)
#   tools/op-mode.sh status
#
# Options:
#   --prefix <compatdata dir>   default: the main Steam prefix
#   --instance N                fleet instance N's prefix instead
#
# THE GAME MUST BE CLOSED for that prefix. Openplanet rewrites Settings.ini
# from memory when the game exits ("Saving settings" in Openplanet.log), so a
# change made while it runs is clobbered on exit. headless-main.sh calls this
# before it launches, which is the intended path.
#
# What it flips in [App] / [Scripting]:
#   dev    -> DeveloperMode=true   ShowUnsignedPluginWarning=false
#   signed -> DeveloperMode=false  ShowUnsignedPluginWarning=true
# ShowUnsignedPluginWarning=false matters: left true, every unsigned plugin
# load pops a modal you'd have to dismiss over VNC - the whole thing this
# script exists to avoid.
set -euo pipefail

MODE=""
FORCE=0
PREFIX="/mnt/4TB/SteamLibrary/steamapps/compatdata/2225070"
while [ $# -gt 0 ]; do
  case "$1" in
    dev|signed|school|status) MODE="$1"; shift ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --instance) PREFIX="/mnt/4TB/tm2020-ai-prefixes/instance-$(printf '%02d' "$2")"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ "$MODE" = school ] && MODE=signed
[ -n "$MODE" ] || { echo "usage: $0 {dev|signed|status} [--prefix DIR | --instance N]" >&2; exit 2; }

INI="$PREFIX/pfx/drive_c/users/steamuser/OpenplanetNext/Settings.ini"
[ -f "$INI" ] || { echo "no Settings.ini at $INI" >&2
                   echo "(launch the game once so Openplanet creates it)" >&2; exit 1; }

get() { grep -E "^$1=" "$INI" | head -1 | cut -d= -f2- ; }

if [ "$MODE" = status ]; then
  dm=$(get DeveloperMode); uw=$(get ShowUnsignedPluginWarning)
  echo "$INI"
  echo "  DeveloperMode=${dm:-<unset>}   ShowUnsignedPluginWarning=${uw:-<unset>}"
  case "$dm" in
    true)  echo "  => DEVELOPER mode (unsigned plugins load)";;
    *)     echo "  => signed-only mode (TMAITelemetry will NOT load)";;
  esac
  exit 0
fi

# Refuse while a game is running - Openplanet rewrites Settings.ini from
# memory on exit and would clobber this. The wine process shows the exe with
# BACKSLASHES (Z:\...\Trackmania.exe), so match case-insensitively and don't
# assume slashes. Prefix-specific detection is unreliable across Proton
# versions; if you really are editing a different prefix than the running
# game, pass --force.
if [ "$FORCE" != 1 ] && {
     pgrep -if 'trackmania\.exe' >/dev/null 2>&1 \
     || pgrep -f 'proton .*[Tt]rackmania' >/dev/null 2>&1 \
     || ss -tlnH 2>/dev/null | grep -qE '127.0.0.1:(8766|9000) '
   }; then
  echo "a Trackmania game is running - close it first (or --force if you are" >&2
  echo "sure it is a different prefix). Openplanet overwrites Settings.ini on" >&2
  echo "exit and would undo this." >&2
  exit 1
fi

case "$MODE" in
  dev)    DM=true;  UW=false ;;
  signed) DM=false; UW=true  ;;
esac

set_key() {   # section key value
  local sec="$1" key="$2" val="$3"
  if grep -qE "^\[$sec\]" "$INI"; then
    if grep -qE "^$key=" "$INI"; then
      sed -i -E "s|^$key=.*|$key=$val|" "$INI"
    else
      sed -i -E "s|^\[$sec\]|[$sec]\n$key=$val|" "$INI"
    fi
  else
    printf '\n[%s]\n%s=%s\n' "$sec" "$key" "$val" >> "$INI"
  fi
}

cp -f "$INI" "$INI.bak"
set_key App DeveloperMode "$DM"
set_key Scripting ShowUnsignedPluginWarning "$UW"

echo "$INI"
echo "  DeveloperMode=$DM  ShowUnsignedPluginWarning=$UW   (backup: $INI.bak)"
[ "$MODE" = dev ] && echo "  next launch boots in Developer mode - no overlay clicking"

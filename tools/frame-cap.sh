#!/usr/bin/env bash
# Cap each game install's frame rate, via a dxvk.conf next to Trackmania.exe.
#
#   tools/frame-cap.sh              apply the default (60) everywhere
#   tools/frame-cap.sh 30           apply 30 everywhere
#   tools/frame-cap.sh 60 1 2       only tmai01 and tmai02
#   tools/frame-cap.sh off          remove the cap
#
# WHY a dxvk.conf and not DXVK_FRAME_RATE: the default launch path goes through
# the Steam CLIENT (`steam://rungameid/...`), so environment set in our shell
# never reaches the game. DXVK reads dxvk.conf from the game's own directory,
# which every launch path honours, and each instance has its own game dir.
#
# READ THIS BEFORE LOWERING IT. Measured on the dev instance 2026-09-07: the
# game re-reads the gamepad, and republishes the vehicle struct we read, ONCE
# PER RENDERED FRAME. Capping the frame rate caps the control rate and the
# observation rate 1:1.
#
#   uncapped   in_steer re-read 37/s   struct updated 41/s   clock step 24 ms
#   cap 20     in_steer re-read 20/s   struct updated 19/s   clock step 50 ms
#
# env/tm_env.py runs at control_hz = 40, and the game only manages ~41 fps in
# this headless setup anyway, so 60 is a ceiling that costs nothing today and
# stops a lighter map from burning GPU on frames nobody looks at. Anything
# BELOW ~45 throws away policy decisions.
#
# It is also not much of a saving: 20 fps took the game from 133% CPU to 110%
# and from 30% GPU to 22%. The per-instance ceiling is VRAM (~2.7 GB), and a
# frame cap does not move that at all - resolution and quality settings do.
set -uo pipefail

FPS="${1:-60}"; shift || true
MAIN="/mnt/4TB/SteamLibrary/steamapps/common/Trackmania"

write_cap() {  # $1 = game dir, $2 = "sudo user" or ""
  local dir="$1" owner="${2:-}"
  # A fleet user's home is not readable by us, so `test -d` has to run as root
  # for those. Without this every instance reported "game not installed yet".
  if [ -n "$owner" ]; then sudo test -d "$dir" || return 1
  else [ -d "$dir" ] || return 1; fi
  if [ "$FPS" = off ]; then
    if [ -n "$owner" ]; then sudo rm -f "$dir/dxvk.conf"; else rm -f "$dir/dxvk.conf"; fi
    echo "  $dir: cap removed"; return 0
  fi
  if [ -n "$owner" ]; then
    printf 'dxvk.maxFrameRate = %s\n' "$FPS" | sudo tee "$dir/dxvk.conf" >/dev/null
    sudo chown "$owner:$owner" "$dir/dxvk.conf"
  else
    printf 'dxvk.maxFrameRate = %s\n' "$FPS" > "$dir/dxvk.conf"
  fi
  echo "  $dir: $FPS fps"
}

if [ "$FPS" != off ] && [ "$FPS" -lt 45 ] 2>/dev/null; then
  echo "!! $FPS is below env/tm_env.py's control_hz of 40 with no margin."
  echo "!! Control and observation are frame-locked - see the header. Continuing."
fi

instances=("$@")
if [ "${#instances[@]}" -eq 0 ]; then
  echo "main install:"; write_cap "$MAIN" || echo "  $MAIN: not found"
  instances=()
  for u in /mnt/games/tm2020-ai-users/tmai*; do
    [ -d "$u" ] && instances+=("$(basename "$u" | sed 's/^tmai0*//')")
  done
fi

echo "fleet instances:"
for n in "${instances[@]}"; do
  U="tmai$(printf '%02d' "$n")"
  H="$(getent passwd "$U" | cut -d: -f6)" || { echo "  $U: no such user"; continue; }
  D="$H/.local/share/Steam/steamapps/common/Trackmania"
  write_cap "$D" "$U" || echo "  $U: game not installed yet"
done

#!/usr/bin/env bash
# Copy the main account's local maps into every fleet instance's Wine prefix.
#
#   tools/sync-maps.sh              all instances that have a prefix
#   tools/sync-maps.sh 1 2          just tmai01 and tmai02
#
# WHY: a free (Starter) account cannot open PLAY > LOCAL > PLAY A TRACK - that
# needs Club Access. It CAN drive a local map in SPLITSCREEN, and splitscreen
# reads the same "My Maps" folder, so every instance needs its own copy of the
# maps: a Wine prefix is per-user and sees nothing of the main one.
#
# An instance only grows its prefix on the FIRST game launch. Run this after
# that, not before - it says so rather than creating a half-prefix by hand.
set -uo pipefail

SRC="/mnt/4TB/SteamLibrary/steamapps/compatdata/2225070/pfx/drive_c/users/steamuser/Documents/Trackmania/Maps"
[ -d "$SRC" ] || { echo "no source maps at $SRC" >&2; exit 1; }

instances=("$@")
if [ "${#instances[@]}" -eq 0 ]; then
  instances=()
  for u in /mnt/games/tm2020-ai-users/tmai*; do
    [ -d "$u" ] && instances+=("$(basename "$u" | sed 's/^tmai0*//')")
  done
fi

for n in "${instances[@]}"; do
  U="tmai$(printf '%02d' "$n")"
  H="$(getent passwd "$U" | cut -d: -f6)" || { echo "$U: no such user"; continue; }
  DST="$H/.local/share/Steam/steamapps/compatdata/2225070/pfx/drive_c/users/steamuser/Documents/Trackmania/Maps"
  if ! sudo test -d "$(dirname "$DST")"; then
    echo "$U: no Trackmania prefix yet - launch the game once as $U first"
    continue
  fi
  sudo mkdir -p "$DST"
  sudo rsync -a --delete "$SRC"/ "$DST"/
  sudo chown -R "$U:$U" "$DST"
  echo "$U: $(sudo find "$DST" -name '*.Map.Gbx' | wc -l) maps"
done

#!/usr/bin/env bash
# Packages the plugin as a .op (a plain zip) and installs it into the TM2020
# Wine prefix's Openplanet folder. The repo copy is the source of truth.
#
# Why .op and not a plain folder: Openplanet did not discover the unpacked
# folder at all (no load line, no error, in Openplanet.log). Every working
# third-party plugin in this prefix is a .op, so that is the proven path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/plugin/TMAITelemetry"
PLUGINS="/mnt/4TB/SteamLibrary/steamapps/compatdata/2225070/pfx/drive_c/users/steamuser/OpenplanetNext/Plugins"

[ -d "$PLUGINS" ] || { echo "Openplanet Plugins dir not found: $PLUGINS" >&2; exit 1; }

# A leftover unpacked folder would collide with the .op on plugin id.
if [ -d "$PLUGINS/TMAITelemetry" ]; then
    echo "Removing stale unpacked folder $PLUGINS/TMAITelemetry"
    rm -rf "$PLUGINS/TMAITelemetry"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$SRC/info.toml" "$SRC/main.as" "$TMP/"
# zip(1) isn't installed on this box; Python's zipfile does the same job.
python3 -c '
import sys, zipfile, os
tmp = sys.argv[1]
with zipfile.ZipFile(os.path.join(tmp, "TMAITelemetry.op"), "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("info.toml", "main.as"):
        z.write(os.path.join(tmp, name), name)
' "$TMP"
mv "$TMP/TMAITelemetry.op" "$PLUGINS/TMAITelemetry.op"

echo "Installed $PLUGINS/TMAITelemetry.op"
ls -l "$PLUGINS/TMAITelemetry.op"
echo
echo "In game: F3 -> Openplanet -> Plugins -> Install/reload, or relaunch."
echo "Openplanet must be in DEVELOPER mode: this plugin is unsigned,"
echo "and School mode loads only signed plugins. (Developer mode needs"
echo "a paid account; free accounts cannot run it at all.)"

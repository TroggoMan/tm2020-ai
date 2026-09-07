#!/usr/bin/env bash
# What instances exist, where to look at them, and what is actually running.
#
#   tools/instances.sh            table of every instance
#   tools/instances.sh --vnc      just the VNC addresses, one per line
#
# VNC binds to LOCALHOST only (x11vnc -localhost, no password), so the LAN
# address below is reachable only over an SSH tunnel:
#
#   ssh -L 5901:127.0.0.1:5901 <this-box>     then connect to 127.0.0.1:5901
#
# That is deliberate: an open, passwordless VNC on the LAN is a remote desktop
# for anyone on the network.
set -uo pipefail

LAN="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+' | head -1)"
[ -n "$LAN" ] || LAN="?"

up()   { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3>&-; }
dpy()  { [ -e "/tmp/.X11-unix/X${1#:}" ]; }

rows() {
  # the dev instance is not one of the tmai users - it is you, on :99
  echo "0|troggoman (dev, paid)|:99|5999"
  for u in /mnt/games/tm2020-ai-users/tmai*; do
    [ -d "$u" ] || continue
    b="$(basename "$u")"; n="$((10#$(echo "$b" | sed 's/^tmai//')))"
    echo "$n|$b|:$((99 + n))|$((5900 + n))"
  done
}

if [ "${1:-}" = --vnc ]; then
  rows | while IFS='|' read -r n who disp vnc; do
    dpy "$disp" && echo "$who  vncviewer 127.0.0.1:$vnc"
  done
  exit 0
fi

printf '%-4s %-22s %-8s %-22s %-4s %-6s %s\n' \
       inst user display "vnc (tunnel to this)" X steam game
rows | while IFS='|' read -r n who disp vnc; do
  case "$n" in
    0) sp="$(pgrep -u troggoman -f 'ubuntu12_32/steam\b' >/dev/null && echo up || echo -)"
       gp="$(pgrep -u troggoman -f 'Trackmania.exe' >/dev/null && echo up || echo -)" ;;
    *) sp="$(pgrep -u "$who" -f 'ubuntu12_32/steam\b' >/dev/null && echo up || echo -)"
       gp="$(pgrep -u "$who" -f 'Trackmania.exe' >/dev/null && echo up || echo -)" ;;
  esac
  printf '%-4s %-22s %-8s %-22s %-4s %-6s %s\n' \
         "$n" "$who" "$disp" "127.0.0.1:$vnc" \
         "$(dpy "$disp" && echo up || echo -)" "$sp" "$gp"
done

echo
echo "this box: $LAN   (VNC is localhost-only - tunnel it:"
echo "  ssh -L 5901:127.0.0.1:5901 $LAN  )"
echo
echo "pad ports per game (4 splitscreen seats each):"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 0
.venv/bin/python - <<'PY' 2>/dev/null || echo "  (venv not available)"
import sys; sys.path.insert(0, ".")
from env.ports import seat_ports
for g in range(3):
    pads = [seat_ports(s, game=g, raw=True)["pad"] for s in range(4)]
    print(f"  game {g}: pads {pads}  broker {seat_ports(0, game=g, raw=True)['broker']}")
PY

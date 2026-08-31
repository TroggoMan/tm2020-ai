#!/usr/bin/env python3
"""Get a map onto every instance, and load it, without touching a menu.

Two questions this answers.

**"Can a free account play custom maps?"** Yes. A Starter Access account can
play any local map: drop the `.Map.Gbx` into that prefix's

    Documents/Trackmania/Maps/Downloaded/

and it is playable. Nothing about that path is account-gated - the entitlement
check is on *online* play, not on reading a file off disk.

**"Do I have to click into the map on each instance?"** No. The telemetry
plugin already exposes `playmap`, which calls the game's own
`ManiaTitleControlScriptAPI.PlayMap` - the same call the map-rotation plugins
use. So Trackmania Exchange is not off the table at all; it just goes through
the file system instead of through the in-game TMX browser:

    fetch the .Map.Gbx over HTTP  ->  copy into each prefix  ->  playmap

    tools/maps.py --list                        what each instance has
    tools/maps.py push "SD Trainer.Map.Gbx"     copy to every instance
    tools/maps.py push foo.Map.Gbx -n 3         to instances 0..2
    tools/maps.py tmx 12345                     download TMX map 12345, push it
    tools/maps.py play "Downloaded/SD Trainer.Map.Gbx" -n 3   load it everywhere

Prefixes: instance 0 is the existing Steam prefix unless $TMAI_PREFIX_0 says
otherwise; instance i>0 is `prefixes/instance-NN` beside this repo, which is
where the per-account prefix snapshots live (see ACCOUNTS.md).
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.ports import instance_ports          # noqa: E402
from env.prefixes import docs_dir, exists, prefix_for   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMX_DOWNLOAD = "https://trackmania.exchange/maps/download/{}"


def maps_dir(i: int, sub: str = "Downloaded") -> str:
    return os.path.join(docs_dir(i), "Maps", sub)


def instances_present(limit: int) -> list[int]:
    return [i for i in range(limit) if exists(i)]


def cmd_list(a) -> int:
    for i in range(a.instances):
        base = os.path.join(docs_dir(i), "Maps")
        if not os.path.isdir(base):
            print(f"instance {i}: no prefix at {prefix_for(i)}")
            continue
        print(f"instance {i}: {base}")
        for sub in ("Downloaded", "My Maps"):
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            names = sorted(f for f in os.listdir(d) if f.lower().endswith(".map.gbx"))
            print(f"    {sub}/  {len(names)} map(s)")
            for n in names:
                print(f"        {n}")
    return 0


def cmd_push(a) -> int:
    src = a.path
    if not os.path.isfile(src):
        # Convenience: accept a bare name and find it in instance 0.
        for sub in ("Downloaded", "My Maps"):
            cand = os.path.join(maps_dir(0, sub), os.path.basename(src))
            if os.path.isfile(cand):
                src = cand
                break
    if not os.path.isfile(src):
        print(f"no such map file: {a.path}", file=sys.stderr)
        return 1
    name = os.path.basename(src)
    targets = instances_present(a.instances)
    if not targets:
        print("no instance prefixes found - nothing to push to.", file=sys.stderr)
        return 1
    for i in targets:
        d = maps_dir(i, "Downloaded")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, name)
        if os.path.abspath(dst) == os.path.abspath(src):
            print(f"instance {i}: already there ({dst})")
            continue
        shutil.copy2(src, dst)
        print(f"instance {i}: {dst}")
    skipped = set(range(a.instances)) - set(targets)
    if skipped:
        print(f"skipped (no prefix yet): {sorted(skipped)}")
    return 0


def cmd_tmx(a) -> int:
    """Download one map from Trackmania Exchange, then push it.

    Deliberately a plain file download, not the in-game TMX browser: the
    browser needs the map picked by hand on every instance, and the file does
    not.
    """
    import urllib.request
    url = TMX_DOWNLOAD.format(a.id)
    out = os.path.join(ROOT, "maps", "tmx")
    os.makedirs(out, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tm2020-ai/1.0"})
    print(f"GET {url}")
    with urllib.request.urlopen(req, timeout=60) as r:
        disp = r.headers.get("Content-Disposition", "")
        name = a.name or ""
        if not name and "filename=" in disp:
            name = disp.split("filename=", 1)[1].strip('";\' ')
        name = name or f"tmx-{a.id}.Map.Gbx"
        if not name.lower().endswith(".map.gbx"):
            name += ".Map.Gbx"
        blob = r.read()
    path = os.path.join(out, name)
    with open(path, "wb") as f:
        f.write(blob)
    print(f"saved {path} ({len(blob) / 1024:.0f} KB)")
    a.path = path
    return cmd_push(a)


def cmd_play(a) -> int:
    """Tell each running instance to load a map. No menu navigation."""
    rc = 0
    for i in range(a.instances):
        port = instance_ports(i)["broker"]
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
                s.sendall(f"playmap {a.path}\n".encode())
                s.settimeout(5)
                # The reply is one line among the telemetry stream; read a
                # little and look for it rather than assuming it comes first.
                buf = b""
                while b"playmap" not in buf and len(buf) < 200000:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
            print(f"instance {i}: playmap sent"
                  + ("" if b"playmap" in buf else "  (no ack seen)"))
        except OSError as ex:
            print(f"instance {i}: broker on {port} not reachable ({ex})")
            rc = 1
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--instances", type=int, default=8)
    # Repeated on every subcommand as well, so `maps.py push foo -n 3` works
    # as readily as `maps.py -n 3 push foo`. argparse otherwise silently only
    # accepts it before the verb, which reads as a broken flag.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-n", "--instances", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", parents=[common]).set_defaults(fn=cmd_list)

    p = sub.add_parser("push", parents=[common],
                       help="copy a .Map.Gbx into every prefix")
    p.add_argument("path")
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser("tmx", parents=[common], help="download a Trackmania Exchange map, then push")
    p.add_argument("id")
    p.add_argument("--name", help="override the saved filename")
    p.set_defaults(fn=cmd_tmx)

    p = sub.add_parser("play", parents=[common], help="load a map on every running instance")
    p.add_argument("path", help="path as the game sees it, e.g. "
                               "'Downloaded/SD Trainer.Map.Gbx'")
    p.set_defaults(fn=cmd_play)

    top = ap.parse_known_args()[0]
    a = ap.parse_args()
    if a.instances is None:
        a.instances = top.instances if top.instances is not None else 8
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

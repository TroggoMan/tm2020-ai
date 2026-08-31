#!/usr/bin/env python3
"""Let a trained model drive. No learning, no buffer, nothing written to it.

This is the payoff: take what the policy has learned and point it at a track.

    tools/drive.py                              drive the loaded map
    tools/drive.py --map "Downloaded/foo.Map.Gbx"
    tools/drive.py --name sac_split4 --laps 10
    tools/drive.py --seats 4                    all four cars at once

**Any map works, with no recorded lap.** The reference line is built from the
map's own spawn / checkpoint / finish landmarks, exactly as the explore stage
does, so a track the model has never seen needs no preparation beyond being
loaded. Whether it can *drive* the track is the open question - that is what
you are testing - but nothing about the setup stops it trying.

Deterministic by default. SAC's policy is stochastic during training because
exploration is the point; when you are measuring what it has learned, sampling
adds noise you did not ask for. `--stochastic` restores the training behaviour
if you want to see the spread.

The model is opened read-only and never saved, so this cannot damage a policy
you care about - drive with it as much as you like.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_map(path: str, broker_port: int) -> None:
    with socket.create_connection(("127.0.0.1", broker_port), 5) as s:
        s.sendall(f"playmap {path}\n".encode())
        time.sleep(0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sac_split4",
                    help="model in models/ to drive with")
    ap.add_argument("--map", help="load this map first, e.g. "
                                  "'Downloaded/foo.Map.Gbx'")
    ap.add_argument("--laps", type=int, default=5,
                    help="episodes to drive before stopping")
    ap.add_argument("--seats", type=int, default=1,
                    help="drive N splitscreen seats at once")
    ap.add_argument("--control-hz", type=float, default=40.0)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample from the policy instead of taking its mean")
    a = ap.parse_args()

    import numpy as np
    from env.mapdata import Gates, provisional_line
    from env.ports import broker_addr, instance_ports, seat_ports
    from env.tm_env import TelemetryLink, TrackmaniaEnv
    from train.bootstrap import BootstrapSAC

    path = os.path.join(ROOT, "models", a.name)
    if not os.path.isfile(path + ".zip"):
        print(f"no model at {path}.zip", file=sys.stderr)
        return 1

    if a.map:
        load_map(a.map, broker_addr(0)[1])
        print(f"loading {a.map} ...", flush=True)
        time.sleep(8.0)

    # The line comes from the map's own landmarks, so an unseen track needs
    # nothing prepared in advance.
    link = TelemetryLink()
    reply = None
    for _ in range(10):
        reply = link.command("landmarks", wait=3.0)
        if reply and reply.get("ok"):
            break
        time.sleep(1.0)
    link.close()
    if not reply or not reply.get("ok"):
        print("no landmarks - is a map loaded and the plugin running?",
              file=sys.stderr)
        return 1
    gates = Gates(reply.get("items", []))
    line = provisional_line(gates)
    print(f"track: {gates.describe()}, line {line.length:.0f}m", flush=True)

    n = max(1, a.seats)
    envs = []
    for i in range(n):
        ports = seat_ports(i) if n > 1 else instance_ports(i)
        envs.append(TrackmaniaEnv(
            line, control_hz=a.control_hz, profile="explore",
            instance=i, slot=(i if n > 1 else None), rebuild_line=True,
            max_offset=250.0, max_episode_s=120.0, capture_replays=False,
            pad_addr=("127.0.0.1", ports["pad"]),
            telem_addr=("127.0.0.1", ports["broker"])))

    model = BootstrapSAC.load(path, device="cpu")
    print(f"driving with {a.name} ({model.num_timesteps:,} steps of training), "
          f"{'sampled' if a.stochastic else 'deterministic'}\n", flush=True)

    finished, times = 0, []
    try:
        for lap in range(a.laps):
            obs = [e.reset()[0] for e in envs]
            done = [False] * n
            while not all(done):
                for i, e in enumerate(envs):
                    if done[i]:
                        continue
                    act, _ = model.predict(obs[i],
                                           deterministic=not a.stochastic)
                    obs[i], _r, term, trunc, info = e.step(act)
                    if term or trunc:
                        done[i] = True
                        ok = bool(info.get("finished"))
                        t = info.get("race_time")
                        if ok and t:
                            finished += 1
                            times.append(int(t))
                        seat = f"seat {i} " if n > 1 else ""
                        print(f"  lap {lap + 1} {seat}"
                              f"{'FINISH ' + format(t / 1000, '.3f') + 's' if ok else info.get('reason', 'ended')}",
                              flush=True)
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        for e in envs:
            e.close()

    total = a.laps * n
    print(f"\n{finished}/{total} finished", flush=True)
    if times:
        print(f"best {min(times) / 1000:.3f}s, "
              f"mean {sum(times) / len(times) / 1000:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

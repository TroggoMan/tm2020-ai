#!/usr/bin/env python3
"""What every model on disk is, and which one you are actually using.

    python3 tools/models.py

The short answer to "am I supposed to switch models?" is **no** - the stage
picks the model, and it has since the names were made stage-aware:

    --stage race     -> models/sac_tm_v2.zip     (default)
    --stage explore  -> models/sac_explore.zip

You only pass --name to run a deliberate experiment alongside the main one, or
to go back to an old model. `_best` files are written automatically when a
finished lap beats the previous best; they are never resumed from unless you
ask for them by name, because the newest model has more experience even when
it is momentarily slower.

This prints the inventory with the three things that decide whether a model is
usable: its observation width (a mismatch will not load at all), what it was
trained against (a match that loads but means something different), and how
much experience is in it.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

ROLES = {
    "sac_tm": "v1 ancestor - 37-dim observation. Cannot load in the current "
              "environment; sac_tm_v2 was transplanted from it.",
    "sac_tm_v2": "THE RACING MODEL. What --stage race resumes by default.",
    "sac_explore": "THE EXPLORING MODEL. What --stage explore resumes by "
                   "default - no recorded lap needed.",
}


def describe(name: str) -> dict:
    from stable_baselines3 import SAC
    path = os.path.join(MODELS, name + ".zip")
    out = {"name": name, "size_mb": os.path.getsize(path) / 1e6}
    try:
        m = SAC.load(path, device="cpu")
        out["obs"] = int(m.observation_space.shape[0])
        out["steps"] = int(m.num_timesteps)
    except Exception as ex:
        out["err"] = str(ex)[:60]
    try:
        with open(path[:-4] + ".meta.json") as f:
            out["meta"] = json.load(f)
    except (OSError, json.JSONDecodeError):
        out["meta"] = None
    buf = os.path.join(MODELS, name + "_buffer.pkl")
    out["buffer_mb"] = os.path.getsize(buf) / 1e6 if os.path.exists(buf) else 0
    return out


def main() -> int:
    from env.tm_env import OBS_DIM
    names = sorted(f[:-4] for f in os.listdir(MODELS) if f.endswith(".zip"))
    if not names:
        print("no models yet.")
        return 0

    print(f"the environment's observation is {OBS_DIM} dims. A model with a "
          f"different width cannot be loaded at all.\n")
    print(f"{'model':24s} {'obs':>5} {'steps':>9} {'buffer':>9}  notes")
    print("-" * 92)
    for n in names:
        d = describe(n)
        if "err" in d:
            print(f"{n:24s} {'?':>5} {'?':>9} {'':>9}  UNREADABLE: {d['err']}")
            continue
        flag = "" if d["obs"] == OBS_DIM else "  <- WRONG WIDTH, will not load"
        buf = f"{d['buffer_mb']:.0f}MB" if d["buffer_mb"] else "-"
        print(f"{n:24s} {d['obs']:>5} {d['steps']:>9,} {buf:>9}{flag}")
        role = ROLES.get(n)
        if role:
            print(f"{'':24s} {role}")
        meta = d["meta"]
        if meta:
            act = meta.get("action") or {}
            print(f"{'':24s} trained {meta.get('saved', '?')}, "
                  f"{meta.get('stage', '?')} stage, "
                  f"{meta.get('control_hz', '?')}Hz, throttle "
                  f"{'on/off' if act.get('binary_gas') else 'continuous'}, "
                  f"checkpoints by {meta.get('cp_mode', '?')}")
        elif n in ROLES or not n.endswith("_best"):
            print(f"{'':24s} no metadata - predates the sidecar, so what it "
                  f"was trained against is unknown")

    print("\nnothing here needs switching by hand. --stage race uses "
          "sac_tm_v2, --stage explore uses sac_explore.")
    print("Roll back to an older snapshot from the panel (Checkpoints & "
          "snapshots) or with --name.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Widen a saved model's input layers for the edge-awareness observation.

    tools/migrate_obs.py models/sac_padfix.zip models/sac_edge.zip

The edge-awareness inputs (`border` 4 dims + `sides_ahead` 6) were APPENDED to
the observation, taking it 119 -> 129. Every older dimension keeps its index, so
the weights that read them are still valid - but the first layer's weight
MATRIX changes shape, and SB3's `set_parameters(..., exact_match=False)` handles
that by silently SKIPPING the layer. You would get a model that loads without
complaint and drives like a random init.

This copies the old columns into the wider matrix and ZERO-fills the new ones.
Zero weight on a new input means the network's output is bit-for-bit what it was
before, so the migrated model drives exactly as well as the one it came from and
can learn to use the new inputs from there.

The critic needs care: its input is [obs | action], so the new observation
columns have to be inserted at index 119 - BEFORE the 3 action columns - not
appended at the end. Getting that wrong shifts the action inputs and wrecks the
critic while looking fine.
"""
import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from env.tm_env import OBS_DIM, OBS_DIM_PRE_EDGE  # noqa: E402

ACT_DIM = 3


def widen(t: torch.Tensor, old_in: int, new_in: int, insert_at: int):
    """Return a copy of `t` with (new_in - old_in) zero columns inserted."""
    extra = new_in - old_in
    out = torch.zeros(t.shape[0], t.shape[1] + extra, dtype=t.dtype)
    out[:, :insert_at] = t[:, :insert_at]
    out[:, insert_at + extra:] = t[:, insert_at:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--old-obs", type=int, default=OBS_DIM_PRE_EDGE)
    ap.add_argument("--new-obs", type=int, default=OBS_DIM)
    a = ap.parse_args()
    if a.new_obs <= a.old_obs:
        sys.exit(f"new obs ({a.new_obs}) must be wider than old ({a.old_obs})")
    if not os.path.isfile(a.src):
        sys.exit(f"no such model: {a.src}")

    print(f"{a.src}  ->  {a.dst}")
    print(f"observation {a.old_obs} -> {a.new_obs} "
          f"(+{a.new_obs - a.old_obs} zero-initialised)\n")

    # Work on the raw archive rather than loading a full SAC: constructing the
    # model needs an env, and the whole point is that the saved policy no longer
    # matches the current one.
    with zipfile.ZipFile(a.src) as z:
        members = z.namelist()
        blobs = {n: z.read(n) for n in members}

    changed = []
    for name in list(blobs):
        if not name.endswith(".pth"):
            continue
        import io
        sd = torch.load(io.BytesIO(blobs[name]), map_location="cpu",
                        weights_only=False)
        if not isinstance(sd, dict):
            continue
        touched = False
        for k, v in list(sd.items()):
            if not (hasattr(v, "shape") and len(getattr(v, "shape", ())) == 2):
                continue
            cols = v.shape[1]
            if cols == a.old_obs:                 # actor: obs only
                sd[k] = widen(v, a.old_obs, a.new_obs, a.old_obs)
                changed.append(f"{name}:{k}  {tuple(v.shape)} -> "
                               f"{tuple(sd[k].shape)}   [actor, appended]")
                touched = True
            elif cols == a.old_obs + ACT_DIM:     # critic: [obs | action]
                sd[k] = widen(v, a.old_obs + ACT_DIM, a.new_obs + ACT_DIM,
                              a.old_obs)
                changed.append(f"{name}:{k}  {tuple(v.shape)} -> "
                               f"{tuple(sd[k].shape)}   [critic, inserted "
                               f"before the {ACT_DIM} action cols]")
                touched = True
        if touched:
            buf = io.BytesIO()
            torch.save(sd, buf)
            blobs[name] = buf.getvalue()

    # Adam's momentum buffers mirror the shape of the parameters they track,
    # and they live NESTED inside optimizer["state"][param_index], where the
    # loop above never looks. Leaving them at the old width crashes on the
    # first gradient step with
    #
    #   RuntimeError: The size of tensor a (122) must match the size of
    #                 tensor b (144) at non-singleton dimension 1
    #
    # ...which is exactly what happened the first time this ran.
    #
    # They are CLEARED rather than widened. Adam re-initialises them lazily on
    # the first step, and exp_avg / exp_avg_sq are short-horizon statistics
    # (beta 0.9 / 0.999) - a few hundred steps of readaptation. Widening them
    # by hand means inventing second-moment estimates for columns that have
    # never had a gradient, which is a worse guess than starting clean.
    # param_groups is preserved, so the learning rate and betas survive.
    for name in list(blobs):
        if not name.endswith(".optimizer.pth"):
            continue
        import io
        sd = torch.load(io.BytesIO(blobs[name]), map_location="cpu",
                        weights_only=False)
        if isinstance(sd, dict) and sd.get("state"):
            n_cleared = len(sd["state"])
            sd["state"] = {}
            buf = io.BytesIO()
            torch.save(sd, buf)
            blobs[name] = buf.getvalue()
            changed.append(f"{name}  cleared Adam state for {n_cleared} params "
                           f"(re-initialised lazily; param_groups kept)")

    if not changed:
        sys.exit(f"nothing to widen - no 2-D weight had {a.old_obs} or "
                 f"{a.old_obs + ACT_DIM} input columns. Is this model already "
                 f"migrated?")

    os.makedirs(os.path.dirname(os.path.abspath(a.dst)) or ".", exist_ok=True)
    with zipfile.ZipFile(a.dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in members:
            z.writestr(n, blobs[n])

    for line in changed:
        print("  " + line)
    print(f"\n{len(changed)} layer(s) widened. New inputs are zero-weighted, so "
          f"this model\nbehaves identically to {os.path.basename(a.src)} until "
          f"it learns to use them.")
    print(f"\nresume with:  --init-from {a.dst} --bootstrap off")
    return 0


if __name__ == "__main__":
    sys.exit(main())

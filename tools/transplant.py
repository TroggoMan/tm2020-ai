#!/usr/bin/env python3
"""Carry a trained policy forward onto a wider observation.

Adding surfaces, effects and lidar took the observation from 37 dims to 106,
which means `sac_tm_best.zip` - the model that actually finished a lap - can no
longer be loaded. Retraining it from scratch would throw away every hour that
went into it.

It does not have to be thrown away. The new observation was built by
*appending*, never reordering, so its first 37 values are byte-identical in
meaning to the old ones:

    speed 1 | vel 3 | up 3 | ahead 18 | offset 1 | slip 4 | adher 1 |
    grnd 1 | gear 1 | rpm 1 | prev 3            = 37, unchanged
    ... then surface 24, icing 4, dirt 4, wet 1, turbo 3, reactor 5,
        cruise 1, slowmo 1, side 1, airbrake 1, fx_ahead 8, lidar 16

So every layer that touches the observation - the actor's first layer and both
critics' - gets the old weights in its first 37 input columns and **zeros** in
the rest. Zero, not random: a zeroed column contributes nothing, so at step one
the new network computes exactly what the old one did. Same driving. It then
learns to use the new inputs from there rather than starting blind.

Every other layer is deeper than the input and copies across unchanged.

    python3 tools/transplant.py models/sac_tm_best.zip models/sac_tm_v2.zip

This is a warm start, not a free lunch: the old model learned specific tracks.
What transfers is the low-level "throttle makes it go, this much steering turns
it that much"; the route knowledge does not.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th
from stable_baselines3 import SAC

from env.tm_env import OBS_DIM


def widen(dst: th.Tensor, src: th.Tensor, name: str,
          old_dim: int, new_dim: int, n_act: int) -> str:
    """Copy src into dst, widening the observation block and leaving the new
    columns at zero.

    The critic is the subtle one. Its input is the observation and the action
    CONCATENATED - 37+3 = 40 before, 106+3 = 109 now - so the action weights do
    not live at the front. Copying the first 40 columns straight across would
    drop the old action weights onto columns 37..39, which in the new layout
    are per-wheel surface inputs. The value function would then be reading tyre
    surface as though it were throttle, and nothing about the shapes would look
    wrong. The action tail is therefore moved explicitly.
    """
    if dst.shape == src.shape:
        dst.copy_(src)
        return f"    {name:52s} copied {tuple(src.shape)}"
    if dst.dim() != 2 or src.dim() != 2 or dst.shape[0] != src.shape[0]:
        return f"    {name:52s} SKIPPED ({tuple(src.shape)} -> {tuple(dst.shape)})"

    dst.zero_()
    if src.shape[1] == old_dim and dst.shape[1] == new_dim:
        dst[:, :old_dim].copy_(src)
        tail = ""
    elif src.shape[1] == old_dim + n_act and dst.shape[1] == new_dim + n_act:
        dst[:, :old_dim].copy_(src[:, :old_dim])
        dst[:, new_dim:].copy_(src[:, old_dim:])
        tail = f", {n_act} action columns moved to {new_dim}..{new_dim+n_act-1}"
    else:
        return (f"    {name:52s} SKIPPED (unrecognised layout "
                f"{tuple(src.shape)} -> {tuple(dst.shape)})")
    return (f"    {name:52s} widened {tuple(src.shape)} -> {tuple(dst.shape)}"
            f", {dst.shape[1]-src.shape[1]} zeroed{tail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="trained model with the narrower observation")
    ap.add_argument("dst", help="where to write the widened model")
    ap.add_argument("--net-arch", default="256,256")
    args = ap.parse_args()

    old = SAC.load(args.src, device="cpu")
    old_dim = old.observation_space.shape[0]
    print(f"source: {args.src}  obs={old_dim}")
    print(f"target: obs={OBS_DIM}")
    if old_dim == OBS_DIM:
        print("\nsame width - nothing to transplant, just use --resume.")
        return 1
    if old_dim > OBS_DIM:
        print("\nsource is WIDER than the current observation; this tool only "
              "widens. Nothing written.")
        return 1

    # A fresh model of the right shape, whose weights we then overwrite. Built
    # against the real spaces so every hyperparameter matches what training
    # will construct.
    import gymnasium as gym
    from gymnasium import spaces

    class Shell(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        action_space = old.action_space

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, False, True, {}

    arch = [int(x) for x in args.net_arch.split(",")]
    new = SAC("MlpPolicy", Shell(), policy_kwargs={"net_arch": arch},
              device="cpu", verbose=0)

    n_act = int(np.prod(old.action_space.shape))
    src_sd = old.policy.state_dict()
    dst_sd = new.policy.state_dict()

    missing = [k for k in dst_sd if k not in src_sd]
    if missing:
        print(f"\n{len(missing)} tensors have no counterpart and keep their "
              f"fresh initialisation:")
        for k in missing[:6]:
            print(f"    {k}")

    print("\ntransplanting:")
    widened = 0
    with th.no_grad():
        for k, dst in dst_sd.items():
            if k not in src_sd:
                continue
            line = widen(dst, src_sd[k], k, old_dim, OBS_DIM, n_act)
            if "widened" in line:
                widened += 1
                print(line)
            elif "SKIPPED" in line:
                print(line)
    new.policy.load_state_dict(dst_sd)

    # The transplant is only exact if the shared prefix really is identical.
    # Check it rather than assert it: feed an observation whose new columns are
    # zero and confirm both networks produce the same action.
    rng = np.random.RandomState(0)
    probe = np.zeros((1, OBS_DIM), dtype=np.float32)
    probe[0, :old_dim] = rng.randn(old_dim).astype(np.float32)
    act = rng.uniform(-1, 1, (1, n_act)).astype(np.float32)
    with th.no_grad():
        a_old = old.policy.actor(th.as_tensor(probe[:, :old_dim]), deterministic=True)
        a_new = new.policy.actor(th.as_tensor(probe), deterministic=True)
        # The critic has to be checked separately and explicitly: it is the one
        # whose input layout changes shape, and a wrong transplant there is
        # invisible from the actor.
        q_old = old.policy.critic(th.as_tensor(probe[:, :old_dim]), th.as_tensor(act))
        q_new = new.policy.critic(th.as_tensor(probe), th.as_tensor(act))
    d_act = float(th.max(th.abs(a_old - a_new)))
    # Q is unbounded - this model's values are around 5000 - so an absolute
    # tolerance is meaningless here. The two networks sum 109 terms and 40 in
    # different orders, and float32 accumulation alone accounts for roughly
    # sqrt(n)*eps*|Q|. Compare relatively.
    q_mag = max(float(th.abs(o).max()) for o in q_old) or 1.0
    d_q = max(float(th.max(th.abs(o - n))) for o, n in zip(q_old, q_new))
    rel_q = d_q / q_mag
    print(f"\ncheck on a shared-prefix observation:")
    print(f"    actor  max difference {d_act:.2e}")
    print(f"    critic max difference {d_q:.2e} on |Q|~{q_mag:.0f}"
          f"  -> relative {rel_q:.2e}")
    if d_act > 1e-5 or rel_q > 1e-5:
        print("  both should be ~0. The transplant is NOT faithful - "
              "nothing written.")
        return 1

    new.save(args.dst)
    print(f"\nwrote {args.dst}  ({widened} input layers widened)")
    print("The replay buffer cannot come with it - its transitions are the old "
          "width - so it starts empty and refills.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

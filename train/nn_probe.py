"""Live activation probe for the SAC policy.

Taps the actor's hidden layers with forward hooks and writes a small JSON
snapshot the web panel renders. These are the network's real activations on the
observation it is acting on right now - not a decorative animation.

Three things make the picture honest rather than pretty:

  * The hidden layers are 256 wide, far too many to draw, so a fixed-stride
    sample is taken. Fixed, not random, so a given dot is always the same
    neuron and you can watch it over time.
  * The edges are real weights. At training start the strongest few incoming
    connections per drawn neuron are extracted from the actual weight matrices,
    restricted to neurons that are themselves drawn - so an edge on screen is a
    connection that exists, with the sign it really has.
  * The critic's Q estimate for the current state and action is sampled too.
    That is the number the policy is actually maximising, and it is what lets
    the panel colour the driven path by what the AI thought each bit of track
    was worth.

Nothing here is allowed to break training: every hook and every extraction is
wrapped, and a failure disables the probe rather than raising into the learning
loop.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from env.tm_env import OBS_GROUPS

SAMPLE = 24     # neurons drawn per hidden layer
TOP_K = 3       # incoming edges drawn per neuron


class NNProbe(BaseCallback):
    def __init__(self, path: str, hz: float = 8.0, sample: int = SAMPLE):
        super().__init__()
        self.path = path
        self.period = 1.0 / hz
        self.sample = sample
        self.last = 0.0
        self.acts: dict[int, np.ndarray] = {}
        self.handles: list = []
        self.enabled = True
        self.wiring: dict | None = None
        self.picks: list[list[int]] = []

    # -- setup ------------------------------------------------------------

    def _on_training_start(self) -> None:
        try:
            layers = [m for m in self.model.policy.actor.latent_pi
                      if hasattr(m, "out_features")]
            for i, layer in enumerate(layers):
                self.handles.append(layer.register_forward_hook(self._make_hook(i)))
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.wiring = self._extract_wiring(layers)
        except Exception as ex:  # never take training down with us
            print(f"nn probe disabled: {ex}", flush=True)
            self.enabled = False

    @staticmethod
    def _stride(n: int, k: int) -> np.ndarray:
        if n <= k:
            return np.arange(n)
        return np.linspace(0, n - 1, k).astype(int)

    def _extract_wiring(self, layers) -> dict:
        """Strongest incoming connections per drawn neuron, from the real
        weights. Restricted to drawn neurons so every edge has both ends on
        screen; an edge to a node nobody can see is just noise."""
        w0 = layers[0].weight.detach().cpu().numpy()          # (h0, obs)
        self.picks = [self._stride(layers[i].out_features, self.sample).tolist()
                      for i in range(len(layers))]

        edges = {"l0": [], "l1": [], "out": []}
        for j in self.picks[0]:
            row = w0[j]
            idx = np.argsort(-np.abs(row))[:TOP_K]
            edges["l0"].append([[int(i), round(float(row[i]), 4)] for i in idx])

        if len(layers) > 1:
            w1 = layers[1].weight.detach().cpu().numpy()      # (h1, h0)
            prev = np.asarray(self.picks[0])
            for j in self.picks[1]:
                row = w1[j][prev]                             # only drawn inputs
                idx = np.argsort(-np.abs(row))[:TOP_K]
                edges["l1"].append([[int(i), round(float(row[i]), 4)] for i in idx])

        mu = self.model.policy.actor.mu
        wm = mu.weight.detach().cpu().numpy()                 # (3, h_last)
        prev = np.asarray(self.picks[-1])
        for j in range(wm.shape[0]):
            row = wm[j][prev]
            idx = np.argsort(-np.abs(row))[:TOP_K]
            edges["out"].append([[int(i), round(float(row[i]), 4)] for i in idx])

        return {"edges": edges, "picks": self.picks,
                "widths": [int(l.out_features) for l in layers]}

    def _make_hook(self, idx: int):
        def hook(_module, _inp, out):
            try:
                self.acts[idx] = out.detach().float().cpu().numpy()[0]
            except Exception:
                pass
        return hook

    # -- sampling ---------------------------------------------------------

    def _run_actor(self, obs):
        """Push the observation through the actor ourselves.

        The hooks cannot be relied on to fire during collection: for the first
        `learning_starts` steps SB3 samples actions straight from the action
        space and never touches the actor, so the whole hidden-layer view would
        sit empty for the first hour of every run. Running it here costs one
        forward pass at 8Hz and means the picture is always the policy's real
        response to the observation on screen.

        Returns the actor's deterministic action, which is what the drawn
        network actually computed - as opposed to the action that was sent,
        which during warm-up (or under exploration noise) is something else.
        """
        try:
            import torch as th
            obs_t, _ = self.model.policy.obs_to_tensor(obs)
            with th.no_grad():
                out = self.model.policy.actor(obs_t, deterministic=True)
            return [round(float(x), 4) for x in np.asarray(out.cpu()).reshape(-1)]
        except Exception:
            return None

    def _q_value(self, obs, act) -> float | None:
        """What the critic thinks this state and action are worth.

        SAC keeps twin critics and trains on the smaller of the two, so that is
        the number reported - taking the mean would flatter it.
        """
        try:
            import torch as th
            obs_t, _ = self.model.policy.obs_to_tensor(obs)
            act_t = th.as_tensor(np.asarray(act), device=self.model.device).float()
            if act_t.ndim == 1:
                act_t = act_t.unsqueeze(0)
            with th.no_grad():
                qs = self.model.critic(obs_t, act_t)
            return round(float(min(q.item() for q in qs)), 3)
        except Exception:
            return None

    def _on_step(self) -> bool:
        if not self.enabled:
            return True
        now = time.time()
        if now - self.last < self.period:
            return True
        self.last = now
        try:
            # With a fleet, `new_obs` is (n_envs, obs_dim). Flattening it
            # would splice every game's observation into one vector and label
            # the result with instance 0's group names - a picture that looks
            # fine and means nothing. Draw one game; the model is shared, so
            # the weights on screen are the weights all of them are using.
            raw_obs = np.atleast_2d(np.asarray(self.locals.get("new_obs")))[:1]
            raw_act = np.atleast_2d(np.asarray(self.locals.get("actions")))[:1]
            obs = raw_obs.reshape(-1)
            act = raw_act.reshape(-1)

            # Order matters: this populates self.acts via the hooks, so it has
            # to happen before the activations are read.
            policy_act = self._run_actor(raw_obs)

            hidden = []
            for layer_i in sorted(self.acts):
                vec = self.acts[layer_i]
                pick = (self.picks[layer_i] if layer_i < len(self.picks)
                        else self._stride(vec.size, self.sample).tolist())
                hidden.append([round(float(vec[i]), 4) for i in pick
                               if i < vec.size])

            snap = {
                "t": now,
                "step": int(self.num_timesteps),
                "obs": [round(float(x), 4) for x in obs],
                "groups": OBS_GROUPS,
                "hidden": hidden,
                # What the network computed, and what was actually sent. They
                # differ during warm-up and under exploration noise, and seeing
                # the gap is the point - it is what "still exploring" looks
                # like from the outside.
                "policy_action": policy_act,
                "action": [round(float(x), 4) for x in act],
                "labels": ["steer", "gas", "brake"],
                "warmup": bool(self.num_timesteps < getattr(
                    self.model, "learning_starts", 0)),
                # Which game is on screen, and how many are feeding the model.
                "instance": 0,
                "n_envs": int(getattr(self.training_env, "num_envs", 1)),
                "bootstrap": getattr(self.model, "bootstrap", "off"),
                "q": self._q_value(raw_obs, raw_act),
                "wiring": self.wiring,
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(snap, f)
            os.replace(tmp, self.path)   # atomic, so the panel never reads half
        except Exception:
            pass
        return True

    def _on_training_end(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass

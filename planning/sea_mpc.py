"""AdaJEPA MPC with slow and fast weights.

Fast weights: the per-replan AdaJEPA update on the current episode (inherited),
reset to the slow weights at the start of every episode.

Slow weights: shared across episodes. Two ways to obtain them:
  streaming    after every episode, take `stream_steps` Adam steps on random
               batches of all segments executed so far (one optimizer for the
               whole run); the next episode starts from the result
  calibration  load `slow_init`; if it does not exist, train it offline on the
               episodes written by a calibration run with `dump_dir`

Config (planner.sea):
  stream_steps   >0 enables streaming slow weights
  stream_lr      predictor lr of the slow update (encoder keeps the online lr ratio)
  stream_batch   segments per slow step
  seed           seed of the slow-update batch sampling
  slow_init      path of offline slow weights (calibration mode)
  calib_dir      dump_dir of the calibration run used to train slow_init
  offline        offline training: lrs, max_steps, batch, eval_every, max_episodes
  dump_dir       write every executed episode here (to calibrate a later run)
"""

import json
import os
import random

import numpy as np
import torch

from .adajepa_mpc import AdaJEPAMPCPlanner


class SEAMPCPlanner(AdaJEPAMPCPlanner):
    def __init__(self, sea=None, **kwargs):
        super().__init__(**kwargs)
        cfg = dict(sea or {})
        self.stream_steps = int(cfg.get("stream_steps", 0))
        self.stream_lr = float(cfg.get("stream_lr", 0.0))
        self.stream_batch = int(cfg.get("stream_batch", 16))
        self.seed = int(cfg.get("seed", 0))
        self.slow_init = cfg.get("slow_init")
        self.calib_dir = cfg.get("calib_dir")
        self.offline = dict(cfg.get("offline") or {})
        self.dump_dir = cfg.get("dump_dir")

        tr = self.adajepa_trainer
        self._tensors = [t for t, _ in tr._snapshot if t.is_floating_point()]
        self._segments = []  # raw (visual, proprio, actions) of all finished episodes
        self._episode = []
        self._slow_optim = None

        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
        if self.slow_init:
            if not os.path.exists(self.slow_init):
                train_offline_slow_weights(self)
            self._load_slow(self.slow_init)

    # ---------- episode hooks ----------
    def _plan_single(self, **kwargs):
        self._episode = []
        out = super()._plan_single(**kwargs)
        self._end_episode()
        return out

    def _post_env_feedback(self, taken_actions, e_obses):
        fs = self.evaluator.frameskip
        T = taken_actions.shape[1]
        start = self.iter * T * fs
        raw = {k: np.array(v[:, start : start + T * fs + 1 : fs]) for k, v in e_obses.items()}
        self._episode.append((raw["visual"], raw["proprio"], taken_actions.cpu().numpy()))
        return super()._post_env_feedback(taken_actions, e_obses)

    def _end_episode(self):
        if not self._episode:
            return
        if self.dump_dir:
            self._dump_episode()
        if self.stream_steps <= 0:
            return
        self._segments.extend(self._episode)
        self.adajepa_trainer.reset()  # drop the fast weights, back to the slow weights
        self._stream_update()

    def _dump_episode(self):
        vis, pro, act = zip(*self._episode)
        path = os.path.join(self.dump_dir, f"ep{self._sample_idx:03d}.pt")
        torch.save({"visual": np.stack(vis), "proprio": np.stack(pro), "actions": np.stack(act)}, path)

    # ---------- streaming slow weights ----------
    def _stream_update(self):
        tr = self.adajepa_trainer
        segs = self._segments
        rng = random.Random(self.seed * 7919 + len(segs))
        if self._slow_optim is None:
            self._slow_optim = make_optimizer(tr, self.stream_lr)
        losses = []
        for _ in range(self.stream_steps):
            ids = [rng.randrange(len(segs)) for _ in range(min(self.stream_batch, len(segs)))]
            obs, act = batch_segments(self.preprocessor, [segs[i] for i in ids])
            losses.append(finetune_step(tr, self._slow_optim, obs, act))
        tr._snapshot = tr._take_snapshot()  # the next episode resets to these weights
        print(f"[sea] slow update: segments {len(segs)} loss {np.mean(losses):.6f}")

    # ---------- slow weights on disk ----------
    def _load_slow(self, path):
        state = torch.load(path, weights_only=False)
        with torch.no_grad():
            for t, s in zip(self._tensors, state["tensors"]):
                t.copy_(s.to(t.device))
        tr = self.adajepa_trainer
        tr._snapshot = tr._take_snapshot()


# ---------- helpers ----------
def batch_segments(preprocessor, segs):
    """Stack raw (visual, proprio, actions) segments into one model batch."""
    vis, pro, acts = [], [], []
    for v, p, a in segs:
        o = preprocessor.transform_obs({"visual": v, "proprio": p})
        vis.append(o["visual"])
        pro.append(o["proprio"])
        acts.append(torch.as_tensor(a))
    return {"visual": torch.cat(vis), "proprio": torch.cat(pro)}, torch.cat(acts)


def make_optimizer(tr, lr):
    """Optimizer over the adapted tensors with predictor lr `lr`; the encoder keeps the online lr ratio."""
    saved = (tr.lr, tr.encoder_lr)
    tr.encoder_lr = lr * tr.encoder_lr / tr.lr
    tr.lr = lr
    opt = tr._make_optimizer()
    tr.lr, tr.encoder_lr = saved
    return opt


def finetune_step(tr, opt, obs, act):
    """One AdaJEPA training step on a batch of segments with a given optimizer."""
    saved = (tr._make_optimizer, tr.steps)
    tr._make_optimizer, tr.steps = (lambda: opt), 1
    try:
        losses = tr.finetune([obs], [act], merge=False)
    finally:
        tr._make_optimizer, tr.steps = saved
    return losses[-1]


def load_calibration_segments(calib_dir, max_episodes=0):
    """Returns a list of episodes, each a list of raw (visual, proprio, actions) segments."""
    files = sorted(f for f in os.listdir(calib_dir) if f.startswith("ep") and f.endswith(".pt"))
    if max_episodes:
        files = files[:max_episodes]
    episodes = []
    for f in files:
        ep = torch.load(os.path.join(calib_dir, f), weights_only=False)
        episodes.append(list(zip(ep["visual"], ep["proprio"], ep["actions"])))
    return episodes


def train_offline_slow_weights(planner):
    """Train slow weights on calibration episodes; keep the lr and step with the lowest holdout loss.

    10% of the calibration episodes are held out. Saves to planner.slow_init.
    """
    cfg = planner.offline
    tr = planner.adajepa_trainer
    tensors = planner._tensors
    assert planner.calib_dir, f"missing {planner.slow_init} and no calib_dir to train it"
    episodes = load_calibration_segments(planner.calib_dir, int(cfg.get("max_episodes", 0) or 0))
    steps = int(cfg.get("max_steps", 2000))
    bs = int(cfg.get("batch", 16))
    eval_every = int(cfg.get("eval_every", 100))
    lrs = [float(x) for x in str(cfg.get("lrs", "5e-4,1e-4,3e-5")).split(",")]

    rng = random.Random(0)
    hold_eps = set(rng.sample(range(len(episodes)), max(1, len(episodes) // 10)))
    train = [s for i, ep in enumerate(episodes) if i not in hold_eps for s in ep]
    hold = [s for i, ep in enumerate(episodes) if i in hold_eps for s in ep]
    hold_batches = [batch_segments(planner.preprocessor, hold[i : i + 32]) for i in range(0, len(hold), 32)]

    def hold_loss():
        obs, act = zip(*hold_batches)
        return float(np.mean(tr.score_segments(list(obs), list(act))))

    theta0 = [t.detach().clone() for t in tensors]
    best, best_lr, best_state = hold_loss(), 0.0, theta0
    for lr in lrs:
        copy_into(tensors, theta0)
        opt = make_optimizer(tr, lr)
        for it in range(1, steps + 1):
            obs, act = batch_segments(planner.preprocessor, rng.sample(train, bs))
            finetune_step(tr, opt, obs, act)
            if it % eval_every:
                continue
            loss = hold_loss()
            print(f"[sea offline] lr {lr:g} step {it} holdout {loss:.6f}")
            if loss < best:
                best, best_lr, best_state = loss, lr, [t.detach().clone() for t in tensors]

    copy_into(tensors, theta0)
    os.makedirs(os.path.dirname(planner.slow_init) or ".", exist_ok=True)
    torch.save({"tensors": [t.cpu() for t in best_state], "best_lr": best_lr}, planner.slow_init)
    print(f"[sea offline] saved {planner.slow_init} (lr {best_lr:g})")


@torch.no_grad()
def copy_into(tensors, values):
    for t, v in zip(tensors, values):
        t.copy_(v)

"""The practice gym maths: how the brain learns from its own games (PPO), trimmed from FirstLight Zero's learner.py.

Plain-language version of what happens here:
  1. `gae` looks back over a finished game and asks "which of my moves were better than expected?" (the advantage).
  2. `ppo_loss` nudges the brain to repeat good moves and avoid bad ones, but only a small step at a time
     (the `clip` keeps it from changing too fast), while also keeping it a little curious (the entropy bonus).
  3. `train_step` runs one such nudge on a batch of games.

No game engine is needed for any of this: it only needs recorded games in the format of `store.py`.
"""
import numpy as np
import torch

DEFAULTS = dict(gamma=.9997, lam=.95, clip=.2, value_clip=.2, vf_coef=.5, ent_coef=.01, max_grad_norm=1.0, lr=3e-4)
# NOTE: the original run's exact settings live in its settings.json / launch.py; these are sensible PPO starting values.


def gae(reward, value, gamma, lam):
    """Generalised advantage estimation over one game. `reward` and `value` are 1-D arrays of the same length."""
    adv = np.zeros_like(reward); last = 0.
    for t in range(len(reward) - 1, -1, -1):
        nxt = value[t + 1] if t + 1 < len(reward) else 0.
        delta = reward[t] + gamma * nxt - value[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


def ppo_loss(logp, entropy, value, old_logp, old_value, adv, ret, valid, s):
    """The same loss as the original learner. All inputs are flat tensors of the same length; `valid` masks padding."""
    adv = (adv - adv[valid.bool()].mean()) / (adv[valid.bool()].std() + 1e-8)
    ratio = torch.exp(logp - old_logp)
    pg = -torch.min(ratio * adv, ratio.clamp(1 - s['clip'], 1 + s['clip']) * adv)
    v_clip = old_value + (value - old_value).clamp(-s['value_clip'], s['value_clip'])
    vl = torch.max((value - ret) ** 2, (v_clip - ret) ** 2)
    w = valid.float()
    return ((pg + s['vf_coef'] * .5 * vl - s['ent_coef'] * entropy) * w).sum() / w.sum().clamp_min(1.)


def train_step(net, opt, x, act, old_logp, old_value, adv, ret, valid, time_steps, settings=None):
    """One PPO update on a padded batch of games.

    `x`, `act` follow the layout built by the original `collate()`: flat [T*B] tensors, time-major.
    `net.evaluate` replays the games through the brain with the moves that were actually played.
    """
    s = {**DEFAULTS, **(settings or {})}
    net.train()
    logp, entropy, value = net.evaluate(x, act, time_steps)
    loss = ppo_loss(logp, entropy, value, old_logp, old_value, adv, ret, valid, s)
    opt.zero_grad(set_to_none=True); loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(net.parameters(), s['max_grad_norm']); opt.step()
    return dict(loss=float(loss), grad_norm=float(grad), entropy=float((entropy * valid.float()).sum() / valid.float().sum()))

from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import A2CPolicy


class PPOPolicy(A2CPolicy):

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        critic_optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_adv: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        if not self._rew_norm:
            assert not self._value_clip, \
                "value clip is available only when `reward_normalization` is True"
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_adv
        self.device = 'cpu'
        self.critic_optim = critic_optim

    def to(self, device):
        self.device = device
        return super().to(device)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        state_dict = {k.replace('network.', 'actor.'):v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = Batch(batch)
        batch.rew[batch.done] = np.sign(batch.rew[batch.done] + 8)
        batch.obs = to_torch(batch.obs, device=self.device).float()
        batch.mask = to_torch(batch.mask, device=self.device)
        batch.obs_next = to_torch(batch.obs_next, device=self.device).float()
        batch.act = to_torch(batch.act, device=self.device)
        batch = self._compute_returns(batch, buffer, indices)
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self.update_forward(b).dist.log_prob(b.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        return batch

    def forward(self, obs):
        self.actor.eval()
        is_batch = obs[1].ndim > 1
        mask = torch.from_numpy(obs[1]).to(self.device)
        shape = mask.shape[:-1]
        mask = mask.view(-1, 235)
        obs = torch.from_numpy(obs[0]).to(self.device).float().view(-1, 161, 4, 9)[:, :145, :, :]
        with torch.no_grad():
            logits = self.actor(obs)
        logits = logits + (logits.min() - logits.max() - 20) * ~mask
        dist = self.dist_fn(logits)
        action = dist.sample()
        return action.view(*shape).cpu().numpy() if is_batch else action[0].cpu().numpy()

    def update_forward(self, batch):
        logits = self.actor(batch.obs[:, :145, :, :])
        logits = logits + (logits.min() - logits.max() - 20) * ~batch.mask
        dist = self.dist_fn(logits)
        return Batch(dist=dist)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        self.actor.eval()
        self.critic.train()
        assert self._batch == batch_size, "for batch norm consistency"
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        first = True
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for b in batch.split(batch_size, shuffle=False, merge_last=True):
                # calculate loss for actor
                dist = self.update_forward(b).dist
                if self._norm_adv:
                    mean, std = b.adv.mean(), b.adv.std() + 1e-5
                    b.adv = (b.adv - mean) / std  # per-batch norm
                ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                assert not first or (ratio == 1).all()
                first = False
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * b.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * b.adv)
                    clip_loss = -torch.where(b.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                value = self.critic(b.obs).flatten()
                if self._value_clip:
                    v_clip = b.v_s + (value -
                                      b.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (b.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                self.optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        set(self.actor.parameters()).union(self.critic.parameters()),
                        max_norm=self._grad_norm,
                        error_if_nonfinite=True
                    )
                self.critic_optim.step()
                self.optim.step()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }

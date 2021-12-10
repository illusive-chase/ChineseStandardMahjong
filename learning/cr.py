from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.utils import RunningMeanStd


class CRPolicy(BasePolicy):

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        value_clip: bool = False,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        reward_normalization = False,
        max_batchsize = 1024,
        lr_scheduler = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=False,
            action_bound_method='',
            **kwargs
        )
        self.actor = actor
        self.critic = critic
        self._gamma = discount_factor
        self._lambda = gae_lambda
        self._value_clip = value_clip
        self._grad_norm = max_grad_norm
        self._rew_norm = reward_normalization
        self._batch = max_batchsize
        if not self._rew_norm:
            assert not self._value_clip, \
                "value clip is available only when `reward_normalization` is True"
        self.device = 'cpu'
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8

    def to(self, device):
        self.device = device
        return super().to(device)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        state_dict = {k.replace('network.', 'actor.'):v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        v_s, v_s_ = [], []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                b.obs = to_torch(b.obs, device=self.device).float()
                b.obs_next = to_torch(b.obs_next, device=self.device).float()
                v_s.append(self.critic(b.obs))
                v_s_.append(self.critic(b.obs_next))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        v_s_ = torch.cat(v_s_, dim=0).flatten().cpu().numpy()
        # when normalizing values, we do not minus self.ret_rms.mean to be numerically
        # consistent with OPENAI baselines' value normalization pipeline. Emperical
        # study also shows that "minus mean" will harm performances a tiny little bit
        # due to unknown reasons (on Mujoco envs, not confident, though).
        if self._rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
            v_s_ = v_s_ * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns, _ = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda
        )
        if self._rew_norm:
            batch.returns = unnormalized_returns / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        return batch

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = Batch(batch)
        # batch.rew[batch.done] = np.sign(batch.rew[batch.done] + 8)
        batch.rew[batch.done] /= 10
        batch = self._compute_returns(batch, buffer, indices)
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
        action = logits.argmax(dim=-1)
        return action.view(*shape).cpu().numpy() if is_batch else action[0].cpu().numpy()

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        assert self._batch == batch_size, "for batch norm consistency"
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        self.critic.train()
        for step in range(repeat):
            for b in batch.split(batch_size, shuffle=True, merge_last=True):
                b = to_torch(b, device=self.device)
                # calculate loss for critic
                value = self.critic(b.obs.float()).flatten()
                if self._value_clip:
                    v_clip = b.v_s + (value -
                                      b.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (b.returns - value).pow(2)
                    vf2 = (b.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (b.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                loss = vf_loss
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        max_norm=self._grad_norm,
                        error_if_nonfinite=True
                    )
                self.optim.step()
                losses.append(loss.item())
                print(loss.item())
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss": losses,
        }

from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from tianshou.utils import RunningMeanStd


class PGPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=False,
            action_bound_method='',
            **kwargs
        )
        self.network = model
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        return super().to(device)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def process_fn(
        self, batch: dict, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each transition.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        batch = Batch(batch)
        batch.rew[batch.done] = np.sign(batch.rew[batch.done] + 8)
        self.eval()
        unnormalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indices, v_s_=v_s_, gamma=self._gamma, gae_lambda=1.0
        )
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return to_torch(batch, device=self.device)

    def forward(self, obs):
        self.eval()
        is_batch = obs[1].ndim > 1
        mask = torch.from_numpy(obs[1]).to(self.device)
        shape = mask.shape[:-1]
        mask = mask.view(-1, 235)
        obs = torch.from_numpy(obs[0]).to(self.device).float().view(-1, 161, 4, 9)[:, :145, :, :]
        with torch.no_grad():
            logits = self.network(obs)
        logits = logits + (logits.min() - logits.max() - 20) * ~mask
        dist = self.dist_fn(logits)
        action = dist.sample()
        # action = logits.argmax(dim=-1)
        return action.view(*shape).cpu().numpy() if is_batch else action[0].cpu().numpy()

    def update_forward(self, batch):
        logits = self.network(batch.obs.float()[:, :145, :, :])
        logits = logits + (logits.min() - logits.max() - 20) * ~batch.mask
        dist = self.dist_fn(logits)
        act = dist.sample()
        return Batch(logits=logits, act=act, state=None, dist=dist)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        ratio_10s = []
        ratio_25s = []
        ratio_100s = []
        eq_ratios = []
        eq_ratios_after = []
        self.eval()
        # assert repeat == 1
        for _ in range(repeat):
            for b in batch.split(batch_size, merge_last=False, shuffle=True):
                self.optim.zero_grad()
                result = self.update_forward(b)
                dist = result.dist
                a = to_torch(b.act, device=self.device)
                ret = to_torch(b.returns, device=self.device)
                log_prob = dist.log_prob(a).reshape(len(ret), -1).transpose(0, 1)
                loss = -(log_prob * ret).mean()
                # loss = -log_prob.mean()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())

                with torch.no_grad():
                    eq_ratios.append((result.logits.argmax(dim=-1) == a).float().mean().item())
                    result = self.update_forward(b)
                    eq_ratios_after.append((result.logits.argmax(dim=-1) == a).float().mean().item())
                    new_log_prob = result.dist.log_prob(a).reshape(len(ret), -1).transpose(0, 1)
                    ratio = (new_log_prob - log_prob).exp().float()
                    ratio[ratio < 1] = 1 / ratio[ratio < 1]
                    ratio_10 = (ratio > 1.1).float().mean().item()
                    ratio_25 = (ratio > 1.25).float().mean().item()
                    ratio_100 = (ratio > 2).float().mean().item()
                    ratio_10s.append(ratio_10)
                    ratio_25s.append(ratio_25)
                    ratio_100s.append(ratio_100)
                    
                break
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return { "loss": losses, "cg-10%": ratio_10s, "cg-25%": ratio_25s, "cg-100%": ratio_100s, "eqr": eq_ratios, "eqr-after": eq_ratios_after }

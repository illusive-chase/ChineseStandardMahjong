# -*- coding: utf-8 -*-
__all__ = ('tianshou_imitation_policy',)

from utils.vec_data import VecData
from torch import nn
import torch
import gym
import numpy as np
from tianshou.data import Batch, to_torch



class tianshou_imitation_policy(nn.Module):
    def __init__(self, network, lr, weight_decay, mode='pi'):
        assert mode in ['pi', 'q', 'v']
        super().__init__()
        self._grad_step = 0
        self.observation_space = gym.spaces.Box(0, 1, shape=VecData.state_shape[1:], dtype=np.bool)
        self.mode = mode
        self.action_space = gym.spaces.Discrete(VecData.action_shape[1])
        self.network = network
        self.device = 'cpu'
        weight_decay_list = (param for name, param in network.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
        no_decay_list = (param for name, param in network.named_parameters() if name[-4:] == 'bias' or "bn" in name)
        parameters = [{'params': weight_decay_list},
                      {'params': no_decay_list, 'weight_decay': 0.}]
        self.optim = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        for m in network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def to(self, device):
        self.device = device
        return super().to(device)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def forward(self, batch, state=None, mask=None):
        if mask is None:
            return Batch(act=batch.obs.gt_action, state=state)
        logits = self.network(batch)
        return logits + (logits.min() - logits.max() - 20) * mask


    def post_process_fn(self, batch, buffer, indices):
        if hasattr(buffer, "update_weight") and hasattr(batch, "weight"):
            buffer.update_weight(indices, batch.weight)

    def update(self, sample_size, buffer, val=False):
        batch, indices = buffer.sample(sample_size)
        if type(batch) is dict:
            batch = Batch(obs=batch)
        obs = to_torch(batch.obs.obs, device=self.device).float().view(-1, 145, 4, 9)
        mask = (~to_torch(batch.obs.mask, device=self.device)).float()
        gt_action = to_torch(batch.obs.gt_action, device=self.device).long()
        rew = to_torch(batch.obs.rew, device=self.device).float()
        losses = []

        if self.mode == 'pi':
            if val:
                logits = self(obs, mask=mask)
                loss = nn.CrossEntropyLoss()(logits, gt_action)
                losses.append(loss.item())
            else:
                for i in range(1):
                    self.optim.zero_grad()
                    logits = self(obs, mask=mask)
                    loss = nn.CrossEntropyLoss()(logits, gt_action)
                    loss.backward()
                    self.optim.step()
                    losses.append(loss.item())
                self.post_process_fn(batch, buffer, indices)

            return {("val-loss" if val else "loss"): losses, "eq-ratio": (logits.detach().argmax(dim=-1) == gt_action).float().mean().item()}
        elif self.mode == 'q':
            norm_rew = (-0.2 * (rew + rew.mean() * 3)).exp()
            norm_rew = 0.4 * norm_rew / (1 + norm_rew).pow(2)
            if val:
                with torch.no_grad():
                    logits = self(obs, mask=mask).squeeze(1)
                    logit = torch.gather(logits.log_softmax(dim=-1), 1, gt_action.unsqueeze(1)).squeeze(0)
                    loss = (-logit.exp() * norm_rew).mean()
                losses.append(loss.item())
            else:
                for i in range(1):
                    self.optim.zero_grad()
                    logits = self(obs, mask=mask).squeeze(1)
                    logit = torch.gather(logits.log_softmax(dim=-1), 1, gt_action.unsqueeze(1)).squeeze(0)
                    loss = (-logit.exp() * norm_rew).mean()
                    loss.backward()
                    self.optim.step()
                    losses.append(loss.item())
                self.post_process_fn(batch, buffer, indices)

            return {("val-loss" if val else "loss"): losses, "eq-ratio": (logits.detach().argmax(dim=-1) == gt_action).float().mean().item()}
        elif self.mode == 'v':
            # rew = rew * 0.1
            # rew = rew.sgn()
            if val:
                with torch.no_grad():
                    logits = self.network(obs)
                    category = torch.empty_like(rew).long()
                    category[:] = 4
                    category[rew < 50] = 3
                    category[rew < 32] = 2
                    category[rew < 0] = 1
                    category[rew < -8] = 0
                    correct_ratio = (logits.argmax(dim=-1) == category).float().mean()
                    win_ratio = (logits.argmax(dim=-1)[category > 2] == category[category > 2]).float().mean()
                    loss = nn.CrossEntropyLoss()(logits, category)
                    # loss = (logits.squeeze(1) - rew * 0.1).pow(2).mean()
                losses.append(loss.item())
            else:
                for i in range(1):
                    if self._grad_step % 5 == 0:
                        self.optim.zero_grad()
                    logits = self.network(obs)
                    category = torch.empty_like(rew).long()
                    category[:] = 4
                    category[rew < 50] = 3
                    category[rew < 32] = 2
                    category[rew < 0] = 1
                    category[rew < -8] = 0
                    correct_ratio = (logits.argmax(dim=-1) == category).float().mean()
                    win_ratio = (logits.argmax(dim=-1)[category > 2] == category[category > 2]).float().mean()
                    loss = nn.CrossEntropyLoss()(logits, category)
                    # loss = (logits.squeeze(1) - rew * 0.1).pow(2).mean()
                    loss.backward()
                    losses.append(loss.item())
                self.post_process_fn(batch, buffer, indices)
                self._grad_step += 1
                if self._grad_step % 5 == 0:
                    self.optim.step()

            # return {("val-loss" if val else "loss"): losses}
            return {("val-loss" if val else "loss"): losses, "cr": [correct_ratio.item()] * 10, "wr": [win_ratio.item()] * 10}

    def map_action(self, action):
        return action



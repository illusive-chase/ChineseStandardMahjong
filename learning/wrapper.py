# -*- coding: utf-8 -*-
__all__ = ('wrapper_policy',)

from utils.vec_data import VecData
from torch import nn
import torch
import numpy as np


class wrapper_policy(nn.Module):
    def __init__(self, network, deterministic=True):
        super().__init__()
        self.network = network
        self.device = 'cpu'
        self.deterministic = deterministic

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict({k.replace('module.', ''):v for k, v in state_dict.items()})

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, obs):
        is_batch = obs[1].ndim > 1
        mask = torch.from_numpy(obs[1]).to(self.device)
        shape = mask.shape[:-1]
        mask = mask.view(-1, 235)
        obs = torch.from_numpy(obs[0]).to(self.device).float().view(-1, 161, 4, 9)[:, :145, :, :]
        with torch.no_grad():
            logits = self.network(obs)
        logits = logits + (logits.min() - logits.max() - 40) * ~mask
        if self.deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            while True:
                action = dist.sample()
                if (dist.log_prob(action).exp() > 1e-10).all():
                    break
        return action.view(*shape).cpu().numpy() if is_batch else action[0].cpu().numpy()

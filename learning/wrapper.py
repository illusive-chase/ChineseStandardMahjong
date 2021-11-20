# -*- coding: utf-8 -*-
__all__ = ('wrapper_policy',)

from utils.vec_data import VecData
from torch import nn
import torch
import gym
import numpy as np


class wrapper_policy(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, shape=VecData.state_shape[1:], dtype=np.bool)
        self.action_space = gym.spaces.Discrete(VecData.action_shape[1])
        self.network = network
        self.device = 'cpu'

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, obs):
        is_batch = obs[0].ndim == 4
        mask = torch.from_numpy(obs[1]).to(self.device).view(-1, 235)
        obs = torch.from_numpy(obs[0]).to(self.device).float().view(-1, 145, 4, 9)
        with torch.no_grad():
            logits = self.network(obs)
        logits[~mask] = logits.min() - 20
        return torch.argmax(logits, dim=-1)

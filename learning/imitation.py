# -*- coding: utf-8 -*-
__all__ = ('tianshou_imitation_policy',)

from utils.vec_data import VecData
from torch import nn
import torch
import gym
import numpy as np
from tianshou.data import Batch, to_torch, to_torch_as



class tianshou_imitation_policy(nn.Module):
    def __init__(self, network, lr):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, shape=VecData.state_shape[1:], dtype=np.bool)
        self.action_space = gym.spaces.Discrete(VecData.action_shape[1])
        self.network = network
        self.device = 'cpu'
        self.optim = torch.optim.Adam(network.parameters(), lr=lr)
        for m in network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def to(self, device):
        self.device = device
        return self.to(device)

    def forward(self, batch, state=None):
        obs = to_torch(batch.obs.obs, device=self.device).float().view(-1, 145, 4, 9)
        return Batch(act=self.network(obs), state=state)

    def post_process_fn(self, batch, buffer, indices):
        if hasattr(buffer, "update_weight") and hasattr(batch, "weight"):
            buffer.update_weight(indices, batch.weight)

    def update(self, sample_size, buffer):
        batch, indices = buffer.sample(sample_size)
        gt_action = to_torch(batch.obs.gt_action, device=self.device).long()
        losses = []

        self.optim.zero_grad()
        action = self(batch).act
        loss = nn.CrossEntropyLoss()(action, gt_action)
        loss.backward()
        self.optim.step()

        losses.append(loss.item())
        self.post_process_fn(batch, buffer, indices)
        return {"loss": losses}

    def map_action(self, action):
        return action



# -*- coding: utf-8 -*-
__all__ = ('tianshou_imitation_policy',)

from utils.vec_data import VecData
from torch import nn
import torch
import gym
import numpy as np
from tianshou.data import Batch, to_torch



class tianshou_imitation_policy(nn.Module):
    def __init__(self, network, lr, weight_decay):
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, shape=VecData.state_shape[1:], dtype=np.bool)
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
        losses = []

        if val:
            action = self(obs, mask=mask)
            loss = nn.CrossEntropyLoss()(action, gt_action)
            losses.append(loss.item())
        else:
            for i in range(1):
                self.optim.zero_grad()
                action = self(obs, mask=mask)
                loss = nn.CrossEntropyLoss()(action, gt_action)
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
            self.post_process_fn(batch, buffer, indices)

        return {("val-loss" if val else "loss"): losses}

    def map_action(self, action):
        return action



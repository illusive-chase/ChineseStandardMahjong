# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
from env.bot import Bot


def MLP(channels, do_bn=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=not do_bn))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def Conv(channels, kernel_size, stride, padding, do_bn):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernel_size, stride=stride, padding=padding, bias=(not do_bn) or (i == n - 1)))
        if i < n - 1:
            if do_bn:
                layers.append(nn.BatchNorm2d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self, state_shape, extra_shape, action_shape, device):
        super().__init__()
        self.embedding1 = nn.Embedding(22, 8)
        self.embedding2 = nn.Embedding(4, 8)
        self.embedding3 = nn.Embedding(8, 8)
        self.conv1 = Conv([state_shape[1], 128], 3, 1, 1, True)
        self.conv2 = Conv([128, 128, 128], 3, 1, 1, True)
        self.conv3 = Conv([128, 128, 128], 3, 1, 1, True)
        self.model = nn.Linear(128 * 4 * 9 + 8 * 4, 256)
        self.state_shape = state_shape
        self.extra_shape = extra_shape
        self.action_shape = action_shape
        self.device = device
    def forward(self, s, **kwargs):
        obs = torch.as_tensor(s['obs']['obs'], device=self.device, dtype=torch.float32).view(-1, self.state_shape[1], 4, 9)
        extra = torch.as_tensor(s['obs']['extra'], device=self.device, dtype=torch.int32)
        state = nn.ReLU(inplace=True)(self.conv1(obs))
        state = nn.ReLU(inplace=True)(state + self.conv2(state))
        state = nn.ReLU(inplace=True)(state + self.conv3(state))
        return nn.ReLU(inplace=True)(self.model(torch.cat((
            state.view(-1, 128 * 4 * 9),
            self.embedding1(extra[:, 0]),
            self.embedding2(extra[:, 1]),
            self.embedding2(extra[:, 2]),
            self.embedding3(extra[:, 3])
        ), dim=1)))

class Actor(nn.Module):
    def __init__(self, net, state_shape, extra_shape, action_shape, device):
        super().__init__()
        self.net = net
        self.model = nn.Sequential(*[
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, np.prod(action_shape[1:]), bias=False)
        ])
        self.device = device
    def forward(self, s, state=None, info={}):
        mask = torch.as_tensor(s['mask'], device=self.device, dtype=torch.int32)
        output = self.model(self.net(s))
        logits = output + (output.min() - output.max() - 20) * (1 - mask)
        return logits, state


def main():
    device = torch.device('cpu')
    net = Net(Bot.state_shape, Bot.extra_shape, Bot.action_shape, device)
    actor = Actor(net, Bot.state_shape, Bot.extra_shape, Bot.action_shape, device)
    state_dict = {
        k[6:]: v
        for k, v in torch.load('./data/policy.pth', map_location='cpu').items() if k[:6] == 'actor.'
    }
    actor.load_state_dict(state_dict)
    actor.eval()
    bot = Bot()
    
    def policy(obs):
        dist = torch.distributions.Categorical
        with torch.no_grad():
            logits, _ = actor(obs, None)
        action = torch.argmax(dist(logits=logits).probs, dim=-1).item()
        return action

    bot.stepOneRound(policy)

if __name__ == "__main__":
    main()

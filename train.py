from Referee import Referee as REnv
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import DQNPolicy, RandomPolicy
from ppo import PPOPolicy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
import torch
import torch.nn as nn
import numpy as np
import argparse
import gym
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
import random



def PPO(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}')
    train_envs = DummyVectorEnv([lambda:REnv(fixPolicy=RandomPolicy(), verbose=False)] * 8)
    test_envs = DummyVectorEnv([lambda:REnv(fixPolicy=RandomPolicy(), verbose=False, eval=True)] * 5)
    class Net(nn.Module):
        def __init__(self, state_shape, extra_shape, action_shape, device):
            super().__init__()
            self.embedding1 = nn.Embedding(22, 8)
            self.embedding2 = nn.Embedding(4, 8)
            self.embedding3 = nn.Embedding(8, 8)
            self.linear = nn.Sequential(*[
                nn.Linear(np.prod(state_shape[1:]), args.hsize), nn.ReLU(inplace=True),
            ])
            self.model = nn.Sequential(*[
                nn.Linear(args.hsize+8*4, args.hsize), nn.ReLU(inplace=True)
            ])
            self.device = device
        def forward(self, s, **kwargs):
            obs = torch.as_tensor(s.obs.obs, device=self.device, dtype=torch.float32)
            extra = torch.as_tensor(s.obs.extra, device=self.device, dtype=torch.int32)
            batch = obs.shape[0]
            state = self.linear(obs.view(batch, -1))
            return self.model(torch.cat((
                state,
                self.embedding1(extra[:, 0]),
                self.embedding2(extra[:, 1]),
                self.embedding2(extra[:, 2]),
                self.embedding3(extra[:, 3])
            ), dim=1))

    class Critic(nn.Module):
        def __init__(self, net, state_shape, extra_shape, action_shape, device):
            super().__init__()
            self.net = net
            self.model = nn.Sequential(*[
                nn.Linear(args.hsize, args.hsize), nn.ReLU(inplace=True),
                nn.Linear(args.hsize, 1)
            ])
        def forward(self, s, **kwargs):
            return self.model(self.net(s))

    class Actor(nn.Module):
        def __init__(self, net, state_shape, extra_shape, action_shape, device):
            super().__init__()
            self.net = net
            self.model = nn.Sequential(*[
                nn.Linear(args.hsize, args.hsize), nn.ReLU(inplace=True),
                nn.Linear(args.hsize, np.prod(action_shape[1:]), bias=False)
            ])
            self.device = device
        def forward(self, s, state=None, info={}):
            mask = torch.as_tensor(s.mask, device=self.device, dtype=torch.int32)
            probs = nn.Softmax(dim=1)(self.model(self.net(s))) * mask
            return probs, state

    net = Net(REnv.state_shape, REnv.extra_shape, REnv.action_shape, device).to(device)
    critic = Critic(net, REnv.state_shape, REnv.extra_shape, REnv.action_shape, device).to(device)
    actor = Actor(net, REnv.state_shape, REnv.extra_shape, REnv.action_shape, device).to(device)
    dist = torch.distributions.Categorical

    # orthogonal initialization
    for m in set(actor.modules()).union(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=args.lr, eps=1e-8)
    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=0.5,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        recompute_advantage=True,
        action_space=REnv.action_space
    )
    if args.mode == 'eval':
        policy.load_state_dict(torch.load(f'./{args.log_dir}/{args.exp_name}/policy.pth' if args.load_path == '' else args.load_path, map_location=device))
        return '', policy
    buffer = VectorReplayBuffer(1000000, 8)
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)
    writer = SummaryWriter(f'./{args.log_dir}/{args.exp_name}')
    logger = TensorboardLogger(writer)
    result = onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=100, step_per_epoch=40000, repeat_per_collect=10, episode_per_collect=40,
        episode_per_test=50, batch_size=256,
        stop_fn=lambda mean_rewards: False,
        save_fn=lambda policy: torch.save(policy.state_dict(), f'./{args.log_dir}/{args.exp_name}/policy.pth'),
        logger=logger)
    with open(f'./{args.log_dir}/{args.exp_name}/args.txt', 'w') as f:
        f.write(str(args))
    return result, policy


def eval(args):
    result, policy = PPO(args)
    policy.eval()
    vis_envs = DummyVectorEnv([lambda:REnv(fixPolicy=policy, verbose=True)])
    result = Collector(policy, vis_envs).collect(n_episode=1, render=.5)

def main(args):
    result, policy = PPO(args)
    print(f'Finished training! Use {result["duration"]}')
    print(result)
    policy.eval()
    vis_envs = DummyVectorEnv([lambda:REnv(fixPolicy=policy, verbose=True)])
    result = Collector(policy, vis_envs).collect(n_episode=1, render=.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--cuda', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hsize', type=int, default=256)
    parser.add_argument('--load-path', type=str, default='')
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.9)
    args = parser.parse_args()
    if args.mode == 'eval':
        eval(args)
    elif args.mode == 'train':
        main(args)
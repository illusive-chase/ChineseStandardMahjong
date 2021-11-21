from env.runner import Runner as REnv
from test import Net, Actor
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import DQNPolicy, RandomPolicy
from learning.ppo import PPOPolicy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.data import Batch
import torch
import torch.nn as nn
import numpy as np
import argparse
import gym
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger


def wrapped_policy(raw_policy):
    def wrapped(x):
        return raw_policy(Batch({'obs': x})).act
    return wrapped


def PPO(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.cuda}')
    random_policy = RandomPolicy()
    fixPolicy = wrapped_policy(random_policy)
    train_envs = DummyVectorEnv([lambda:REnv(fixPolicy=fixPolicy, verbose=False)] * 8)
    test_envs = DummyVectorEnv([lambda:REnv(fixPolicy=fixPolicy, verbose=False, eval=True)] * 5)

    class Critic(nn.Module):
        def __init__(self, net, state_shape, extra_shape, action_shape, device):
            super().__init__()
            self.net = net
            self.model = nn.Sequential(*[
                nn.Linear(256, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 1)
            ])
        def forward(self, s, **kwargs):
            return self.model(self.net(s))

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
        lambda x:dist(logits=x),
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
    buffer = VectorReplayBuffer(20000, 8)
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)
    writer = SummaryWriter(f'./{args.log_dir}/{args.exp_name}')
    logger = TensorboardLogger(writer)
    with open(f'./{args.log_dir}/{args.exp_name}/args.txt', 'w') as f:
        f.write(str(args))
    result = onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=args.max_epoch, step_per_epoch=40000, repeat_per_collect=10,
        episode_per_collect=40, episode_per_test=50, batch_size=256,
        stop_fn=lambda mean_rewards: False,
        save_fn=lambda policy: torch.save(policy.state_dict(), f'./{args.log_dir}/{args.exp_name}/policy.pth'),
        logger=logger)
    return result, policy


def eval(args):
    torch.set_num_threads(1)
    result, policy = PPO(args)
    policy.eval()
    vis_envs = DummyVectorEnv([lambda:REnv(fixPolicy=wrapped_policy(policy), verbose=True)])
    result = Collector(policy, vis_envs).collect(n_episode=1, render=.5)

def main(args):
    torch.set_num_threads(1)
    result, policy = PPO(args)
    print(f'Finished training! Use {result["duration"]}')
    print(result)
    policy.eval()
    vis_envs = DummyVectorEnv([lambda:REnv(fixPolicy=wrapped_policy(policy), verbose=True)])
    result = Collector(policy, vis_envs).collect(n_episode=1, render=.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=100)
    # parser.add_argument('--hsize', type=int, default=512)
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

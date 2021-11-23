from utils.policy import random_policy
from env.runner import Runner as REnv
from learning.wrapper import wrapper_policy
from learning.model import *
import torch
import argparse
import tqdm



def performance(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu') if args.cuda < 0 else torch.device(f'cuda:{args.cuda}')
    network = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm)
    policy = wrapper_policy(network).to(device)
    policy.load(args.path)
    policy.eval()

    other_policy = random_policy()
    if args.compare != '':
        other_network = resnet18(use_bn=True)
        other_policy = wrapper_policy(other_network).to(device)
        other_policy.load(args.compare)
        other_policy.eval()

    total_reward = 0
    env = REnv(other_policy=other_policy, seed=args.seed, verbose=False, eval=True)
    with tqdm.trange(100, desc=f"Matching", dynamic_ncols=True, ascii=True) as t:
        for eps in t:
            for i in range(4):
                if i == 0:
                    env.tileWallDummy = None
                obs = env.reset()
                done = False
                while not done:
                    obs, rew, done, info = env.step(policy(obs))
                total_reward += rew / 4
            t.set_postfix(rew=total_reward / (eps + 1))
    print('Perf: {:.1f}'.format(total_reward / eps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-p', '--path', type=str, default='./data/best.pth')
    parser.add_argument('-cp', '--compare', type=str, default='')
    parser.add_argument('-cu', '--cuda', type=int, default=-1)
    parser.add_argument('-bn', '--batch-norm', action='store_true')
    parser.add_argument('--resnet', type=int, choices=[18, 34, 50, 101, 152], default=18)
    args = parser.parse_args()
    performance(args)
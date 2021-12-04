from utils.policy import random_policy
from env.multirunner import MultiRunner as MEnv
from learning.wrapper import wrapper_policy
from learning.model import *
import torch
import argparse



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

    env = MEnv(policy, other_policy, n_env_parallel=1, n_torch_parallel=2, max_env_num=300, max_batch_size=20000, max_eps=1000)
    env.collect(eval=True, verbose=(not args.quiet))
    print(env.mean_reward())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-p', '--path', type=str, default='./data/best.pth')
    parser.add_argument('-cp', '--compare', type=str, default='')
    parser.add_argument('-cu', '--cuda', type=int, default=-1)
    parser.add_argument('-bn', '--batch-norm', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('--resnet', type=int, choices=[18, 34, 50, 101, 152], default=18)
    args = parser.parse_args()
    performance(args)
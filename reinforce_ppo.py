from learning.reinforce_trainer import onpolicy_trainer
from learning.ppo import PPOPolicy
from learning.wrapper import wrapper_policy
from learning.model import *
import torch
import argparse
import numpy as np
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter


class LazyOptimizer:
    def __init__(self):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    sample_device = torch.device('cpu') if args.sample_cuda < 0 else torch.device(f'cuda:{args.sample_cuda}')
    update_device = torch.device('cpu') if args.update_cuda < 0 else torch.device(f'cuda:{args.update_cuda}')

    network = resnet18(use_bn=True, dropout=args.dropout).to(update_device)
    critic = resnet18(use_bn=False, dropout=args.dropout, shape=(161, 1)).to(update_device)

    param_set = set(network.parameters())
    bn_set = set(sum([list(m.parameters()) for m in network.modules() if isinstance(m, torch.nn.BatchNorm2d)], []))
    assert len(bn_set) == len(param_set & bn_set) and len(bn_set) > 0
    # optim = torch.optim.Adam(param_set - bn_set, lr=args.learning_rate)
    optim = torch.optim.Adam(param_set, lr=args.learning_rate)

    param_set = set(critic.parameters())
    '''bn_set = set(sum([list(m.parameters()) for m in critic.modules() if isinstance(m, torch.nn.BatchNorm2d)], []))
    assert len(bn_set) == len(param_set & bn_set) and len(bn_set) > 0'''
    critic_optim = torch.optim.Adam(param_set, lr=args.learning_rate)

    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        network,
        critic,
        optim,
        critic_optim,
        lambda x:dist(logits=x),
        discount_factor=0.95,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=1e-4,
        gae_lambda=0.95,
        max_batchsize=args.batch_size
    ).to(update_device)

    # orthogonal initialization
    for m in set(network.modules()).union(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    other_policy = wrapper_policy(resnet18(use_bn=True, dropout=args.dropout)).to(sample_device)
    if args.path:
        policy.load(args.path)
    if args.compare_path:
        other_policy.load(args.compare_path)

    writer = SummaryWriter(f'./{args.log_dir}/{args.exp_name}')
    logger = TensorboardLogger(writer, update_interval=5)
    with open(f'./{args.log_dir}/{args.exp_name}/args.txt', 'w') as f:
        f.write(str(args))

    onpolicy_trainer(
        policy,
        other_policy,
        max_epoch=args.num_epoch,
        collect_per_epoch=5,
        test_eps_per_epoch=100,
        step_per_collect=args.batch_size * 5,
        repeat_per_collect=3,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        logger=logger,
        save_fn=lambda p: torch.save(p.state_dict(), f'./{args.log_dir}/{args.exp_name}/policy.pth'),
        force_to_save=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-d', '--log-dir', required=True, type=str)
    parser.add_argument('-e', '--exp-name', required=True, type=str)
    parser.add_argument('-p', '--path', type=str, default='data/best.pth')
    parser.add_argument('-cp', '--compare-path', type=str, default='data/best.pth')
    parser.add_argument('-scu', '--sample-cuda', type=int, default=-1)
    parser.add_argument('-ucu', '--update-cuda', type=int, default=-1)
    parser.add_argument('-ne', '--num-epoch', type=int, default=10000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch-size', type=int, default=4096)
    parser.add_argument('-nbn', '--no-batch-norm', action='store_true')
    parser.add_argument('-dp', '--dropout', type=float, default=0)
    parser.add_argument('--resnet', type=int, choices=[18, 34, 50, 101, 152], default=18)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()
    assert args.resnet == 18
    assert not args.eval
    assert not args.no_batch_norm
    train(args)
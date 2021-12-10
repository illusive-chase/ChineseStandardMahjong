from learning.reinforce_trainer import onpolicy_trainer
from learning.pg import PGPolicy
from learning.wrapper import wrapper_policy
from learning.model import *
import torch
import argparse
import numpy as np
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    sample_device = torch.device('cpu') if args.sample_cuda < 0 else torch.device(f'cuda:{args.sample_cuda}')
    update_device = torch.device('cpu') if args.update_cuda < 0 else torch.device(f'cuda:{args.update_cuda}')

    network = resnet18(use_bn=not args.no_batch_norm, dropout=args.dropout).to(update_device)

    param_set = set(network.parameters())
    bn_set = set(sum([list(m.parameters()) for m in network.modules() if isinstance(m, torch.nn.BatchNorm2d)], []))
    assert args.no_batch_norm or len(bn_set) == len(param_set & bn_set) and len(bn_set) > 0

    optim = torch.optim.Adam(param_set - bn_set, lr=args.learning_rate)
    dist = torch.distributions.Categorical
    policy = PGPolicy(
        network,
        optim,
        lambda x:dist(logits=x),
        discount_factor=0.95
    ).to(update_device)

    # orthogonal initialization
    for m in network.modules():
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
        step_per_collect=args.batch_size * 2,
        repeat_per_collect=1,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        logger=logger,
        save_fn=lambda p: torch.save(p.state_dict(), f'./{args.log_dir}/{args.exp_name}/policy.pth')
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
    parser.add_argument('-bs', '--batch-size', type=int, default=32768)
    parser.add_argument('-nbn', '--no-batch-norm', action='store_true')
    parser.add_argument('-dp', '--dropout', type=float, default=0)
    parser.add_argument('--resnet', type=int, choices=[18, 34, 50, 101, 152], default=18)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()
    assert args.resnet == 18
    assert not args.eval
    train(args)
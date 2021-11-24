from utils.paired_data import PairedDataset
from env.runner import Runner as REnv
from learning.imitation import tianshou_imitation_policy
from learning.wrapper import wrapper_policy
from learning.imitation_trainer import offline_trainer
import argparse
import torch
import time
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from learning.model import *


def evaluate(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu') if args.cuda < 0 else torch.device(f'cuda:{args.cuda}')
    network = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)
    policy = wrapper_policy(network).to(device)
    policy.load(f'./{args.log_dir}/{args.exp_name}/policy.pth')
    policy.eval()

    env = REnv(other_policy=policy, seed=args.seed, verbose=True)
    obs = env.reset()
    done = False
    while not done:
        obs, rew, done, info = env.step(policy(obs))
        env.render()
        time.sleep(0.5)
    env.render()


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu') if args.cuda < 0 else torch.device(f'cuda:{args.cuda}')
    dataset = PairedDataset(args.aug)
    with open(args.file, 'rb') as f:
        dataset.load(f)
    train_set, val_set = dataset.split(0.001)
    val_set.augmentation = 12
    network = eval(f'resnet{args.resnet}')(use_bn=args.batch_norm, dropout=args.dropout)
    # network = Slider()
    policy = tianshou_imitation_policy(network, lr=args.learning_rate, weight_decay=args.weight_decay).to(device)
    if args.path != '':
        policy.load(args.path)
    writer = SummaryWriter(f'./{args.log_dir}/{args.exp_name}')
    logger = TensorboardLogger(writer, update_interval=5)
    with open(f'./{args.log_dir}/{args.exp_name}/args.txt', 'w') as f:
        f.write(str(args))
    result = offline_trainer(
        policy,
        train_set,
        val_set,
        max_epoch=args.num_epoch,
        update_per_epoch=500,
        batch_size=args.batch_size,
        save_fn=lambda p: torch.save(p.state_dict(), f'./{args.log_dir}/{args.exp_name}/policy.pth'),
        logger=logger)
    return result, policy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-f', '--file', type=str, default='pair.pkl')
    parser.add_argument('-d', '--log-dir', required=True, type=str)
    parser.add_argument('-e', '--exp-name', required=True, type=str)
    parser.add_argument('-p', '--path', type=str, default='')
    parser.add_argument('-cu', '--cuda', type=int, default=-1)
    parser.add_argument('-ne', '--num-epoch', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-2)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0)
    parser.add_argument('-bs', '--batch-size', type=int, default=512)
    parser.add_argument('-dp', '--dropout', type=float, default=0.5)
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('-bn', '--batch-norm', action='store_true')
    parser.add_argument('--resnet', type=int, choices=[18, 34, 50, 101, 152], default=18)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
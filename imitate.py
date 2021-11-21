from env.imitator import Imitator as IEnv
from env.runner import Runner as REnv
from utils.match_data import MatchDataset
from learning.imitation import tianshou_imitation_policy
from learning.wrapper import wrapper_policy
from learning.imitation_trainer import imitation_trainer
import argparse
import torch
import time
from tianshou.trainer import offline_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from learning.model import resnet34 as resnet



def eval(args):
    torch.set_num_threads(1)
    device = torch.device('cpu') if args.cuda < 0 else torch.device(f'cuda:{args.cuda}')
    network = resnet(use_bn=False)
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
    torch.set_num_threads(1)
    device = torch.device('cpu') if args.cuda < 0 else torch.device(f'cuda:{args.cuda}')
    dataset = MatchDataset(args.file)
    envs = SubprocVectorEnv([lambda:IEnv(dataset, verbose=False, seed=args.seed + 1000 * i) for i in range(args.parallel)])
    val_envs = DummyVectorEnv([lambda:IEnv(dataset, verbose=False, seed=0)])
    network = resnet(use_bn=False)
    policy = tianshou_imitation_policy(network, lr=args.learning_rate).to(device)
    train_collector = Collector(policy, envs, VectorReplayBuffer(100000, args.parallel))
    val_collector = Collector(policy, val_envs, VectorReplayBuffer(5000, 1))
    writer = SummaryWriter(f'./{args.log_dir}/{args.exp_name}')
    logger = TensorboardLogger(writer, update_interval=5)
    with open(f'./{args.log_dir}/{args.exp_name}/args.txt', 'w') as f:
        f.write(str(args))
    result = imitation_trainer(
        policy,
        train_collector,
        val_collector,
        max_epoch=args.num_epoch,
        update_per_epoch=50,
        episode_per_train=100,
        batch_size=8196,
        save_fn=lambda p: torch.save(p.state_dict(), f'./{args.log_dir}/{args.exp_name}/policy.pth'),
        logger=logger)
    return result, policy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-f', '--file', type=str, default='expert.pkl')
    parser.add_argument('-d', '--log-dir', required=True, type=str)
    parser.add_argument('-e', '--exp-name', required=True, type=str)
    parser.add_argument('-p', '--parallel', type=int, default=1)
    parser.add_argument('-cu', '--cuda', type=int, default=-1)
    parser.add_argument('-ne', '--num-epoch', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    if args.eval:
        eval(args)
    else:
        train(args)
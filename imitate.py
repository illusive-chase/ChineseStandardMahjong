from env.imitator import Imitator as IEnv
from learning.imitation import tianshou_imitation_policy
import argparse
import torch
from tianshou.trainer import offline_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from learning.model import resnet18




def main(args):
    torch.set_num_threads(1)
    parallel = 1
    envs = DummyVectorEnv([lambda:IEnv(args.file, verbose=False, seed=args.seed + 1000 * i) for i in range(parallel)])
    network = resnet18(use_bn=True)
    policy = tianshou_imitation_policy(network, lr=args.learning_rate).to(args.cuda)
    buffer = VectorReplayBuffer(20000, parallel)
    test_collector = Collector(policy, envs, buffer)
    writer = SummaryWriter(f'./{args.log_dir}/{args.exp_name}')
    logger = TensorboardLogger(writer)
    with open(f'./{args.log_dir}/{args.exp_name}/args.txt', 'w') as f:
        f.write(str(args))
    result = offline_trainer(
        policy,
        buffer,
        test_collector,
        max_epoch=args.num_epoch,
        update_per_epoch=10,
        episode_per_test=100,
        batch_size=256,
        save_fn=lambda policy: torch.save(policy.state_dict(), f'./{args.log_dir}/{args.exp_name}/policy.pth'),
        logger=logger,
        verbose=False)
    return result, policy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-f', '--file', type=str, default='expert.pkl')
    parser.add_argument('-d', '--log-dir', required=True, type=str)
    parser.add_argument('-e', '--exp-name', required=True, type=str)
    parser.add_argument('-cu', '--cuda', type=str, default='cpu')
    parser.add_argument('-ne', '--num-epoch', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
from referee import Referee as REnv
from tianshou.data import Collector
from tianshou.policy import RandomPolicy
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch
import argparse
import numpy as np

def wrapped_policy(raw_policy):
    def wrapped(x):
        return raw_policy(Batch({'obs': x})).act
    return wrapped

def main(args):
    policy = RandomPolicy()
    vis_envs = DummyVectorEnv([lambda:REnv(fixPolicy=wrapped_policy(policy), seed=args.seed, verbose=True)])
    result = Collector(policy, vis_envs).collect(n_episode=100, render=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
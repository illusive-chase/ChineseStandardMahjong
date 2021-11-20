from env.runner import Runner as REnv
from env.imitator import Imitator as IEnv
from utils.policy import random_policy, stdin_policy, imitation_policy
from utils.match_data import MatchDataset
import argparse
import numpy as np
import time
import random



def random_test(args):
    # episode per second = 33.5
    env = REnv(other_policy=random_policy(), seed=args.seed, verbose=False)
    policy = random_policy()
    start = time.time()
    for eps in range(500):
        print(eps)
        obs = env.reset()
        done = False
        while not done:
            obs, rew, done, info = env.step(policy(obs))
        env.render()
    print('EPS: {:.1f}'.format(eps / (time.time() - start)))

def imitation_test(args):
    # episode per second = 47.6
    dataset = MatchDataset('expert.pkl')
    start = time.time()
    for eps in range(500):
        match_data = dataset.get(random.randint(0, dataset.size()))
        policy = imitation_policy(match_data)
        env = REnv(other_policy=policy, seed=args.seed, verbose=False)
        obs = env.reset(match_data)
        done = False
        while not done:
            obs, rew, done, info = env.step(policy(obs))
        env.render()
        print(match_data.match_id, rew)
    print('EPS: {:.1f}'.format(eps / (time.time() - start)))

def imitator_test(args):
    env = IEnv('expert.pkl', False, args.seed)
    start = time.time()
    for eps in range(500):
        print(eps)
        policy = lambda x:None
        obs = env.reset()
        done = False
        while not done:
            obs, rew, done, info = env.step(policy(obs))
        env.render()
    print('EPS: {:.1f}'.format(eps / (time.time() - start)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    imitation_test(args)
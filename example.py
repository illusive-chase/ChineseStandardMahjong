from env.runner import Runner as REnv
from utils.policy import random_policy, stdin_policy
import argparse
import numpy as np
import time


def main(args):
    # episode per second = 30
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
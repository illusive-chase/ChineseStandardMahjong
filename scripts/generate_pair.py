import argparse
from utils.match_data import MatchDataset
from utils.paired_data import PairedDataset
from env.imitator import Imitator as IEnv
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('match_file', type=str)
    parser.add_argument('-o', '--output', type=str, default='a.out')
    args = parser.parse_args()
    dataset = MatchDataset(args.match_file)
    pdataset = PairedDataset()
    env = IEnv(dataset, verbose=False, seed=1)
    for i in range(dataset.size()):
        obs = env.reset(i)
        done = False
        while not done:
            pdataset.add(obs['obs'], obs['gt_action'], obs['mask'])
            obs, rew, done, info = env.step(None)
        print(pdataset.size)
    with open(args.output, 'wb') as f:
        pdataset.dump(f)
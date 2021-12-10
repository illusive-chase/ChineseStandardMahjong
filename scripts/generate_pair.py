import argparse
from utils.match_data import MatchDataset
from utils.paired_data import PairedDataset
from env.imitator import Imitator as IEnv
import numpy as np
import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('match_file', type=str)
    parser.add_argument('-o', '--output', type=str, default='a.out')
    args = parser.parse_args()
    dataset = MatchDataset(args.match_file)
    pdataset = PairedDataset()
    env = IEnv(dataset, verbose=False, seed=1)
    with tqdm.trange(dataset.size(), desc=f"Generating PairedDataset", dynamic_ncols=True, ascii=True) as t:
        for i in t:
            obs_lst = [[], [], [], []]
            obs = env.reset(i)
            done = False
            while not done:
                obs_lst[obs['player']].append({'obs': obs['obs'].copy(), 'mask': obs['mask'].copy(), 'gt_action': obs['gt_action']})
                obs, rew, done, info = env.step(None)
            for i in range(4):
                for obs in obs_lst[i]:
                    pdataset.add(obs['obs'], obs['gt_action'], obs['mask'], int(rew[i]))
            t.set_postfix(size=pdataset.size())
    with open(args.output, 'wb') as f:
        pdataset.dump(f)
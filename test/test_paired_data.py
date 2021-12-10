import argparse
from utils.match_data import MatchDataset
from utils.paired_data import PairedDataset
from env.imitator import Imitator as IEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('match_file', type=str)
    args = parser.parse_args()
    dataset = MatchDataset(args.match_file)
    pdataset = PairedDataset()
    env = IEnv(dataset, verbose=False, seed=1)
    for i in range(100):
        obs = env.reset()
        done = False
        while not done:
            pdataset.reset()
            pdataset.add(obs['obs'], obs['gt_action'], obs['mask'], i)
            item = pdataset.sample(0)[0]
            check = bool((obs['obs'] == item['obs'][0]).all() & (obs['gt_action'] == item['gt_action']).all() & (obs['mask'] == item['mask'][0]).all() & (obs['rew'] == item['rew']).all())
            if (not check):
                print('incorrect encode/decode')
                print((obs['obs'] != item['obs'][0]).nonzero())
                print((obs['gt_action'] != item['gt_action']).nonzero())
                print((obs['mask'] != item['mask'][0]).nonzero())
                print((obs['rew'] != item['rew']).nonzero())
            obs, rew, done, info = env.step(None)
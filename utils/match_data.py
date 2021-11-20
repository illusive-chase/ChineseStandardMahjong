# -*- coding: utf-8 -*-

__all__ = ('MatchData',)

from utils.tile_traits import str2tile, tile2str
import pickle

class MatchData:
    def __init__(self, match_id, id, quan, tile_wall, actions, final_info, scores):
        self.match_id = match_id
        self.id = id
        self.quan = quan
        self.tile_wall = tile_wall
        self.actions = actions
        self.final_info = final_info
        self.scores = scores

    def dump(self, fb):
        # f.write(self.match_id + '\n')
        # f.write(str(self.id) + '\n')
        # f.write(str(self.quan) + '\n')
        # f.write(' '.join(self.tile_wall) + '\n')
        # for i in range(4):
        #     f.write(' '.join([str(item) for item in self.actions[i]]) + '\n')
        # f.write(self.final_info + '\n')
        # f.write(' '.join([str(item) for item in self.scores]) + '\n')
        pickle.dump(self, fb)

    @classmethod
    def load(self, fb):
        # self.match_id = f.readline().rstrip()
        # self.id = int(f.readline().rstrip())
        # self.quan = int(f.readline().rstrip())
        # self.tile_wall = f.readline().rstrip().split()
        # self.actions = np.asarray([f.readline().rstrip().split() for i in range(4)])
        # self.final_info = f.readline().rstrip()
        # self.scores = np.asarray(f.readline().rstrip().split())
        return pickle.load(fb)

class MatchDataset:
    def __init__(self, path):
        self.matches = []
        with open(path, 'rb') as fb:
            while True:
                try:
                    match = MatchData.load(fb)
                except EOFError:
                    break
                self.matches.append(match)

    def size(self):
        return len(self.matches)
        
    def get(self, idx):
        return self.matches[idx]

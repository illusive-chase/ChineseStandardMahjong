# -*- coding: utf-8 -*-

__all__ = ('MatchData',)


class MatchData:
    def __init__(self, match_id, id, quan, tile_wall, actions, final_info, scores):
        self.match_id = match_id
        self.id = id
        self.quan = quan
        self.tile_wall = tile_wall
        self.actions = actions
        self.final_info = final_info
        self.scores = scores

    def dump(self, f):
        f.write(self.match_id + '\n')
        f.write(str(self.id) + '\n')
        f.write(str(self.quan) + '\n')
        f.write(' '.join(self.tile_wall) + '\n')
        for i in range(4):
            f.write(' '.join([str(item) for item in self.actions[i]]) + '\n')
        f.write(self.final_info + '\n')
        f.write(' '.join([str(item) for item in self.scores]) + '\n')

    @classmethod
    def load(self, f):
        self.match_id = f.readline().rstrip()
        self.id = int(f.readline().rstrip())
        self.quan = int(f.readline().rstrip())
        self.tile_wall = f.readline().rstrip().split()
        self.actions = np.asarray([f.readline().rstrip().split() for i in range(4)])
        self.final_info = f.readline().rstrip()
        self.scores = np.asarray(f.readline().rstrip().split())

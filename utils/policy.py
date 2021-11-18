# -*- coding: utf-8 -*-
__all__ = ('random_policy', 'stdin_policy', 'imitation_policy', 'inference_policy')

from utils.distribution import sample_by_mask
from utils.match_data import MatchData
import numpy as np


class random_policy:
    def __init__(self):
        pass

    def __call__(self, obs):
        return sample_by_mask(obs[1])

class stdin_policy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        assert self.env.verbose, "To use stdin policy, you must set env to be verbose."
        self.env.render()
        if obs[1].astype(np.uint8).sum() == 1:
            return obs[1].nonzero()[0]
        return input('> ')


class inference_policy:
    def __init__(self, match_lines):
        self.inferred = [[], [], [], []]
        self.match_lines = match_lines[6:-2]
        self.idx = 0
        self.match_id = match_lines[0].split()[-1]
        self.quan = int(match_lines[1].split()[-1])
        self.tile_wall = [[], [], [], []]
        self.fan_info = match_lines[-2]
        self.scores = list(map(int, match_lines[-1].split()[1:]))
        self.translate_fn = None
        self.last = None
        for i in range(4):
            self.tile_wall[i] = match_lines[2+i].split()[3:]
        no_ignored = []
        for idx, line in enumerate(self.match_lines):
            lst = line.split()
            if lst[2] == "Draw":
                self.tile_wall[int(lst[1])].append(lst[3])
            elif lst[2] == "Chi":
                lst[2] = "rChi"
            if 'Ignore' in lst:
                subset = [' '.join(lst[:lst.index('Ignore')])]
                while 'Ignore' in lst:
                    j = lst.index('Ignore')
                    subset.append(' '.join(lst[j+1:j+4] + [lst[3]]))
                    lst = lst[:j] + lst[j+4:]
                subset.sort(key=lambda x:int(x[len('Player x') - 1]))
                no_ignored += subset
            else:
                no_ignored.append(' '.join(lst))
        self.match_lines = no_ignored + ['PLAY 4 PASS PASS']
        for i in range(4):
            self.tile_wall[i] = self.tile_wall[i] + ['??'] * (34 - len(self.tile_wall[i]))
        assert sum([len(self.tile_wall[i]) == 34 for i in range(4)]) == 4
        

    def __call__(self, obs):
        players = obs[2].reshape(-1)
        mask = obs[1].reshape(players.shape[0], -1)
        lst = self.match_lines[self.idx].split()
        while lst[2] == 'Draw':
            self.last = None
            self.idx += 1
            lst = self.match_lines[self.idx].split()
        next_player = int(lst[1])
        next_last_tile = lst[3] if lst[2] == 'Play' else None
        for idx, player in enumerate(players):
            if player == next_player:
                action = self.valid_to_do(mask[idx], lst[2], lst[3])
                if action is not None:
                    self.inferred[player].append(action)
                    self.idx += 1
                    lst = self.match_lines[self.idx].split()
                    next_player = int(lst[1])
                    continue
            self.inferred[player].append(self.translate_fn('PASS'))
        if next_last_tile:
            self.last = next_last_tile
        if players.shape[0] == 1:
            return self.inferred[players[0]][-1]
        return np.asarray([self.inferred[player][-1] for player in players])
        
    def valid_to_do(self, mask, action, tile):
        action = action.upper()
        if action in ['DRAW', 'PASS']:
            return None
        if action == 'CHI':
            if tile[1] <= '7':
                try_action = self.translate_fn(action + ' ' + tile[0] + chr(ord(tile[1]) + 1), self.last)
                if mask[try_action]:
                    return try_action
            if '2' <= tile[1] <= '8':
                try_action = self.translate_fn(action + ' ' + tile, self.last)
                if mask[try_action]:
                    return try_action
            if '3' <= tile[1]:
                try_action = self.translate_fn(action + ' ' + tile[0] + chr(ord(tile[1]) - 1), self.last)
                if mask[try_action]:
                    return try_action
            return None
        if action == 'RCHI':
            action = 'CHI'
        if action in ['CHI', 'BUGANG', 'PLAY']:
            action += ' ' + tile
        elif action == 'GANG' and self.last is None:
            action += ' ' + tile
        
        try_action = self.translate_fn(action, self.last)
        if not mask[try_action]:
            return None
        return try_action

    def as_match_data(self, id=0):
        return MatchData(
            match_id=self.match_id,
            id=id,
            quan=self.quan,
            tile_wall=sum(map(lambda x:list(reversed(x)), self.tile_wall), []),
            actions=list(map(np.asarray, self.inferred)),
            final_info=self.fan_info,
            scores=np.asarray(self.scores)
        )
            




class imitation_policy:
    pass

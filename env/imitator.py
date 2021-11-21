# -*- coding: utf-8 -*-

__all__ = ('Imitator',)


from env.runner import Runner, FinishError
from utils.match_data import MatchDataset
from utils.policy import imitation_policy
import numpy as np


class Imitator(Runner):
    def __init__(self, path, verbose=False, seed=1):
        super().__init__(None, verbose=verbose, seed=seed, eval=False)
        if type(path) is MatchDataset:
            self.dataset = path
        else:
            self.dataset = MatchDataset(path)

    def reset(self):
        match_data = self.dataset.get(np.random.randint(0, self.dataset.size()))
        self.other_policy = imitation_policy(match_data)
        self.action_waiting_list = [None] * 4
        super().reset(match_data)
        obs = self.vec_data.get_obs(0, other=[])[0]
        self.action_waiting_list[0] = self.other_policy(obs)
        return {'obs': obs[0], 'mask': obs[1], 'gt_action': self.action_waiting_list[0]}

    def step(self, raw_action):
        if None in self.action_waiting_list:
            idx = self.action_waiting_list.index(None)
            obs = self.vec_data.get_obs(idx, other=[])[0]
            self.action_waiting_list[idx] = self.other_policy(obs)
            if obs[1].sum() > 1:
                return {'obs': obs[0], 'gt_action': self.action_waiting_list[idx], 'mask': obs[1]}, np.asarray(0.0), np.asarray(False), {}
            return self.step(None)

        real_action = [self.vec_data.realize(self.action_waiting_list[i]) for i in range(4)]
        try:
            self.roundInput(real_action)
            self.canHu = [-4] * 4
            self.roundOutput()
        except FinishError:
            obs = self.vec_data.get_obs(self.id, other=[])[0]
            return {'obs': obs[0], 'mask': obs[1], 'gt_action': self.action_waiting_list[self.id]}, np.asarray(0.0), np.array(True), {}
        finally:
            pass
        if self.wait_to_play is None:
            self.action_waiting_list = [None] * 4
            return self.step(None)
        obs = self.vec_data.get_obs(self.wait_to_play, other=[])[0]
        if self.wait_to_play == self.id:
            for i in range(4):
                self.action_waiting_list[i] = self.other_policy(self.vec_data.get_obs(i, other=[])[0])
        else:
            self.action_waiting_list[self.wait_to_play] = self.other_policy(obs)
        if obs[1].sum() > 1:
            return {'obs': obs[0], 'mask': obs[1], 'gt_action': self.action_waiting_list[self.wait_to_play]}, np.asarray(0.0), np.asarray(False), {}
        return self.step(None)
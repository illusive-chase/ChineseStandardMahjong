# -*- coding: utf-8 -*-
__all__ = ('SyncBuffer',)

import numpy as np
from threading import Lock
from collections import deque

class SyncBuffer:
    def __init__(self, store_traj, max_step=None, max_eps=None):
        self.maxlen = 1000 if store_traj else 1
        self.max_step = max_step if store_traj else None
        self.max_eps = max_eps if store_traj else None
        assert (not store_traj) or (max_step is not None) or (max_eps is not None)
        self.total_step = 0
        self.register_lock = Lock()
        self.reset()

    def reset(self):
        self.state = []
        self.mask = []
        self.action = []
        self.status = []
        self.reward = []
        self.open_idxs = []

    def register(self, init_state):
        self.register_lock.acquire()
        i = 0
        while i < len(self.open_idxs):
            id = self.open_idxs[i]
            if self.reward[id] is not None:
                self.total_step += len(self.state[id]) - 1
                self.open_idxs.pop(i)
            else:
                i += 1
        if (self.max_step is not None and self.total_step >= self.max_step) or (self.max_eps is not None and self.size() >= self.max_eps):
            self.register_lock.release()
            return None
        id = len(self.status)
        self.open_idxs.append(id)
        self.state.append(deque([init_state[0]], maxlen=self.maxlen))
        self.mask.append(deque([init_state[1]], maxlen=self.maxlen))
        self.action.append(deque())
        self.reward.append(None)
        self.status.append(0)
        self.register_lock.release()
        return id

    def size(self):
        return len(self.status)

    def open_slots(self):
        return self.open_idxs

    def get_action(self, id):
        if self.status[id] != 2:
            return None
        return self.action[id][-1]

    def get_state(self, id):
        if id >= len(self.status) or self.status[id] != 0:
            return None, None
        self.status[id] = 1
        return self.state[id][-1], self.mask[id][-1]

    def push_state(self, id, state):
        if state is None:
            self.state[id].append(None)
            self.mask[id].append(None)
            self.action[id].append(-1)
        else:
            self.state[id].append(state[0])
            self.mask[id].append(state[1])
            self.status[id] = 0

    def push_action(self, id, action):
        self.action[id].append(action)
        self.status[id] = 2

    def push_final_state(self, id, state, rew):
        self.state[id].append(state[0])
        self.mask[id].append(state[1])
        self.reward[id] = rew
        self.status[id] = 3






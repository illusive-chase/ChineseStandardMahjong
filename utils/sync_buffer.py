# -*- coding: utf-8 -*-
__all__ = ('SyncBuffer',)

import numpy as np
from threading import Lock

class SyncBuffer:
    def __init__(self, store_traj, max_step=None, max_eps=None, max_len=None):
        self.max_step = max_step if store_traj else None
        self.max_eps = max_eps if store_traj else None
        self.max_len = max_len
        self.store_traj = store_traj
        assert (not store_traj) or (max_step is not None) or (max_eps is not None)
        self.register_lock = Lock()
        self.reset()
        self.keep = []

    def reset(self):
        self.total_step = 0
        self.state = []
        self.mask = []
        self.action = []
        self.status = []
        self.reward = []
        self.open_idxs = []
        if self.store_traj:
            self.traj = []

    def register(self, init_state):
        self.register_lock.acquire()
        i = 0
        while i < len(self.open_idxs):
            id = self.open_idxs[i]
            if self.reward[id] is not None:
                if self.store_traj:
                    self.total_step += len(self.traj[id][0]) - 1
                self.open_idxs.pop(i)
            else:
                i += 1
        if (self.max_step is not None and self.total_step >= self.max_step) or (self.max_eps is not None and self.size() >= self.max_eps):
            self.register_lock.release()
            return None
        id = len(self.status)
        self.open_idxs.append(id)
        self.state.append(np.copy(init_state[0]))
        self.mask.append(np.copy(init_state[1]))
        self.action.append(None)
        self.reward.append(None)
        self.status.append(0)
        if self.store_traj:
            self.traj.append(([], [], []))
        self.register_lock.release()
        return id

    def size(self):
        return len(self.status)

    def open_slots(self):
        return self.open_idxs

    def is_joined(self):
        if (self.max_step is not None and self.total_step >= self.max_step) or (self.max_eps is not None and self.size() >= self.max_eps):
            if self.open_idxs == []:
                return True
        return False

    def get_action(self, id):
        if self.status[id] != 2:
            return None
        return self.action[id]

    def get_state(self, id):
        if id >= len(self.status) or self.status[id] != 0:
            return None, None
        self.status[id] = 1
        return self.state[id], self.mask[id]

    def push_state(self, id, state):
        if state is None:
            self.state[id] = None
            self.mask[id] = None
            self.action[id] = -1
        else:
            if self.store_traj and self.mask[id] is not None and self.mask[id].sum() > 1:
                self.traj[id][0].append(self.state[id])
                self.traj[id][1].append(self.mask[id])
                self.traj[id][2].append(self.action[id])
            self.state[id] = np.copy(state[0])
            self.mask[id] = np.copy(state[1])
            self.status[id] = 0
            

    def push_action(self, id, action):
        self.action[id] = np.copy(action)
        self.status[id] = 2

    def push_final_state(self, id, state, rew):
        if self.store_traj and self.mask[id] is not None and self.mask[id].sum() > 1:
            self.traj[id][0].append(self.state[id])
            self.traj[id][1].append(self.mask[id])
            self.traj[id][2].append(self.action[id])
        self.state[id] = np.copy(state[0])
        self.mask[id] = np.copy(state[1])
        self.reward[id] = np.copy(rew)
        self.status[id] = 3

    def unfinished_index(self):
        return np.array([], int)

    def sample(self, batch_size):
        # only for on-policy
        assert batch_size == 0
        self.traj = self.keep + self.traj
        np.random.shuffle(self.traj)

        if batch_size == 0:
            maxlen = np.sum([len(traj[0]) for traj in self.traj])
        else:
            maxlen = 0
            for traj in self.traj:
                maxlen += len(traj[0])
                if maxlen > batch_size:
                    break
        state_batch = np.zeros((maxlen, 161, 4, 9), dtype=np.bool)
        next_state_batch = np.zeros((maxlen, 161, 4, 9), dtype=np.bool)
        mask_batch = np.zeros((maxlen, 235), dtype=np.bool)
        action_batch = np.zeros((maxlen), dtype=np.uint8)
        done_batch = np.zeros((maxlen,), dtype=np.bool)
        reward_batch = np.zeros((maxlen,))

        idx = 0
        for reward, traj in zip(self.reward, self.traj):
            max_step = len(traj[0])
            if max_step == 0:
                continue
            state_batch[idx:idx+max_step] = np.stack(traj[0]).reshape(-1, 161, 4, 9)
            next_state_batch[idx:idx+max_step-1] = state_batch[idx+1:idx+max_step]
            mask_batch[idx:idx+max_step] = np.stack(traj[1])
            action_batch[idx:idx+max_step] = np.stack(traj[2])
            done_batch[idx+max_step-1] = True
            reward_batch[idx+max_step-1] = reward
            idx += max_step
            if idx >= maxlen:
                break

        self.done = done_batch
        indices = np.arange(maxlen)
        if self.max_len is not None:
            self.keep = self.traj[:max_len]

        assert mask_batch.any(1).all()

        return {
            'obs': state_batch,
            'obs_next': next_state_batch,
            'mask': mask_batch,
            'act': action_batch,
            'done': done_batch,
            'rew': reward_batch
        }, indices





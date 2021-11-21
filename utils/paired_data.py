# -*- coding: utf-8 -*-

__all__ = ('PairedDataset',)

import numpy as np
from numba import njit

# item: 1 + 34 + 14 + 4 x (4 + 28) + 39 = 216

@njit
def encode(obs, action, mask, dest):
    dest[0] = action
    dest[1:35] = obs[0:4, :34].sum(0)
    idx = 0
    flat = obs[4:8, :34].sum(0)
    for t in flat.nonzero()[0]:
        dest[35+idx:35+idx+flat[t]] = t + 1
        idx += flat[t]
    assert idx <= 14
    for i in range(4):
        idx = 0
        flat = obs[8+i*4:12+i*4, :34].sum(0)
        for t in flat.nonzero()[0]:
            dest[49+i*32+idx:49+i*32+idx+flat[t]] = t + 1
            idx += flat[t]
        for t in obs[24 + i, :34].nonzero()[0]:
            dest[49 + i * 32 + idx] = 64 + t + 1
            idx += 1
        for t in obs[28 + i, :34].nonzero()[0]:
            dest[49 + i * 32 + idx] = 128 + t + 1
            idx += 1
        if i == 0:
            for t in obs[32, :34].nonzero()[0]:
                dest[49 + i * 32 + idx] = 192 + t + 1
                idx += 1
        assert idx <= 4
        idx = 4
        for j in range(28):
            history = obs[33 + 28 * i + j, :34].nonzero()[0]
            assert history.shape[0] <= 1
            if history.shape[0] == 1:
                dest[49 + i * 32 + idx + j] = history[0] + 1
    flat = mask.nonzero()[0]
    assert flat.shape[0] > 1
    assert idx <= 177
    idx = 177
    for i in flat:
        dest[idx] = i + 1
        idx += 1
    assert idx <= 216
    

@njit
def decode(src, obs, act, mask):
    act[:] = src[0]
    flat = np.zeros(34, dtype=np.uint8)
    for i in range(1, 4):
        obs[0:i, :34] |= src[1:35] == i
    for i in range(35, 49):
        if src[i] > 0:
            flat[src[i] - 1] += 1
    for i in range(1, 4):
        obs[4:4+i, :34] |= flat == i
    for i in range(4):
        flat[:] = 0
        for j in range(4):
            pack = src[49 + i * 32 + j]
            if pack == 0:
                continue
            pack_type = pack // 64
            t = (pack % 64) - 1
            if pack_type == 0:
                flat[t] += 1
            elif pack_type == 1:
                obs[24 + i, t] = True
            elif pack_type == 2:
                obs[28 + i, t] = True
            else:
                assert pack_type == 3 and i == 0
                obs[32, t] = True
        for j in range(1, 4):
            obs[4:4+j, :34] |= flat == i
        for j in range(28):
            history = src[49 + i * 32 + 4 + j]
            if history > 0:
                obs[33 + 28 * i + j] = history - 1
    for i in range(177, 216):
        if src[i] > 0:
            mask[src[i]] = True


class PairedDataset:

    def __init__(self):
        self.page_size = 16
        self.item_size = 216
        self.reset()

    def add(self, obs, act, mask):
        self.full = None
        if self.size % self.page_size == 0:
            self.pages.append(np.zeros((self.page_size, self.item_size), dtype=np.uint8))
        encode(obs, act, mask, self.pages[-1][self.size % self.page_size])
        self.size += 1

    def get(self, idx):
        assert idx < self.size
        page_idx = idx // self.page_size
        page = self.pages[page_idx]
        return page[idx % self.page_size]

    def dump(self, f):
        np.save(f, np.asarray(self.size))
        for page in self.pages:
            np.save(f, page)

    def load(self, f):
        self.size = int(np.load(f))
        while True:
            try:
                page = np.load(f)
            except EOFError:
                break
            self.pages.append(page)
        self.pages = np.random.shuffle(self.pages[:-1]) + self.pages[-1:]
        

    def reset(self):
        self.pages = []
        self.size = 0
        self.full = None

    def sample(self, batch_size):
        if batch_size == 0:
            if self.full is None:
                self.full = self.sample(self.size)
            return self.full
        obs = np.zeros((batch_size, 145, 36), dtype=np.bool)
        mask = np.zeros((batch_size, 235), dtype=np.bool)
        act = np.zeros((batch_size,), dtype=np.uint8)
        indices = np.random.choice(self.size, batch_size)
        for indice in indices:
            item = self.get(indice)
            decode(item, obs, act, mask)
        return {'obs':obs, 'mask':mask, 'gt_action':act}, indices

    def split(self, val_ratio):
        train_set, val_set = PairedDataset(), PairedDataset()
        train_set.size = int(self.size / self.page_size * (1 - val_ratio)) * self.page_size
        val_set.size = self.size - train_set.size
        train_set.pages = self.pages[:train_set.size // self.page_size]
        val_set.pages = self.pages[train_set.size // self.page_size:]
        
        

        

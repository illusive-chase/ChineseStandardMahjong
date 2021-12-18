# -*- coding: utf-8 -*-

__all__ = ('PairedDataset',)

import numpy as np
from numba import njit
import tqdm
from utils.vec_data import VecData

# item: 1 + 34 + 14 + 4 x (4 + 28) + 39 = 216


@njit
def encode(obs, action, mask, rew, dest):
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
    assert idx <= 215
    dest[215] = rew
    

@njit
def decode(src, obs, act, mask, rew):
    act[:] = src[0]
    flat = np.zeros(34, dtype=np.uint8)
    for i in range(1, 5):
        obs[0:i, :34] |= src[1:35] == i
    for i in range(35, 49):
        if src[i] > 0:
            flat[src[i] - 1] += 1
    for i in range(1, 5):
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
        for j in range(1, 5):
            obs[8+i*4:8+i*4+j, :34] |= flat == j
        for j in range(28):
            history = src[49 + i * 32 + 4 + j]
            if history > 0:
                obs[33 + 28 * i + j, history - 1] = 1
    for i in range(177, 215):
        if src[i] > 0:
            mask[src[i] - 1] = True
    rew[:] = src[215]





class PairedDataset:

    action_augment_table = None
    tile_augment_table = None

    def __init__(self, augmentation=1):
        self.page_size = 16
        self.item_size = 216
        self.augmentation = augmentation
        assert 1 <= augmentation <= 12
        self.reset()

    def add(self, obs, act, mask, rew):
        self.full = None
        if self.actual_size % self.page_size == 0:
            self.pages.append(np.zeros((self.page_size, self.item_size), dtype=np.uint8))
        encode(obs, act, mask, rew, self.pages[-1][self.actual_size % self.page_size])
        self.actual_size += 1

    def size(self):
        return self.actual_size * self.augmentation

    def get(self, idx):
        assert idx < self.actual_size * self.augmentation
        aug_type = idx % self.augmentation
        idx = idx // self.augmentation
        page_idx = idx // self.page_size
        page = self.pages[page_idx]
        return self.augment(page[idx % self.page_size], aug_type) if aug_type > 0 else page[idx % self.page_size]

    def dump(self, f):
        np.save(f, np.asarray(self.actual_size))
        for page in self.pages:
            np.save(f, page)

    def load(self, f, max_size=2**31-1):
        self.actual_size = min(max_size // self.augmentation, int(np.load(f)))
        with tqdm.trange((self.actual_size + self.page_size - 1) // self.page_size, desc=f"Loading PairedDataset", dynamic_ncols=True, ascii=True) as t:
            for _ in t:
                page = np.load(f)
                self.pages.append(page)

    def loads(self, fs):
        for f in fs:
            total_size = int(np.load(f)) // self.page_size
            self.actual_size += total_size * self.page_size
            for _ in range(total_size):
                page = np.load(f)
                self.pages.append(page)
        
    @classmethod
    def _compile(self):
        obs = np.zeros((2, 145, 36), dtype=np.bool)
        mask = np.zeros((2, 235), dtype=np.bool)
        act = np.zeros((2, 1), dtype=np.uint8)
        rew = np.zeros((2, 1), dtype=np.uint8)
        item = np.zeros(216, dtype=np.uint8)
        try:
            encode(obs[0], act[0, 0], mask[0], rew[0, 0], item)
        except AssertionError:
            pass
        try:
            decode(item, obs[0], act[0], mask[0], rew[0])
        except AssertionError:
            pass
        self.action_augment_table = np.zeros((12, 235 + 1), dtype=np.uint8)
        self.tile_augment_table = np.zeros((12, 34 + 1), dtype=np.uint8)
        self.action_augment_table[:, 1:] = VecData.action_augment_table + 1
        self.tile_augment_table[:, 1:] = VecData.tile_augment_table + 1

    @classmethod
    def augment(self, item, aug_type):
        ret = np.empty_like(item)
        ret[0] = self.action_augment_table[aug_type, item[0] + 1] - 1
        tile_count_idx = self.tile_augment_table[aug_type, 1:] - 1
        ret[1:35] = ret[1:35][tile_count_idx]
        ret[35:49] = self.tile_augment_table[aug_type, item[35:49]]
        ret[49:177] = (item[49:177] // 64) * 64 + self.tile_augment_table[aug_type, item[49:177] % 64]
        ret[177:215] = self.action_augment_table[aug_type, item[177:215]]
        ret[215] = item[215]
        if aug_type == 0:
            assert (ret == item).all()
        return ret
        

    def reset(self):
        self.pages = []
        self.actual_size = 0
        self.full = None
        self._indices = None

    def sample(self, batch_size, shuffle=True):
        if batch_size == 0:
            if self.full is None:
                self.full = self.sample(self.actual_size * self.augmentation, shuffle=shuffle)
            return self.full
        obs = np.zeros((batch_size, 145, 36), dtype=np.bool)
        mask = np.zeros((batch_size, 235), dtype=np.bool)
        act = np.zeros((batch_size, 1), dtype=np.uint8)
        rew = np.zeros((batch_size, 1), dtype=np.uint8)
        if self._indices is None or self._indices.shape[0] < batch_size:
            self._indices = np.arange(self.actual_size * self.augmentation)
            if shuffle:
                np.random.shuffle(self._indices)
        indices = self._indices[:batch_size]
        self._indices = self._indices[batch_size:]
        for idx, indice in enumerate(indices):
            item = self.get(indice)
            decode(item, obs[idx], act[idx], mask[idx], rew[idx])
        return {'obs':obs, 'mask':mask, 'gt_action':act[:, 0], 'rew':rew[:, 0].astype(np.int8)}, indices

    def split(self, val_ratio):
        np.random.shuffle(self.pages[:-1])
        train_set, val_set = PairedDataset(self.augmentation), PairedDataset(self.augmentation)
        train_set.actual_size = int(self.actual_size / self.page_size * (1 - val_ratio)) * self.page_size
        val_set.actual_size = self.actual_size - train_set.actual_size
        train_set.pages = self.pages[:train_set.actual_size // self.page_size]
        val_set.pages = self.pages[train_set.actual_size // self.page_size:]
        return train_set, val_set
        


PairedDataset._compile()

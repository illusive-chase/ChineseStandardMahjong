# -*- coding: utf-8 -*-

__all__ = ('sample_by_mask',)

import numpy as np

random_pool = np.zeros((1, 0))
pool_idx = 0


def rand(shape):
    global random_pool
    global pool_idx
    if shape[1] != random_pool.shape[1]:
        random_pool = np.random.rand(shape[0], shape[1])
        pool_idx = 0
    elif pool_idx + shape[0] >= random_pool.shape[0]:
        new_capacity = min(max(shape[0], random_pool.shape[0] * 2), shape[0] * 128)
        random_pool = np.random.rand(new_capacity, shape[1])
        pool_idx = 0
    pool_idx += shape[0]
    return random_pool[pool_idx-shape[0]:pool_idx]

def sample_by_mask(mask):
    if mask.ndim == 1:
        logits = rand((1, *mask.shape))[0]
    else:
        logits = rand(mask.shape)
    logits[~mask] = -1.
    return np.argmax(logits, axis=-1)
        


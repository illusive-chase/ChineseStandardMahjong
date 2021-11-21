# -*- coding: utf-8 -*-
__all__ = ('imitation_trainer',)

import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
import tqdm

from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import BasePolicy
from tianshou.trainer import gather_info, test_episode
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config


def imitation_trainer(
    policy: BasePolicy,
    train_collector: Collector,
    val_collector: Collector,
    max_epoch: int,
    update_per_epoch: int,
    episode_per_train: int,
    batch_size: int,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logger: BaseLogger = LazyLogger(),
) -> Dict[str, Union[float, str]]:

    start_epoch, gradient_step = 0, 0
    if resume_from_log:
        start_epoch, _, gradient_step = logger.restore_data()
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()

    train_result = test_episode(
        policy, train_collector, test_fn, start_epoch, episode_per_train, logger,
        gradient_step, reward_metric
    )
    val_result = test_episode(
        policy, val_collector, test_fn, start_epoch, 10, LazyLogger(),
        gradient_step, reward_metric
    )

    policy.eval()
    with torch.no_grad():
        losses = policy.update(0, val_collector.buffer, val=True)
        for k in losses.keys():
            stat[k].add(losses[k])
            losses[k] = stat[k].get()
        print(losses)

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        policy.train()
        with tqdm.trange(update_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config) as t:
            for _ in t:
                gradient_step += 1
                losses = policy.update(batch_size, train_collector.buffer)
                data = {"step": str(gradient_step)}
                for k in losses.keys():
                    stat[k].add(losses[k])
                    losses[k] = stat[k].get()
                    data[k] = f"{losses[k]:.5f}"
                logger.log_update_data(losses, gradient_step)
                t.set_postfix(**data)
        # train
        train_collector.buffer.reset()
        train_result = test_episode(
            policy, train_collector, test_fn, epoch, episode_per_train, logger,
            gradient_step, reward_metric
        )
        # val
        policy.eval()
        with torch.no_grad():
            losses = policy.update(0, val_collector.buffer, val=True)
            for k in losses.keys():
                stat[k].add(losses[k])
                losses[k] = stat[k].get()
            print(losses)
        if save_fn:
            save_fn(policy)
        logger.save_data(epoch, 0, gradient_step, save_checkpoint_fn)
    return gather_info(start_time, None, test_collector, None, None)
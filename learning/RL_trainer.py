# -*- coding: utf-8 -*-
__all__ = ('onpolicy_trainer',)

import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Union, Any

import numpy as np
import torch

from env.multirunner import MultiRunner

from tianshou.policy import BasePolicy
from tianshou.utils import BaseLogger, LazyLogger, MovAvg


def onpolicy_trainer(
    policy: Callable[[Any], Any],
    other_policy: Callable[[Any], Any],
    max_epoch: int,
    collect_per_epoch: int,
    test_eps_per_epoch: int,
    step_per_collect: int,
    repeat_per_collect: int,
    batch_size: int,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    logger: BaseLogger = LazyLogger(),
    verbose: bool = False,
):

    start_epoch, gradient_step = 0, 0
    if resume_from_log:
        start_epoch, _, gradient_step = logger.restore_data()
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()

    # 2,1,400: sps 250, eps 18
    train_runner = MultiRunner(policy, other_policy, 2, 1, 300, 12000, max_step=step_per_collect, max_eps=None)
    val_runner = MultiRunner(policy, other_policy, 2, 1, 300, 12000, max_step=None, max_eps=test_eps_per_epoch)

    policy.eval()
    other_policy.eval()
    val_runner.collect(eval=True, verbose=verbose)
    stat['val-rew'].add(val_runner.mean_reward())
    val_rew = stat['val-rew'].get()
    best_val_rew = val_rew
    print({"val-rew" : val_rew})

    log_data = {"update/val-rew": val_rew}
    logger.write("update/gradient_step", gradient_step, log_data)

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        for collect in range(collect_per_epoch):
            policy.train()
            train_runner.collect(eval=False, verbose=verbose)
            gradient_step += 1
            losses = policy.update(
                0,
                train_runner.get_buffer(),
                batch_size=step_per_collect,
                repeat=repeat_per_collect
            )
            for k in losses.keys():
                stat[k].add(losses[k])
                losses[k] = stat[k].get()
            print(losses)
            logger.log_update_data(losses, gradient_step)

        # val
        policy.eval()
        val_runner.collect(eval=True, verbose=verbose)
        stat['val-rew'].add(val_runner.mean_reward())
        val_rew = stat['val-rew'].get()
        print({"val-rew" : val_rew})

        if best_val_rew > val_rew:
            best_losses = val_rew
            if save_fn:
                save_fn(policy)
        log_data = {f"update/{k}": v for k, v in losses.items()}
        log_data["val-rew"] = val_rew
        logger.write("update/gradient_step", gradient_step, log_data)
        logger.save_data(epoch, 0, gradient_step, save_checkpoint_fn)
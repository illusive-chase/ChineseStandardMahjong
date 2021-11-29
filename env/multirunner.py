# -*- coding: utf-8 -*-
__all__ = ('MultiRunner', 'WrappedTrainRunner', 'WrappedEvalRunner')

import threading
import torch
import numpy as np
from time import sleep, time
from utils.sync_buffer import SyncBuffer
from env.runner import Runner, FinishError


sleep_slot_time = 0.0005


class WrappedTrainRunner(Runner):
    def __init__(self):
        super().__init__(other_policy=None, verbose=False, seed=None, eval=False)

    def reset(self):
        state = super().reset()
        other_state = self.fixData
        return state, other_state


    def step(self, actions):
        if self.wait_to_play is None:
            fix_action = actions[1]
            action = [actions[0] if i == self.id else fix_action[self.other.index(i)] for i in range(4)]
            real_action = [self.vec_data.realize(action[i]) for i in range(4)]
        else:
            action = actions[0] if self.wait_to_play == self.id else actions[1]
            real_action = ['PASS'] * 4
            real_action[self.wait_to_play] = self.vec_data.realize(action)

        while True:
            try:
                self.roundInput(real_action)
                self.canHu = [-4] * 4
                self.roundOutput()
            except FinishError:
                return self.vec_data.get_obs(self.id, self.other), self.rew[self.id], np.array(True), {}
            finally:
                pass
            if self.wait_to_play is None:
                return self.vec_data.get_obs(self.id, self.other), 0.0, np.asarray(False), {}
            state, _ = self.vec_data.get_obs(self.wait_to_play, other=[])
            if self.wait_to_play != self.id:
                return (None, state), 0.0, np.asarray(False), {}
            return (state, None), 0.0, np.asarray(False), {}

class WrappedEvalRunner(Runner):
    def __init__(self):
        super().__init__(other_policy=None, verbose=False, seed=None, eval=True)
        self.rews = np.zeros(4)
        self.game_id = 0
        self.scores = []

    def reset(self):
        state = super().reset()
        other_state = self.fixData
        return state, other_state


    def step(self, actions):
        if self.wait_to_play is None:
            fix_action = actions[1]
            action = [actions[0] if i == self.id else fix_action[self.other.index(i)] for i in range(4)]
            real_action = [self.vec_data.realize(action[i]) for i in range(4)]
        else:
            action = actions[0] if self.wait_to_play == self.id else actions[1]
            real_action = ['PASS'] * 4
            real_action[self.wait_to_play] = self.vec_data.realize(action)

        while True:
            try:
                self.roundInput(real_action)
                self.canHu = [-4] * 4
                self.roundOutput()
            except FinishError:
                for i in range(4):
                    self.rews[i] += self.rew[(self.id + i) % 4]
                self.game_id += 1
                if self.game_id == 4:
                    to_assign = np.asarray([4, 3, 2, 1, 0], dtype=np.float64)
                    to_assign = to_assign[(self.rews > self.rews[0]).sum():-(self.rews < self.rews[0]).sum()-1].mean() - 2.5
                    self.scores.append(to_assign)
                    self.game_id = 0
                    self.rews[:] = 0
                    self.tileWallDummy = None
                if len(self.scores) < 1:
                    return self.reset(), 0.0, np.asarray(False), {}
                return self.vec_data.get_obs(self.id, self.other), sum(self.scores), np.asarray(True), {}
            finally:
                pass
            if self.wait_to_play is None:
                return self.vec_data.get_obs(self.id, self.other), 0.0, np.asarray(False), {}
            state, _ = self.vec_data.get_obs(self.wait_to_play, other=[])
            if self.wait_to_play != self.id:
                return (None, state), 0.0, np.asarray(False), {}
            return (state, None), 0.0, np.asarray(False), {}




class EnvWorker(threading.Thread):
    def __init__(self, buffer, other_buffer, max_env_num, eval):
        super().__init__()
        self.buffer = buffer
        self.other_buffer = other_buffer
        self.max_env_num = max_env_num
        self.env_constructor = WrappedTrainRunner if not eval else WrappedEvalRunner
        self.envs = []
        self.workload = (0, 0)

    def run(self):
        while True:
            dones = []
            worked = 0
            for idx1, idx2, env in self.envs:
                action = self.buffer.get_action(idx1)
                other_action = self.other_buffer.get_action(idx2)
                if action is None or other_action is None:
                    dones.append(False)
                    continue
                worked += 1
                obs, rew, done, info = env.step((action, other_action))
                state, other_state = obs
                dones.append(done)
                if done:
                    self.buffer.push_final_state(idx1, state, rew)
                    self.other_buffer.push_final_state(idx2, other_state, None)
                else:
                    self.buffer.push_state(idx1, state)
                    self.other_buffer.push_state(idx2, other_state)
            self.envs = [env for env, done in zip(self.envs, dones) if not done]
            if worked != 0:
                self.workload = (self.workload[0] + worked, self.workload[1] + 1)
                continue
            if len(self.envs) < self.max_env_num:
                env = self.env_constructor()
                init_state, other_init_state = env.reset()
                idx1 = self.buffer.register(init_state)
                if idx1 is None:
                    if self.envs == []:
                        return 0
                else:
                    idx2 = self.other_buffer.register(other_init_state)
                    self.envs.append((idx1, idx2, env))
            self.workload = (self.workload[0], self.workload[1] + 1)
            sleep(sleep_slot_time)

        return 0



class TorchWorker(threading.Thread):
    def __init__(self, policy, buffer, torch_lock, max_batch_size):
        super().__init__()
        self.policy = policy
        self.buffer = buffer
        self.max_batch_size = max_batch_size
        self.to_close = False
        self.workload = (0, 0)
        self.torch_lock = torch_lock

    def join(self):
        self.to_close = True
        super().join()
    
    def run(self):
        while not self.to_close:
            state_batch = []
            mask_batch = []
            idx_batch = []
            action_idx = 0
            
            self.torch_lock.acquire()
            for i in self.buffer.open_slots():
                state, mask = self.buffer.get_state(i)
                if state is not None:
                    mask = mask.reshape(-1, 235)
                    state = state.reshape(-1, 145, 4, 9)
                    state_batch.append(state)
                    mask_batch.append(mask)
                    idx_batch.append((i, action_idx, action_idx + mask.shape[0]))
                    action_idx += mask.shape[0]
                    if action_idx >= self.max_batch_size:
                        break
            self.torch_lock.release()
            if state_batch != []:
                state_batch = np.vstack(state_batch)
                mask_batch = np.vstack(mask_batch)
                with torch.no_grad():
                    action = self.policy((state_batch, mask_batch))
                for idx, action_begin, action_end in idx_batch:
                    if action_end - action_begin == 1:
                        self.buffer.push_action(idx, action[action_begin])
                    else:
                        self.buffer.push_action(idx, action[action_begin:action_end])

            self.workload = (self.workload[0] + action_idx, self.workload[1] + 1)
            sleep(sleep_slot_time)
        return 0

    

class MultiRunner:
    def __init__(self, policy, other_policy, n_torch_parallel, n_env_parallel, max_env_num, max_batch_size, max_step=None, max_eps=None):
        self.policy = policy
        self.other_policy = other_policy
        self.n_env_parallel = n_env_parallel
        self.n_torch_parallel = n_torch_parallel
        self.max_batch_size = max_batch_size
        self.max_env_num = max_env_num
        self.buffer = SyncBuffer(True, max_step=max_step, max_eps=max_eps)
        self.other_buffer = SyncBuffer(False)
        self.torch_lock = threading.Lock()
        self.other_torch_lock = threading.Lock()

        assert self.n_torch_parallel > 1


    def collect(self, eval=False, verbose=False):
        self.buffer.reset()
        self.other_buffer.reset()

        self.torch_workers = [
            TorchWorker(self.policy, self.buffer, self.torch_lock, self.max_batch_size)
            for _ in range(self.n_torch_parallel // 2)
        ] + [
            TorchWorker(self.other_policy, self.other_buffer, self.other_torch_lock, self.max_batch_size // 3)
            for _ in range((self.n_torch_parallel + 1) // 2)
        ]
        self.env_workers = [EnvWorker(self.buffer, self.other_buffer, self.max_env_num, eval) for _ in range(self.n_env_parallel)]

        start = time()

        for thread in (self.torch_workers + self.env_workers):
            thread.start()
        
        count = threading.activeCount()
        while count > 1 + self.n_torch_parallel:
            if verbose:
                print('Running Threads:', count)
            sleep(1)
            count = threading.activeCount()
            if verbose:
                print('Batch Per Sec: ', end='')
                past_time = time() - start
                for torch_thread in self.torch_workers:
                    print('{:.1f} '.format(torch_thread.workload[0] / past_time), end='')
                print('\nEmpty GPU Fetch Per Sec: ', end='')
                for torch_thread in self.torch_workers:
                    print('{:.1f} '.format(torch_thread.workload[1] / past_time), end='')
                print('\nEnvs Parallel:', sum([len(env_thread.envs) for env_thread in self.env_workers]), end='')
                print('\nStep Per Sec: ', end='')
                for env_thread in self.env_workers:
                    print('{:.1f} '.format(env_thread.workload[0] / past_time), end='')
                print('\nEmpty CPU Fetch Per Sec: ', end='')
                for env_thread in self.env_workers:
                    print('{:.1f} '.format(env_thread.workload[1] / past_time), end='')
                print('\nEPS: {}/{:.2f}'.format(self.buffer.size(), self.buffer.size() / past_time))
                print('SPS: {}/{:.2f}'.format(self.buffer.total_step, self.buffer.total_step / past_time))
                print('')

        for thread in (self.torch_workers + self.env_workers):
            thread.join()

        use = time() - start
        print('Use: {:.2f}s'.format(use))
        print('EPS: {:.2f}'.format(self.buffer.size() / use))
        print('SPS: {:.2f}'.format(self.buffer.total_step / use))

    def mean_reward(self):
        return np.mean(self.buffer.reward)

        




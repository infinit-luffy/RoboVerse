
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import sys


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, obs_shape: dict, action_size: int, replay_dir: str):
        self._obs_shape = obs_shape
        self._action_size = action_size
        self._replay_dir = Path(replay_dir)
        if self._replay_dir.exists():
            shutil.rmtree(self._replay_dir)

        self._replay_dir.mkdir(parents=True, exist_ok=False)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, observation, action, reward, next_observation, done):
        for key in self._obs_shape.keys():
            if "rgb" in key:
                self._current_episode[f"observation_{key}"].append((observation[key] * 255.0).detach().cpu().numpy().astype(np.uint8))
                self._current_episode[f"next_observation_{key}"].append((next_observation[key] * 255.0).detach().cpu().numpy().astype(np.uint8))
            else:
                self._current_episode[f"observation_{key}"].append(observation[key].detach().cpu().numpy())
                self._current_episode[f"next_observation_{key}"].append(next_observation[key].detach().cpu().numpy())
        self._current_episode['action'].append(action.detach().cpu().numpy())
        self._current_episode['reward'].append(reward.unsqueeze(-1).detach().cpu().numpy())
        
        if done:
            episode = dict()
            for key in self._obs_shape.keys():
                if "rgb" in key:
                    episode[f"observation_{key}"] = np.array(self._current_episode[f"observation_{key}"], dtype=np.uint8)
                    episode[f"next_observation_{key}"] = np.array(self._current_episode[f"next_observation_{key}"], dtype=np.uint8)
                else:
                    episode[f"observation_{key}"] = np.array(self._current_episode[f"observation_{key}"])
                    episode[f"next_observation_{key}"] = np.array(self._current_episode[f"next_observation_{key}"])
            episode['action'] = np.array(self._current_episode['action'])
            episode['reward'] = np.array(self._current_episode['reward'])
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, obs_shape: dict, action_size: int, replay_dir: str, max_size: int, num_workers: int, nstep: int, discount: float,
                 fetch_every: int, save_snapshot: bool):
        self._obs_shape = obs_shape
        self._action_size = action_size
        self._replay_dir = Path(replay_dir)
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        print(f"Initializing ReplayBuffer with max_size={max_size}, num_workers={num_workers}, nstep={nstep}, discount={discount}, fetch_every={fetch_every}, save_snapshot={save_snapshot}")
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]
            
    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                continue
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1)
        obs = (
            episode[f'observation_{key}'][idx] 
            if "rgb" not in key 
            else episode[f'observation_{key}'][idx].astype(np.float32) / 255.0
            for key in self._obs_shape.keys()
        )
        next_obs = (
            episode[f'next_observation_{key}'][idx + self._nstep - 1] 
            if "rgb" not in key 
            else episode[f'next_observation_{key}'][idx + self._nstep - 1].astype(np.float32) / 255.0
            for key in self._obs_shape.keys()
        )
        action = episode['action'][idx]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['reward'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= self._discount
        return (*obs, *next_obs, action, reward, discount)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(obs_shape, action_size, replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(obs_shape,
                            action_size,
                            replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn,
                                         )
    return loader

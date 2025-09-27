from __future__ import annotations

import numpy as np
import torch
from loguru import logger as log
from rich.logging import RichHandler
from tensordict.tensordict import TensorDict
from torch import Tensor

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class ReplayBuffer:
    ## Not supported for multi-task setting
    def __init__(
        self,
        observation_shape: dict,
        action_size: int,
        task_embed_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        self.observation = {
            key: np.empty((self.capacity, self.num_envs, shape), dtype=np.float32)
            if "rgb" not in key
            else np.empty((self.capacity, self.num_envs, shape), dtype=np.uint8)
            for key, shape in observation_shape.items()
        }
        self.action = np.empty((self.capacity, self.num_envs, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.terminated = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.task_embed_size = task_embed_size
        if task_embed_size > 0:
            self.task = np.empty((self.capacity, self.num_envs, task_embed_size), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(
        self,
        observation: TensorDict,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        terminated: Tensor,
        task: Tensor | None,
    ):
        for key in self.observation.keys():
            if "rgb" in key:
                self.observation[key][self.buffer_index] = (observation[key].detach().cpu().numpy() * 255.0).astype(
                    np.uint8
                )
            else:
                self.observation[key][self.buffer_index] = observation[key].detach().cpu().numpy()
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.unsqueeze(-1).detach().cpu().numpy()
        self.done[self.buffer_index] = done.unsqueeze(-1).detach().cpu().numpy()
        self.terminated[self.buffer_index] = terminated.unsqueeze(-1).detach().cpu().numpy()
        if self.task_embed_size > 0 and task is not None:
            self.task[self.buffer_index] = task.detach().cpu().numpy()

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size) -> TensorDict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        batch_size_per_env = batch_size // self.num_envs
        last_filled_index = self.buffer_index - chunk_size + 1
        if last_filled_index >= 0 and self.full:
            sample_range = np.concatenate((
                np.arange(self.buffer_index, self.capacity),
                np.arange(0, last_filled_index),
            ))
        elif last_filled_index >= 0:
            sample_range = np.arange(0, last_filled_index)
        elif last_filled_index < 0 and self.full:
            last_filled_index = self.capacity + last_filled_index
            sample_range = np.arange(self.buffer_index, last_filled_index)
        else:
            raise AssertionError("Not enough data in the buffer")
        assert len(sample_range) > batch_size_per_env, "too short dataset or too long chunk_size"
        sample_index = np.random.choice(sample_range, size=(self.num_envs, batch_size_per_env), replace=True)
        chunk_length = np.arange(chunk_size).reshape(1, -1)

        sample_index = (sample_index + chunk_length) % self.capacity  # (num_envs, batch_size_per_env, chunk_size)
        env_index = np.arange(self.num_envs).reshape(-1, 1, 1)
        flattened_index = sample_index * self.num_envs + env_index[:, None]

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        observation = {
            key: torch.as_tensor(flatten(val)[flattened_index], device=self.device)
            .float()
            .reshape(-1, chunk_size, *val.shape[2:])
            .permute(1, 0, *range(2, val.ndim))
            if "rgb" not in key
            else (torch.as_tensor(flatten(val)[flattened_index], dtype=torch.float32, device=self.device) / 255.0)
            .reshape(-1, chunk_size, *val.shape[2:])
            .permute(1, 0, *range(2, val.ndim))
            for key, val in self.observation.items()
        }
        action = (
            torch.as_tensor(flatten(self.action)[flattened_index], device=self.device)
            .reshape(-1, chunk_size, self.action.shape[2])
            .permute(1, 0, *range(2, self.action.ndim))
        )
        reward = (
            torch.as_tensor(flatten(self.reward)[flattened_index], device=self.device)
            .reshape(-1, chunk_size, 1)
            .permute(1, 0, 2)
        )
        done = (
            torch.as_tensor(flatten(self.done)[flattened_index], device=self.device)
            .reshape(-1, chunk_size, 1)
            .permute(1, 0, 2)
        )
        terminated = (
            torch.as_tensor(flatten(self.terminated)[flattened_index], device=self.device)
            .reshape(-1, chunk_size, 1)
            .permute(1, 0, 2)
        )
        if self.task_embed_size > 0:
            task = (
                torch.as_tensor(flatten(self.task)[flattened_index], device=self.device)
                .reshape(-1, chunk_size, self.task_embed_size)
                .permute(1, 0, *range(2, self.task.ndim))[0]
            )
        else:
            task = None

        sample = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "terminated": terminated,
            "task": task,
        }
        return sample

    def mean_reward(self) -> float:
        if self.full:
            return float(self.reward.mean())
        else:
            return float(self.reward[: self.buffer_index].mean())


class ReplayBuffer_PyTorch:
    ## Not supported for multi-task setting
    def __init__(
        self,
        observation_shape: dict,
        action_size: int,
        task_embed_size: int,
        storage_device: str | torch.device = "cpu",
        device: str | torch.device = "cpu",
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.storage_device = storage_device
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        self.observation = TensorDict({
            key: torch.zeros((self.capacity, self.num_envs, *shape), dtype=torch.float32, device=storage_device)
            if "rgb" not in key
            else torch.zeros((self.capacity, self.num_envs, *shape), dtype=torch.uint8, device=storage_device)
            for key, shape in observation_shape.items()
        })
        self.action = torch.zeros(
            (self.capacity, self.num_envs, action_size), dtype=torch.float32, device=storage_device
        )
        self.reward = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32, device=storage_device)
        self.done = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32, device=storage_device)
        self.terminated = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32, device=storage_device)
        self.task_embed_size = task_embed_size
        if task_embed_size > 0:
            self.task = torch.zeros(
                (self.capacity, self.num_envs, task_embed_size), dtype=torch.float32, device=storage_device
            )

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(
        self,
        observation: TensorDict,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        terminated: Tensor,
        task: Tensor | None,
    ):
        for key in self.observation.keys():
            if "rgb" in key:
                self.observation[key][self.buffer_index] = (
                    (observation[key] * 255.0).detach().to(torch.uint8).to(self.storage_device)
                )
            else:
                self.observation[key][self.buffer_index] = observation[key].detach().to(self.storage_device)
        self.action[self.buffer_index] = action.detach().to(self.storage_device)
        self.reward[self.buffer_index] = reward.unsqueeze(-1).detach().to(self.storage_device)
        self.done[self.buffer_index] = done.unsqueeze(-1).detach().to(self.storage_device)
        self.terminated[self.buffer_index] = terminated.unsqueeze(-1).detach().to(self.storage_device)
        if self.task_embed_size > 0 and task is not None:
            self.task[self.buffer_index] = task.detach().to(self.storage_device)

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size) -> TensorDict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        batch_size_per_env = batch_size // self.num_envs
        last_filled_index = self.buffer_index - chunk_size + 1
        if last_filled_index >= 0 and self.full:
            sample_range = np.concatenate((
                np.arange(self.buffer_index, self.capacity),
                np.arange(0, last_filled_index),
            ))
        elif last_filled_index >= 0:
            sample_range = np.arange(0, last_filled_index)
        elif last_filled_index < 0 and self.full:
            last_filled_index = self.capacity + last_filled_index
            sample_range = np.arange(self.buffer_index, last_filled_index)
        else:
            raise AssertionError("Not enough data in the buffer")
        assert len(sample_range) > batch_size_per_env, "too short dataset or too long chunk_size"
        sample_index = np.random.choice(sample_range, size=(self.num_envs, batch_size_per_env), replace=True)
        sample_index = sample_index[:, :, None]  # (num_envs, batch_size_per_env, 1)
        chunk_length = np.arange(chunk_size).reshape(1, -1)

        sample_index = (sample_index + chunk_length) % self.capacity  # (num_envs, batch_size_per_env, chunk_size)
        env_index = np.arange(self.num_envs).reshape(-1, 1, 1)
        flattened_index = sample_index * self.num_envs + env_index

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        observation = TensorDict(
            {
                key: flatten(val)[flattened_index]
                .to(self.device)
                .reshape(-1, chunk_size, *val.shape[2:])
                .permute(1, 0, *range(2, val.ndim))
                if "rgb" not in key
                else (flatten(val)[flattened_index].to(device=self.device).float() / 255.0)
                .reshape(-1, chunk_size, *val.shape[2:])
                .permute(1, 0, *range(2, val.ndim))
                for key, val in self.observation.items()
            },
            batch_size=[chunk_size, batch_size],
        )
        action = (
            flatten(self.action)[flattened_index]
            .to(self.device)
            .reshape(-1, chunk_size, self.action.shape[2])
            .permute(1, 0, 2)
        )
        reward = flatten(self.reward)[flattened_index].to(self.device).reshape(-1, chunk_size, 1).permute(1, 0, 2)
        done = flatten(self.done)[flattened_index].to(self.device).reshape(-1, chunk_size, 1).permute(1, 0, 2)
        terminated = (
            flatten(self.terminated)[flattened_index].to(self.device).reshape(-1, chunk_size, 1).permute(1, 0, 2)
        )
        if self.task_embed_size > 0:
            task = (
                flatten(self.task)[flattened_index]
                .to(self.device)
                .reshape(-1, chunk_size, self.task_embed_size)
                .permute(1, 0, 2)[0]
            )
        else:
            task = None

        sample = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "terminated": terminated,
            "task": task,
        }
        return sample

    def mean_reward(self) -> float:
        if self.full:
            return float(self.reward.mean())
        else:
            return float(self.reward[: self.buffer_index].mean())

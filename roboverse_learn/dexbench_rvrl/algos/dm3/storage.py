from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import numpy as np
import torch
from loguru import logger as log
from rich.logging import RichHandler
from torch import Tensor

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: int,
        action_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        self.observation = np.empty((self.capacity, self.num_envs, observation_shape), dtype=np.float32)
        self.next_observation = np.empty((self.capacity, self.num_envs, observation_shape), dtype=np.float32)
        self.action = np.empty((self.capacity, self.num_envs, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.terminated = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.step_idx = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(
        self,
        observation: Tensor,
        action: Tensor,
        reward: Tensor,
        next_observation: Tensor,
        done: Tensor,
        terminated: Tensor,
        step: Tensor,
    ):
        self.observation[self.buffer_index] = observation.detach().cpu().numpy()
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.unsqueeze(-1).detach().cpu().numpy()
        self.next_observation[self.buffer_index] = next_observation.detach().cpu().numpy()
        self.done[self.buffer_index] = done.unsqueeze(-1).detach().cpu().numpy()
        self.terminated[self.buffer_index] = terminated.unsqueeze(-1).detach().cpu().numpy()
        self.step_idx[self.buffer_index] = step.unsqueeze(-1).detach().cpu().numpy()

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size) -> dict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        last_filled_index = self.buffer_index - chunk_size + 1
        assert self.full or (last_filled_index > batch_size), "too short dataset or too long chunk_size"
        sample_index = np.random.randint(0, self.capacity if self.full else last_filled_index, batch_size).reshape(
            -1, 1
        )
        chunk_length = np.arange(chunk_size).reshape(1, -1)

        sample_index = (sample_index + chunk_length) % self.capacity
        env_index = np.random.randint(0, self.num_envs, batch_size)
        flattened_index = sample_index * self.num_envs + env_index[:, None]

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        observation = torch.as_tensor(flatten(self.observation)[flattened_index], device=self.device).float()
        next_observation = torch.as_tensor(flatten(self.next_observation)[flattened_index], device=self.device).float()
        action = torch.as_tensor(flatten(self.action)[flattened_index], device=self.device)
        reward = torch.as_tensor(flatten(self.reward)[flattened_index], device=self.device)
        done = torch.as_tensor(flatten(self.done)[flattened_index], device=self.device)
        terminated = torch.as_tensor(flatten(self.terminated)[flattened_index], device=self.device)
        if_first = torch.as_tensor(flatten(self.step_idx)[flattened_index], device=self.device) == 1

        sample = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
            "terminated": terminated,
            "if_first": if_first,
        }
        return sample

    def mean_reward(self) -> float:
        if self.full:
            return float(self.reward.mean())
        else:
            return float(self.reward[: self.buffer_index].mean())


class ReplayBuffer_Pytorch:
    def __init__(
        self,
        observation_shape: int,
        action_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        self.observation = torch.zeros((self.capacity, self.num_envs, observation_shape), dtype=torch.float32)
        self.next_observation = torch.zeros((self.capacity, self.num_envs, observation_shape), dtype=torch.float32)
        self.action = torch.zeros((self.capacity, self.num_envs, action_size), dtype=torch.float32)
        self.reward = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32)
        self.done = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32)
        self.terminated = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32)
        self.step_idx = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(
        self,
        observation: Tensor,
        action: Tensor,
        reward: Tensor,
        next_observation: Tensor,
        done: Tensor,
        terminated: Tensor,
        step: Tensor,
    ):
        self.observation[self.buffer_index].copy_(observation.detach())
        self.action[self.buffer_index].copy_(action.detach())
        self.reward[self.buffer_index].copy_(reward.unsqueeze(-1).detach())
        self.next_observation[self.buffer_index].copy_(next_observation.detach())
        self.done[self.buffer_index].copy_(done.unsqueeze(-1).detach())
        self.terminated[self.buffer_index].copy_(terminated.unsqueeze(-1).detach())
        self.step_idx[self.buffer_index].copy_(step.unsqueeze(-1).detach())

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size, chunk_size) -> dict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        last_filled_index = self.buffer_index - chunk_size + 1
        assert self.full or (last_filled_index > batch_size), "too short dataset or too long chunk_size"
        sample_index = np.random.randint(0, self.capacity if self.full else last_filled_index, batch_size).reshape(
            -1, 1
        )
        chunk_length = np.arange(chunk_size).reshape(1, -1)

        sample_index = (sample_index + chunk_length) % self.capacity
        env_index = np.random.randint(0, self.num_envs, batch_size)
        flattened_index = sample_index * self.num_envs + env_index[:, None]

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        observation = flatten(self.observation)[flattened_index].to(self.device)
        next_observation = flatten(self.next_observation)[flattened_index].to(self.device)
        action = flatten(self.action)[flattened_index].to(self.device)
        reward = flatten(self.reward)[flattened_index].to(self.device)
        done = flatten(self.done)[flattened_index].to(self.device)
        terminated = flatten(self.terminated)[flattened_index].to(self.device)
        if_first = (flatten(self.step_idx)[flattened_index] == 1).to(self.device)

        sample = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "done": done,
            "terminated": terminated,
            "if_first": if_first,
        }
        return sample

    def mean_reward(self) -> float:
        if self.full:
            return float(self.reward.mean())
        else:
            return float(self.reward[: self.buffer_index].mean())

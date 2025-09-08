from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor


class ReplayBuffer:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity
        self.observation = torch.zeros((self.capacity, self.num_envs, observation_size), dtype=torch.float32)
        self.next_observation = torch.zeros((self.capacity, self.num_envs, observation_size), dtype=torch.float32)
        self.action = torch.zeros((self.capacity, self.num_envs, action_size), dtype=torch.float32)
        self.reward = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32)
        self.done = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32)
        self.terminated = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32)

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
    ):
        self.observation[self.buffer_index].copy_(observation.detach())
        self.action[self.buffer_index].copy_(action.detach())
        self.reward[self.buffer_index].copy_(reward.unsqueeze(-1))
        self.next_observation[self.buffer_index].copy_(next_observation.detach())
        self.done[self.buffer_index].copy_(done.unsqueeze(-1).detach())
        self.terminated[self.buffer_index].copy_(terminated.unsqueeze(-1).detach())

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size) -> dict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        assert self.full or (self.buffer_index * self.num_envs > batch_size), "too short dataset or too long chunk_size"
        high = self.capacity if self.full else self.buffer_index
        high = high * self.num_envs

        flattened_index = np.random.randint(0, high, size=batch_size)

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        observation = flatten(self.observation)[flattened_index]
        next_observation = flatten(self.next_observation)[flattened_index]
        action = flatten(self.action)[flattened_index]
        reward = flatten(self.reward)[flattened_index]
        done = flatten(self.done)[flattened_index]
        terminated = flatten(self.terminated)[flattened_index]

        sample = TensorDict(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            terminated=terminated,
            batch_size=observation.shape[0],
            device=self.device,
        )
        return sample

    def mean_reward(self) -> float:
        if self.full:
            return float(self.reward.mean())
        else:
            return float(self.reward[: self.buffer_index].mean())

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
        observation_shape: tuple[int],
        action_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        self.observation = np.empty((self.capacity, self.num_envs, *observation_shape), dtype=np.float32)
        self.next_observation = np.empty((self.capacity, self.num_envs, *observation_shape), dtype=np.float32)
        self.action = np.empty((self.capacity, self.num_envs, action_size), dtype=np.float32)
        self.reward = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.done = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)
        self.terminated = np.empty((self.capacity, self.num_envs, 1), dtype=np.float32)

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
        self.observation[self.buffer_index] = observation.detach().cpu().numpy()
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.unsqueeze(-1).detach().cpu().numpy()
        self.next_observation[self.buffer_index] = next_observation.detach().cpu().numpy()
        self.done[self.buffer_index] = done.unsqueeze(-1).detach().cpu().numpy()
        self.terminated[self.buffer_index] = terminated.unsqueeze(-1).detach().cpu().numpy()

        self.buffer_index = (self.buffer_index + 1) % self.capacity
        self.full = self.full or self.buffer_index == 0

    def sample(self, batch_size) -> dict[str, Tensor]:
        """
        Sample elements from the replay buffer in a sequential manner, without considering the episode
        boundaries.
        """
        assert self.full or (self.buffer_index > batch_size), "too short dataset or too long chunk_size"
        sample_index = np.random.randint(0, self.capacity if self.full else self.buffer_index, batch_size)
        env_index = np.random.randint(0, self.num_envs, batch_size)
        flattened_index = sample_index * self.num_envs + env_index

        def flatten(x: np.ndarray) -> np.ndarray:
            return x.reshape(-1, *x.shape[2:])

        observation = torch.as_tensor(flatten(self.observation)[flattened_index], device=self.device).float()
        next_observation = torch.as_tensor(flatten(self.next_observation)[flattened_index], device=self.device).float()
        action = torch.as_tensor(flatten(self.action)[flattened_index], device=self.device)
        reward = torch.as_tensor(flatten(self.reward)[flattened_index], device=self.device)
        done = torch.as_tensor(flatten(self.done)[flattened_index], device=self.device)
        terminated = torch.as_tensor(flatten(self.terminated)[flattened_index], device=self.device)

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

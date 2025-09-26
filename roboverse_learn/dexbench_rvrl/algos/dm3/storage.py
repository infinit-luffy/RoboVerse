from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import numpy as np
import torch
from loguru import logger as log
from rich.logging import RichHandler
from tensordict import TensorDict
from torch import Tensor

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: dict,
        action_size: int,
        device: str | torch.device,
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.num_envs = num_envs
        self.capacity = capacity

        self.observation = {
            key: np.empty((self.capacity, self.num_envs, *shape), dtype=np.float32)
            if "rgb" not in key
            else np.empty((self.capacity, self.num_envs, *shape), dtype=np.uint8)
            for key, shape in observation_shape.items()
        }
        self.next_observation = {
            key: np.empty((self.capacity, self.num_envs, *shape), dtype=np.float32)
            if "rgb" not in key
            else np.empty((self.capacity, self.num_envs, *shape), dtype=np.uint8)
            for key, shape in observation_shape.items()
        }
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
        for key in self.observation.keys():
            if "rgb" in key:
                self.observation[key][self.buffer_index] = (observation[key].detach().cpu().numpy() * 255.0).astype(
                    np.uint8
                )
            else:
                self.observation[key][self.buffer_index] = observation[key].detach().cpu().numpy()
        self.action[self.buffer_index] = action.detach().cpu().numpy()
        self.reward[self.buffer_index] = reward.unsqueeze(-1).detach().cpu().numpy()
        for key in self.next_observation.keys():
            if "rgb" in key:
                self.next_observation[key][self.buffer_index] = (
                    next_observation[key].detach().cpu().numpy() * 255.0
                ).astype(np.uint8)
            else:
                self.next_observation[key][self.buffer_index] = next_observation[key].detach().cpu().numpy()
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

        observation = {
            key: torch.as_tensor(flatten(self.observation[key])[flattened_index], device=self.device).float() / 255.0
            if "rgb" in key
            else torch.as_tensor(flatten(self.observation[key])[flattened_index], device=self.device).float()
            for key in self.observation.keys()
        }
        next_observation = {
            key: torch.as_tensor(flatten(self.next_observation[key])[flattened_index], device=self.device).float()
            / 255.0
            if "rgb" in key
            else torch.as_tensor(flatten(self.next_observation[key])[flattened_index], device=self.device).float()
            for key in self.next_observation.keys()
        }
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
        observation_shape: dict,
        action_size: int,
        device: str | torch.device,
        storage_device: str | torch.device = "cpu",
        num_envs: int = 1,
        capacity: int = 5000000,
    ):
        self.device = device
        self.storage_device = storage_device
        self.num_envs = num_envs
        self.capacity = capacity

        self.observation = TensorDict({
            key: torch.zeros((self.capacity, self.num_envs, *shape), dtype=torch.float32, device=self.storage_device)
            if "rgb" not in key
            else torch.zeros((self.capacity, self.num_envs, *shape), dtype=torch.uint8, device=self.storage_device)
            for key, shape in observation_shape.items()
        })
        self.next_observation = TensorDict({
            key: torch.zeros((self.capacity, self.num_envs, *shape), dtype=torch.float32, device=self.storage_device)
            if "rgb" not in key
            else torch.zeros((self.capacity, self.num_envs, *shape), dtype=torch.uint8, device=self.storage_device)
            for key, shape in observation_shape.items()
        })
        self.action = torch.zeros(
            (self.capacity, self.num_envs, action_size), dtype=torch.float32, device=self.storage_device
        )
        self.reward = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32, device=self.storage_device)
        self.done = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32, device=self.storage_device)
        self.terminated = torch.zeros(
            (self.capacity, self.num_envs, 1), dtype=torch.float32, device=self.storage_device
        )
        self.step_idx = torch.zeros((self.capacity, self.num_envs, 1), dtype=torch.float32, device=self.storage_device)

        self.buffer_index = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.buffer_index

    def add(
        self,
        observation: TensorDict,
        action: Tensor,
        reward: Tensor,
        next_observation: Tensor,
        done: Tensor,
        terminated: Tensor,
        step: Tensor,
    ):
        for key in self.observation.keys():
            if "rgb" in key:
                self.observation[key][self.buffer_index].copy_((observation[key] * 255.0).detach().to(torch.uint8))
            else:
                self.observation[key][self.buffer_index].copy_(observation[key].detach())
        self.action[self.buffer_index].copy_(action.detach())
        self.reward[self.buffer_index].copy_(reward.unsqueeze(-1).detach())
        for key in self.next_observation.keys():
            if "rgb" in key:
                self.next_observation[key][self.buffer_index].copy_(
                    (next_observation[key] * 255.0).detach().to(torch.uint8)
                )
            else:
                self.next_observation[key][self.buffer_index].copy_(next_observation[key].detach())
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

        observation = {
            key: (flatten(self.observation[key])[flattened_index].to(self.device).float() / 255.0)
            if "rgb" in key
            else flatten(self.observation[key])[flattened_index].to(self.device).float()
            for key in self.observation.keys()
        }
        next_observation = {
            key: (flatten(self.next_observation[key])[flattened_index].to(self.device).float() / 255.0)
            if "rgb" in key
            else flatten(self.next_observation[key])[flattened_index].to(self.device).float()
            for key in self.next_observation.keys()
        }
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

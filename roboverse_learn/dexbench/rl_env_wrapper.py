from __future__ import annotations

import copy
import random
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces
from loguru import logger as log

from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import SimType
from metasim.types import Action, EnvState
from metasim.utils.setup_util import get_sim_env_class
from metasim.utils.state import list_state_to_tensor


class BiDexEnvWrapper:
    """BiDex Environment Wrapper for RL tasks."""

    def __init__(self, scenario: ScenarioCfg, seed: int | None = None):
        """Initialize the BiDex environment wrapper."""
        self.sim_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert SimType(scenario.sim) == SimType.ISAACGYM, "Currently only support IsaacGym simulator."
        self.num_envs = scenario.num_envs
        self.robots = scenario.robots
        self.task = scenario.task

        env_class = get_sim_env_class(SimType(scenario.sim))
        self.env = env_class(scenario)

        self.init_states = [copy.deepcopy(scenario.task.init_states) for _ in range(self.num_envs)]
        self.init_states_tensor = list_state_to_tensor(
            handler=self.env.handler, env_states=self.init_states, device=self.sim_device
        )

        # action space is normalized to [-1, 1]
        self.action_shape = self.task.action_shape
        self.num_joints = 0
        for robot in self.robots:
            self.num_joints += robot.num_joints
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_shape,), dtype=np.float32)

        # observation space
        # Create an observation space (398 dimensions) for a single environment, instead of the entire batch (num_envs,398).
        self.obs_type = getattr(scenario.task, "obs_type", "state")
        self.use_prio = getattr(scenario.task, "use_prio", True)
        obs_shape = (self.task.obs_shape,)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=obs_shape, dtype=np.float32)
        if hasattr(self.task, "proprio_shape"):
            self.proprio_shape = self.task.proprio_shape
        else:
            self.proprio_shape = None
        if hasattr(self.task, "proceptual_shape"):
            self.proceptual_shape = self.task.proceptual_shape
        else:
            self.proceptual_shape = None
        if hasattr(self.task, "img_h") and hasattr(self.task, "img_w"):
            self.img_h = self.task.img_h
            self.img_w = self.task.img_w
        else:
            self.img_h = None
            self.img_w = None
        self.tensor_states = None
        self.tensor_obs = None
        log.info(f"Observation space: {self.observation_space}")
        log.info(f"Action space: {self.action_space}")

        self.max_episode_steps = self.task.episode_length
        log.info(f"Max episode steps: {self.max_episode_steps}")

        # Episode tracking variables for EpisodeLogCallback
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim_device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_goal_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.total_reset = 0
        self.total_success = 0
        self.mean_success_rate = 0.0
        self.reset_counts = deque(maxlen=100)
        self.success_counts = deque(maxlen=100)

        self.last_success_rate = 0.0
        if seed is not None:
            self.seed(seed)

    def scale_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale actions to the range of the action space.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).
        """
        return self.task.scale_action_fn(
            actions=actions,
        )

    def reset(self):
        """Reset the environment."""
        obs, _ = self.env.reset(states=self.init_states_tensor)
        self.task.update_state(obs)
        env_ids = torch.arange(self.num_envs, device=self.sim_device)
        self.tensor_states = obs
        observations = self.task.observation_fn(
            obs, torch.zeros((self.num_envs, self.action_shape), device=self.sim_device)
        )
        observations = torch.clamp(
            observations,
            torch.tensor(self.observation_space.low, device=self.sim_device),
            torch.tensor(self.observation_space.high, device=self.sim_device),
        )
        self.tensor_obs = observations
        self.reset_goal_pose(env_ids)

        # Reset episode tracking variables
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim_device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)
        self.episode_goal_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim_device)

        log.info("reset now")
        return observations

    def pre_physics_step(self, actions: torch.Tensor):
        """Step the environment with given actions.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).
        """
        step_action = self.scale_action_tensor(actions)
        return step_action

    def post_physics_step(self, envstates: list[EnvState], actions: torch.Tensor):
        """Post physics step processing."""
        self.episode_lengths += 1
        self.env._episode_length_buf += 1
        (self.episode_rewards, self.episode_reset, self.episode_goal_reset, self.episode_success) = self.task.reward_fn(
            envstates=envstates,
            actions=actions,
            reset_buf=self.episode_reset,
            reset_goal_buf=self.episode_goal_reset,
            episode_length_buf=self.episode_lengths,
            success_buf=self.episode_success,
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step the environment with given actions.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).

        Returns:
            tuple: A tuple containing the following elements:
                - observations (torch.Tensor, shape=(num_envs, obs_dim)): Observations from the environment.
                - rewards (torch.Tensor, shape=(num_envs,)): Reward values for each environment.
                - dones (torch.Tensor, shape=(num_envs,)): Flags indicating if the episode has ended for each environment (due to termination or truncation).
                - infos (list[dict]): List of additional information for each environment. Each dictionary contains the "TimeLimit.truncated" key,
                                      indicating if the episode was truncated due to timeout.
        """
        actions = torch.clamp(actions, -1.0, 1.0)  # Ensure actions are within [-1, 1]
        step_action = self.pre_physics_step(actions)
        envstates, _, _, _, _ = self.env.step(step_action)
        self.task.update_state(envstates)
        self.post_physics_step(envstates, actions)

        self.tensor_obs = self.task.observation_fn(envstates=envstates, actions=actions, device=self.sim_device)
        self.tensor_obs = torch.clamp(
            self.tensor_obs,
            torch.tensor(self.observation_space.low, device=self.sim_device),
            torch.tensor(self.observation_space.high, device=self.sim_device),
        )

        rewards = deepcopy(self.episode_rewards)
        dones = deepcopy(self.episode_reset)
        info = {}
        info["successes"] = deepcopy(self.episode_success)

        step_resets = self.episode_reset.sum().item()
        step_successes = torch.logical_and(self.episode_reset.bool(), self.episode_success.bool()).sum().item()

        success_rate = step_successes / step_resets if step_resets else self.last_success_rate
        self.last_success_rate = success_rate
        if len(self.reset_counts) == 100:
            self.total_reset -= self.reset_counts[0]
            self.total_success -= self.success_counts[0]

        self.reset_counts.append(step_resets)
        self.success_counts.append(step_successes)

        self.total_reset += step_resets
        self.total_success += step_successes

        if self.total_reset > 0:
            self.mean_success_rate = self.total_success / self.total_reset

        env_ids = self.episode_reset.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.episode_goal_reset.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_goal_pose(goal_env_ids)
        elif len(goal_env_ids) > 0:
            self.reset_goal_pose(goal_env_ids)

        if len(env_ids) > 0:
            envstates = self.reset_env(env_ids)
            actions[env_ids] = torch.zeros_like(actions[env_ids])  # Reset actions for the reset environments

        self.task.update_state(envstates)
        self.tensor_states = envstates
        observations = self.task.observation_fn(envstates=envstates, actions=actions, device=self.sim_device)
        observations = torch.clamp(
            observations,
            torch.tensor(self.observation_space.low, device=self.sim_device),
            torch.tensor(self.observation_space.high, device=self.sim_device),
        )

        info["success_rate"] = torch.tensor([success_rate], dtype=torch.float32, device=self.sim_device)
        info["total_succ_rate"] = torch.tensor([self.mean_success_rate], dtype=torch.float32, device=self.sim_device)

        return observations, rewards, dones, info

    def reset_env(self, env_ids: torch.Tensor):
        """Reset specific environments."""
        self.episode_lengths[env_ids] = 0
        self.episode_rewards[env_ids] = 0.0
        self.episode_success[env_ids] = 0
        self.episode_reset[env_ids] = 0
        self.reset_goal_pose(env_ids)
        reset_states = self.task.reset_init_pose_fn(self.init_states_tensor, env_ids=env_ids)
        env_states, _ = self.env.reset(states=reset_states, env_ids=env_ids.tolist())
        return env_states

    def reset_goal_pose(self, env_ids: torch.Tensor):
        """Reset the goal pose for specific environments."""
        self.episode_goal_reset[env_ids] = 0
        if self.task.goal_reset_fn is not None:
            self.task.goal_reset_fn(env_ids=env_ids)
        else:
            log.warning("No goal reset function defined in the task. Skipping goal reset.")

    def close(self):
        """Clean up environment resources."""
        self.env.close()

    def seed(self, seed):
        """Set random seed for reproducibility."""
        if seed == -1 and torch.cuda.is_available():
            seed = torch.randint(0, 10000, (1,))[0].item()
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        if hasattr(self.env.handler, "seed"):
            self.env.handler.seed(seed)
        elif hasattr(self.env.handler, "set_seed"):
            self.env.handler.set_seed(seed)
        else:
            log.warning("Could not set seed on underlying handler.")
        return seed

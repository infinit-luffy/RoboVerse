"""Base class for humanoid tasks."""

from __future__ import annotations

import logging
from typing import Literal

import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg, SimParamCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.types import EnvState
from metasim.utils import configclass, math
from metasim.utils.bidex_reward_util import compute_hand_reward

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Constants adapted from DexterousHands/bidexhands/tasks/shadow_hand_over.py
########################################################


@configclass
class ShadowHandOverTaskCfg(BaseRLTaskCfg):
    """class for bidex shadow hand over tasks."""

    source_benchmark = BenchmarkType.BIDEX
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 75  # TODO: may change
    objects_type = ["cube"]
    objects_urdf_path = ["roboverse_data/assets/bidex/objects/cube_multicolor.urdf"]
    objects = [
        RigidObjCfg(
            name=objects_type[0],
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            urdf_path=objects_urdf_path[0],
        ),
    ]
    robots = ["shadow_hand_right", "shadow_hand_left"]
    sim_params = SimParamCfg(
        dt=1.0 / 60.0,
        contact_offset=0.002,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.2,
        num_threads=4,
        use_gpu_pipeline=True,
        use_gpu=True,
        substeps=2,
    )

    goal_pos = None  # Placeholder for goal position, to be set later, shape (num_envs, 3)
    goal_rot = None  # Placeholder for goal rotation, to be set later, shape (num_envs, 4)
    r_fingertips = ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"]
    l_fingertips = ["robot1_ffdistal", "robot1_mfdistal", "robot1_rfdistal", "robot1_lfdistal", "robot1_thdistal"]
    r_fingertips_idx = None
    l_fingertips_idx = None
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 10.0  # Scale for force and torque observations
    shadow_hand_dof_lower_limits: torch.Tensor = torch.tensor(
        [
            -0.489,
            -0.698,
            -0.349,
            0,
            0,
            0,
            -0.349,
            0,
            0,
            0,
            -0.349,
            0,
            0,
            0,
            0,
            -0.349,
            0,
            0,
            0,
            -1.047,
            0,
            -0.209,
            -0.524,
            -1.571,
        ],
        dtype=torch.float32,
    )
    shadow_hand_dof_upper_limits: torch.Tensor = torch.tensor(
        [
            0.14,
            0.489,
            0.349,
            1.571,
            1.571,
            1.571,
            0.349,
            1.571,
            1.571,
            1.571,
            0.349,
            1.571,
            1.571,
            1.571,
            0.785,
            0.349,
            1.571,
            1.571,
            1.561,
            1.222,
            0.209,
            0.524,
            0,
        ],
        dtype=torch.float32,
    )
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )
    dist_reward_scale = 50
    success_tolerance = 0.1
    reach_goal_bonus = 250
    fall_penalty = 0.0

    def observation_fn(self, envstates: list[EnvState], actions) -> torch.Tensor:
        """Observation function for shadow hand over tasks.

        Args:
            envstates (list[EnvState]): List of environment states to process.
            actions (torch.Tensor): Actions taken by the agents in the environment, shape (num_envs, num_actions).

        Returns:
            Compute the observations of all environment. The observation is composed of three parts:
            the state values of the left and right hands, and the information of objects and target.
            The state values of the left and right hands were the same for each task, including hand
            joint and finger positions, velocity, and force information. The detail 422-dimensional
            observational space as shown in below:

            Index       Description
            0 - 23	    right shadow hand dof position
            24 - 47	    right shadow hand dof velocity
            48 - 71	    right shadow hand dof force
            72 - 136	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
            137 - 166	right shadow hand fingertip force, torque (5 x 6)
            167 - 186	right shadow hand actions
            187 - 210	left shadow hand dof position
            211 - 234	left shadow hand dof velocity
            235 - 258	left shadow hand dof force
            259 - 323	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
            324 - 353	left shadow hand fingertip force, torque (5 x 6)
            354 - 373	left shadow hand actions
            374 - 380	object pose
            381 - 383	object linear velocity
            383 - 386	object angle velocity
            387 - 393	goal pose
            394 - 397	goal rot - object rot
        """
        num_envs = envstates.robots["shadow_hand_right"].root_state.shape[0]
        obs = torch.zeros((num_envs, 398), dtype=torch.float32)
        obs[:, :24] = math.scale_transform(
            envstates.robots["shadow_hand_right"].joint_pos,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        obs[:, 24:48] = envstates.robots["shadow_hand_right"].joint_vel * self.vel_obs_scale
        obs[:, 48:72] = envstates.robots["shadow_hand_right"].joint_force * self.force_torque_obs_scale
        if self.r_fingertips_idx is None:
            self.r_fingertips_idx = [
                envstates.robots["shadow_hand_right"].body_names.index(name) for name in self.r_fingertips
            ]
        obs[:, 72:136] = envstates.robots["shadow_hand_right"].body_state[:, self.r_fingertips_idx, :]
        t = 137
        for name in self.r_fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            force = envstates.sensors[name].force  # (num_envs, 3)
            torque = envstates.sensors[name].torque  # (num_envs, 3)
            obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        obs[:, 167:187] = envstates.robots["shadow_hand_right"].actions[:, :20]  # actions for right hand
        obs[:, 187:211] = math.scale_transform(
            envstates.robots["shadow_hand_left"].joint_pos,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits,
        )
        obs[:, 211:235] = envstates.robots["shadow_hand_left"].joint_vel * self.vel_obs_scale
        obs[:, 235:259] = envstates.robots["shadow_hand_left"].joint_force * self.force_torque_obs_scale
        if self.l_fingertips_idx is None:
            self.l_fingertips_idx = [
                envstates.robots["shadow_hand_left"].body_names.index(name) for name in self.l_fingertips
            ]
        obs[:, 259:323] = envstates.robots["shadow_hand_left"].body_state[:, self.l_fingertips_idx, :]
        t = 324
        for name in self.l_fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            force = envstates.sensors[name].force
            torque = envstates.sensors[name].torque
            obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        obs[:, 354:374] = envstates.robots["shadow_hand_left"].actions[:, :20]  # actions for left hand
        obs[:, 374:387] = envstates.objects["cube"].root_state
        obs[:384:387] *= self.vel_obs_scale  # object angvel
        obs[:, 387:394] = torch.cat([self.goal_pos, self.goal_rot], dim=1)  # goal position and rotation (num_envs, 7)
        obs[:, 394:] = math.quat_mul(
            envstates.objects["cube"].root_state[:, 3:], math.quat_inv(self.goal_rot)
        )  # goal rotation - object rotation
        return obs

    def reward_fn(
        self,
        envstates: list[EnvState],
        actions,
        reset_buf,
        reset_goal_buf,
        _episode_length_buf,
    ) -> torch.Tensor:
        """Compute the reward of all environment. The core function is compute_hand_reward.

        Args:
            envstates (list[EnvState]): States of the environment
            actions (tensor): Actions of agents in the all environment, shape (num_envs, num_actions)
            _episode_length_buf (torch.Tensor): The episode length buffer, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)

        Returns:
            reward (torch.Tensor): The reward of all environments at this time, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)

        """
        (
            reward,
            reset_buf,
            reset_goal_buf,
        ) = compute_hand_reward(
            reset_buf=reset_buf,
            reset_goal_buf=reset_goal_buf,
            _episode_length_buf=_episode_length_buf,
            max_episode_length=self.episode_length,
            object_pos=envstates.objects["cube"].root_state[:, :3],
            object_rot=envstates.objects["cube"].root_state[:, 3:],
            target_pos=self.goal_pos,
            target_rot=self.goal_rot,
            dist_reward_scale=self.dist_reward_scale,
            actions=actions,
            success_tolerance=self.success_tolerance,
            reach_goal_bonus=self.reach_goal_bonus,
            fall_penalty=self.fall_penalty,
            ignore_z_rot=False,  # Todo : set to True if the object is a pen or similar object that does not require z-rotation alignment
        )
        return reward, reset_buf, reset_goal_buf

"""Base class for humanoid tasks."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Literal

import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots import ShadowHandCfg
from metasim.cfg.sensors.contact import ContactForceSensorCfg
from metasim.cfg.tasks.base_task_cfg import BaseRLTaskCfg, SimParamCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.types import EnvState
from metasim.utils import configclass, math
from metasim.utils.bidex_util import randomize_rotation

logging.addLevelName(5, "TRACE")
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

########################################################
## Constants adapted from DexterousHands/bidexhands/tasks/shadow_hand_over.py
########################################################


@configclass
class ShadowHandOver2UnderarmCfg(BaseRLTaskCfg):
    """class for bidex shadow hand over2underarm tasks."""

    source_benchmark = BenchmarkType.BIDEX
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 75
    traj_filepath = "roboverse_data/trajs/bidex/ShadowHandOver2Underarm/v2/initial_state_v2.json"
    device = "cuda:0"
    num_envs = None
    obs_shape = 422
    action_shape = 52
    current_object_type = "egg"
    objects_cfg = {
        "cube": RigidObjCfg(
            name="cube",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            urdf_path="roboverse_data/assets/bidex/objects/cube_multicolor.urdf",
            default_density=500.0,
        ),
        "egg": RigidObjCfg(
            name="egg",
            scale=(1, 1, 1),
            physics=PhysicStateType.RIGIDBODY,
            mjcf_path="roboverse_data/assets/bidex/open_ai_assets/mjcf/hand/egg.xml",
            isaacgym_read_mjcf=True,  # Use MJCF for IsaacGym
        ),
    }
    objects = []
    robots = [
        ShadowHandCfg(
            name="shadow_hand_right",
            fix_base_link=False,
            actuated_root=True,
            angular_damping=100.0,
            linear_damping=100.0,
        ),
        ShadowHandCfg(
            name="shadow_hand_left",
            fix_base_link=False,
            actuated_root=True,
            angular_damping=100.0,
            linear_damping=100.0,
        ),
    ]
    num_actuated_joints = {}
    for robot in robots:
        sum = 0
        for actuator in robot.actuators.values():
            if actuator.fully_actuated:
                sum += 1
        num_actuated_joints[robot.name] = sum
    decimation = 1
    sim_params = SimParamCfg(
        dt=1.0 / 60.0,
        contact_offset=0.002,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.2,
        num_threads=4,
        use_gpu_pipeline=True,
        use_gpu=True,
        substeps=2,
        friction_correlation_distance=0.025,
        friction_offset_threshold=0.04,
    )
    dt = sim_params.dt  # Simulation time step
    transition_scale = 0.1
    orientation_scale = 2
    init_goal_pos = torch.tensor(
        [0.0, -0.4, 0.55], dtype=torch.float32, device=device
    )  # Initial goal position, shape (3,)
    init_goal_rot = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device
    )  # Initial goal rotation, shape (4,)
    goal_pos = None  # Placeholder for goal position, to be set later, shape (num_envs, 3)
    goal_rot = None  # Placeholder for goal rotation, to be set later, shape (num_envs, 4)
    fingertips = ["robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal"]
    sensors = []
    for name in fingertips:
        r_name = "right" + name
        sensors.append(ContactForceSensorCfg(base_link=("shadow_hand_right", name), source_link=None, name=r_name))
    for name in fingertips:
        l_name = "left" + name
        sensors.append(ContactForceSensorCfg(base_link=("shadow_hand_left", name), source_link=None, name=l_name))
    r_fingertips_idx = None
    l_fingertips_idx = None
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 10.0  # Scale for force and torque observations
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )
    dist_reward_scale = 20.0
    action_penalty_scale = 0
    success_tolerance = 0.1
    reach_goal_bonus = 250.0
    throw_bonus = 15.0
    fall_penalty = 0.0
    reset_position_noise = 0.0
    reset_dof_pos_noise = 0.0

    def set_objects(self) -> None:
        self.objects.append(self.objects_cfg[self.current_object_type])

    def set_init_states(self) -> None:
        """Set the initial states for the shadow hand over task."""
        self.joint_reindex = torch.tensor(
            [5, 4, 3, 2, 18, 17, 16, 15, 14, 9, 8, 7, 6, 13, 12, 11, 10, 23, 22, 21, 20, 19, 1, 0],
            dtype=torch.int32,
            device=self.device,
        )
        self.actuated_dof_indices = torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                6,
                7,
                8,
                10,
                11,
                12,
                14,
                15,
                16,
                17,
                19,
                20,
                21,
                22,
                23,
            ],
            dtype=torch.int32,
            device=self.device,
        )
        self.shadow_hand_dof_lower_limits: torch.Tensor = torch.tensor(
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
            device=self.device,
        )
        self.shadow_hand_dof_upper_limits: torch.Tensor = torch.tensor(
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
                1.571,
                1.047,
                1.222,
                0.209,
                0.524,
                0,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.shadow_hand_dof_lower_limits_cpu = self.shadow_hand_dof_lower_limits.cpu()
        self.shadow_hand_dof_upper_limits_cpu = self.shadow_hand_dof_upper_limits.cpu()
        self.init_states = [
            {
                "objects": {
                    self.current_object_type: {
                        "pos": torch.tensor([0.0, -0.05, 1.0]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    },
                },
                "robots": {
                    "shadow_hand_right": {
                        "pos": torch.tensor([0.0, 0.0, 0.5]),
                        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        "dof_pos": {
                            "robot0_WRJ1": 0.0,
                            "robot0_WRJ0": 0.0,
                            "robot0_FFJ3": 0.0,
                            "robot0_FFJ2": 0.0,
                            "robot0_FFJ1": 0.0,
                            "robot0_FFJ0": 0.0,
                            "robot0_MFJ3": 0.0,
                            "robot0_MFJ2": 0.0,
                            "robot0_MFJ1": 0.0,
                            "robot0_MFJ0": 0.0,
                            "robot0_RFJ3": 0.0,
                            "robot0_RFJ2": 0.0,
                            "robot0_RFJ1": 0.0,
                            "robot0_RFJ0": 0.0,
                            "robot0_LFJ4": 0.0,
                            "robot0_LFJ3": 0.0,
                            "robot0_LFJ2": 0.0,
                            "robot0_LFJ1": 0.0,
                            "robot0_LFJ0": 0.0,
                            "robot0_THJ4": 0.0,
                            "robot0_THJ3": 0.0,
                            "robot0_THJ2": 0.0,
                            "robot0_THJ1": 0.0,
                            "robot0_THJ0": 0.0,
                        },
                    },
                    "shadow_hand_left": {
                        "pos": torch.tensor([0.0, -0.8, 0.5]),
                        "rot": torch.tensor([-0.707, 0.707, 0.0, 0.0]),
                        "dof_pos": {
                            "robot0_WRJ1": 0.0,
                            "robot0_WRJ0": 0.0,
                            "robot0_FFJ3": 0.0,
                            "robot0_FFJ2": 0.0,
                            "robot0_FFJ1": 0.0,
                            "robot0_FFJ0": 0.0,
                            "robot0_MFJ3": 0.0,
                            "robot0_MFJ2": 0.0,
                            "robot0_MFJ1": 0.0,
                            "robot0_MFJ0": 0.0,
                            "robot0_RFJ3": 0.0,
                            "robot0_RFJ2": 0.0,
                            "robot0_RFJ1": 0.0,
                            "robot0_RFJ0": 0.0,
                            "robot0_LFJ4": 0.0,
                            "robot0_LFJ3": 0.0,
                            "robot0_LFJ2": 0.0,
                            "robot0_LFJ1": 0.0,
                            "robot0_LFJ0": 0.0,
                            "robot0_THJ4": 0.0,
                            "robot0_THJ3": 0.0,
                            "robot0_THJ2": 0.0,
                            "robot0_THJ1": 0.0,
                            "robot0_THJ0": 0.0,
                        },
                    },
                },
            }
        ]

    def scale_action_fn(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale actions to the range of the action space.

        Args:
            actions (torch.Tensor): Actions in the range of [-1, 1], shape (num_envs, num_actions).
        """
        step_actions = torch.zeros((self.num_envs, 60), device=self.device)
        scaled_actions_1 = math.unscale_transform(
            actions[:, 6:26],
            self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
            self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
        )
        scaled_actions_2 = math.unscale_transform(
            actions[:, 32:52],
            self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
            self.shadow_hand_dof_upper_limits[self.actuated_dof_indices],
        )

        step_actions[:, self.actuated_dof_indices] = scaled_actions_1
        step_actions[:, self.actuated_dof_indices + 24] = scaled_actions_2
        step_actions[:, 48:51] = actions[:, :3] * self.dt * self.transition_scale * 100000
        step_actions[:, 51:54] = actions[:, 26:29] * self.dt * self.transition_scale * 100000
        step_actions[:, 54:57] = actions[:, 3:6] * self.dt * self.orientation_scale * 1000
        step_actions[:, 57:60] = actions[:, 29:32] * self.dt * self.orientation_scale * 1000
        return step_actions

    def observation_fn(self, envstates: list[EnvState], actions: torch.Tensor, device=None) -> torch.Tensor:
        """Observation function for shadow hand over tasks.

        Args:
            envstates (list[EnvState]): List of environment states to process.
            actions (torch.Tensor): Actions taken by the agents in the environment, shape (num_envs, num_actions).
            device (torch.device | None): The device to use for the observations. If None, use the task's device.

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
            167 - 169	right shadow hand base position
            170 - 172	right shadow hand base rotation
            173 - 198	right shadow hand actions
            199 - 222	left shadow hand dof position
            223 - 246	left shadow hand dof velocity
            247 - 270	left shadow hand dof force
            271 - 335	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
            336 - 365	left shadow hand fingertip force, torque (5 x 6)
            366 - 368	left shadow hand base position
            369 - 371	left shadow hand base rotation
            372 - 397	left shadow hand actions
            398 - 404	object pose
            405 - 407	object linear velocity
            408 - 410	object angle velocity
            411 - 417	goal pose
            418 - 421	goal rot - object rot
        """
        if device is None:
            device = self.device
        num_envs = envstates.robots["shadow_hand_right"].root_state.shape[0]
        if self.num_envs is None:
            self.num_envs = num_envs
        if self.goal_pos is None:
            self.goal_pos = torch.tensor(self.init_goal_pos, dtype=torch.float32).view(1, -1).repeat(num_envs, 1)
        if self.goal_rot is None:
            self.goal_rot = torch.tensor(self.init_goal_rot, dtype=torch.float32).view(1, -1).repeat(num_envs, 1)
        obs = torch.zeros((num_envs, self.obs_shape), dtype=torch.float32, device=device)
        obs[:, :24] = math.scale_transform(
            envstates.robots["shadow_hand_right"].joint_pos,
            self.shadow_hand_dof_lower_limits[self.joint_reindex],
            self.shadow_hand_dof_upper_limits[self.joint_reindex],
        )
        obs[:, 24:48] = envstates.robots["shadow_hand_right"].joint_vel * self.vel_obs_scale
        obs[:, 48:72] = envstates.robots["shadow_hand_right"].joint_force * self.force_torque_obs_scale
        if self.r_fingertips_idx is None:
            self.r_fingertips_idx = [
                envstates.robots["shadow_hand_right"].body_names.index(name) for name in self.fingertips
            ]
        obs[:, 72:137] = (
            envstates.robots["shadow_hand_right"].body_state[:, self.r_fingertips_idx, :].view(num_envs, -1)
        )
        t = 137
        for name in self.fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            r_name = "right" + name
            force = envstates.sensors[r_name].force  # (num_envs, 3)
            torque = envstates.sensors[r_name].torque  # (num_envs, 3)
            obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        obs[:, 167:170] = envstates.robots["shadow_hand_right"].root_state[:, :3]  # right hand base position
        roll, pitch, yaw = math.euler_xyz_from_quat(envstates.robots["shadow_hand_right"].root_state[:, 3:7])
        obs[:, 170] = roll
        obs[:, 171] = pitch
        obs[:, 172] = yaw  # right hand base rotation (roll, pitch, yaw)
        # actions for right hand
        obs[:, 173:199] = actions[:, :26]  # actions for right hand
        obs[:, 199:223] = math.scale_transform(
            envstates.robots["shadow_hand_left"].joint_pos,
            self.shadow_hand_dof_lower_limits[self.joint_reindex],
            self.shadow_hand_dof_upper_limits[self.joint_reindex],
        )
        obs[:, 223:247] = envstates.robots["shadow_hand_left"].joint_vel * self.vel_obs_scale
        obs[:, 247:271] = envstates.robots["shadow_hand_left"].joint_force * self.force_torque_obs_scale
        if self.l_fingertips_idx is None:
            self.l_fingertips_idx = [
                envstates.robots["shadow_hand_left"].body_names.index(name) for name in self.fingertips
            ]
        obs[:, 271:336] = (
            envstates.robots["shadow_hand_left"].body_state[:, self.l_fingertips_idx, :].view(num_envs, -1)
        )
        t = 336
        for name in self.fingertips:
            # shape: (num_envs, 3) + (num_envs, 3) => (num_envs, 6)
            l_name = "left" + name
            force = envstates.sensors[l_name].force
            torque = envstates.sensors[l_name].torque
            obs[:, t : t + 6] = torch.cat([force, torque], dim=1) * self.force_torque_obs_scale  # (num_envs, 6)
            t += 6
        obs[:, 366:369] = envstates.robots["shadow_hand_left"].root_state[:, :3]  # left hand base position
        roll, pitch, yaw = math.euler_xyz_from_quat(envstates.robots["shadow_hand_left"].root_state[:, 3:7])
        obs[:, 369] = roll
        obs[:, 370] = pitch
        obs[:, 371] = yaw  # left hand base rotation (roll, pitch, yaw)
        obs[:, 372:398] = actions[:, 26:]  # actions for left hand
        obs[:, 398:411] = envstates.objects[self.current_object_type].root_state
        obs[:, 408:411] *= self.vel_obs_scale  # object angvel
        obs[:, 411:418] = torch.cat([self.goal_pos, self.goal_rot], dim=1)  # goal position and rotation (num_envs, 7)
        obs[:, 418:] = math.quat_mul(
            envstates.objects[self.current_object_type].root_state[:, 3:7], math.quat_inv(self.goal_rot)
        )  # goal rotation - object rotation
        return obs

    def reward_fn(
        self,
        envstates: list[EnvState],
        actions,
        reset_buf,
        reset_goal_buf,
        episode_length_buf,
        success_buf,
    ) -> torch.Tensor:
        """Compute the reward of all environment. The core function is compute_hand_reward.

        Args:
            envstates (list[EnvState]): States of the environment
            actions (tensor): Actions of agents in the all environment, shape (num_envs, num_actions)
            episode_length_buf (torch.Tensor): The episode length buffer, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)
            success_buf (torch.Tensor): The success buffer of all environments at this time, shape (num_envs,)
            success_rate (float): The success rate of the environment, used to determine the success condition

        Returns:
            reward (torch.Tensor): The reward of all environments at this time, shape (num_envs,)
            reset_buf (torch.Tensor): The reset buffer of all environments at this time, shape (num_envs,)
            reset_goal_buf (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs,)
            success_buf (torch.Tensor): The success buffer of all environments at this time, shape (num_envs,)
        """
        (reward, reset_buf, reset_goal_buf, success_buf) = compute_hand_reward(
            reset_buf=reset_buf,
            reset_goal_buf=reset_goal_buf,
            episode_length_buf=episode_length_buf,
            success_buf=success_buf,
            max_episode_length=self.episode_length,
            object_pos=envstates.objects[self.current_object_type].root_state[:, :3],
            object_rot=envstates.objects[self.current_object_type].root_state[:, 3:7],
            target_pos=self.goal_pos,
            target_rot=self.goal_rot,
            right_hand_base_pos=envstates.robots["shadow_hand_right"].root_state[:, :3],
            left_hand_base_pos=envstates.robots["shadow_hand_left"].root_state[:, :3],
            dist_reward_scale=self.dist_reward_scale,
            action_penalty_scale=self.action_penalty_scale,
            actions=actions,
            success_tolerance=self.success_tolerance,
            reach_goal_bonus=self.reach_goal_bonus,
            throw_bonus=self.throw_bonus,
            fall_penalty=self.fall_penalty,
            ignore_z_rot=False,  # Todo : set to True if the object is a pen or similar object that does not require z-rotation alignment
        )
        return reward, reset_buf, reset_goal_buf, success_buf

    def goal_reset_fn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset the goal position and rotation for the environment.

        Args:
            env_ids (torch.Tensor): The reset goal buffer of all environments at this time, shape (num_envs_to_reset,).
        """
        rand_floats = math.torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
        x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], x_unit_tensor, y_unit_tensor)

        self.goal_pos[env_ids] = self.init_goal_pos.clone()
        self.goal_rot[env_ids] = new_rot

        return

    def reset_init_pose_fn(self, init_states: list[EnvState], env_ids: torch.Tensor) -> list[EnvState]:
        """Reset the initial pose of the environment.

        Args:
            init_states (list[EnvState]): States of the environment
            env_ids (torch.Tensor): The indices of the environments to reset.

        Returns:
            reset_state: The updated states of the environment after resetting.
        """
        if self.reset_dof_pos_noise == 0.0 and self.reset_position_noise == 0.0:
            # If no noise is applied, return the initial states directly
            return deepcopy(init_states)
        reset_state = deepcopy(init_states)
        num_shadow_hand_dofs = self.shadow_hand_dof_lower_limits.shape[0]
        x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device="cpu").repeat((len(env_ids), 1))
        y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device="cpu").repeat((len(env_ids), 1))

        # generate random values
        rand_floats = math.torch_rand_float(-1.0, 1.0, (len(env_ids), num_shadow_hand_dofs + 5), device="cpu")

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], x_unit_tensor, y_unit_tensor)

        robot_dof_default_pos = torch.tensor(
            list(self.init_states[0]["robots"][self.robots[0].name]["dof_pos"].values()),
            dtype=torch.float32,
            device="cpu",
        )
        delta_max = self.shadow_hand_dof_upper_limits_cpu - robot_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits_cpu - robot_dof_default_pos

        for i, env_id in enumerate(env_ids):
            # reset object
            for obj_name in reset_state[env_id]["objects"].keys():
                reset_state[env_id]["objects"][obj_name]["pos"][:3] += self.reset_position_noise * rand_floats[i, :3]
                reset_state[env_id]["objects"][obj_name]["rot"] = new_object_rot[i]

            # reset shadow hand
            for robot_name in reset_state[env_id]["robots"].keys():
                rand_delta = delta_min + (delta_max - delta_min) * rand_floats[i, 5 : 5 + num_shadow_hand_dofs]
                dof_pos = robot_dof_default_pos + self.reset_dof_pos_noise * rand_delta
                reset_state[env_id]["robots"][robot_name]["dof_pos"] = {
                    name: dof_pos[j].item()
                    for j, name in enumerate(reset_state[env_id]["robots"][robot_name]["dof_pos"].keys())
                }

        return reset_state


@torch.jit.script
def compute_hand_reward(
    reset_buf,
    reset_goal_buf,
    episode_length_buf,
    success_buf,
    max_episode_length: float,
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    right_hand_base_pos,
    left_hand_base_pos,
    dist_reward_scale: float,
    action_penalty_scale: float,
    actions,
    success_tolerance: float,
    reach_goal_bonus: float,
    throw_bonus: float,
    fall_penalty: float,
    ignore_z_rot: bool,
):
    """Compute the reward of all environment.

    Args:
        reset_buf (tensor): The reset buffer of all environments at this time, shape (num_envs,)

        reset_goal_buf (tensor): The reset goal buffer of all environments at this time, shape (num_envs,)

        episode_length_buf (tensor): The porgress buffer of all environments at this time, shape (num_envs,)

        success_buf (tensor): The success buffer of all environments at this time, shape (num_envs,)

        max_episode_length (float): The max episode length in this environment

        object_pos (tensor): The position of the object

        object_rot (tensor): The rotation of the object

        target_pos (tensor): The position of the target

        target_rot (tensor): The rotate of the target

        right_hand_base_pos (tensor): The base position of the right hand

        left_hand_base_pos (tensor): The base position of the left hand

        dist_reward_scale (float): The scale of the distance reward

        action_penalty_scale (float): The scale of the action penalty

        actions (tensor): The action buffer of all environments at this time

        success_tolerance (float): The tolerance of the success determined

        reach_goal_bonus (float): The reward given when the object reaches the goal

        throw_bonus (float): The reward given when the object is thrown

        fall_penalty (float): The reward given when the object is fell

        ignore_z_rot (bool): Is it necessary to ignore the rot of the z-axis, which is usually used
            for some specific objects (e.g. pen)
    """
    # Distance from the hand to the object
    # diff_xy = target_pos[:, :2] - object_pos[:, :2]
    # reward_dist_xy = torch.norm(diff_xy, p=2, dim=-1)
    # reward_dist_z = torch.clamp(torch.abs(target_pos[:, 2] - object_pos[:, 2]), max=0.03)
    # reward_dist = torch.where(reward_dist_xy <= 0.23, reward_dist_xy + 0.05 * reward_dist_z, reward_dist_xy)
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    reward_dist = goal_dist
    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = math.quat_mul(object_rot, math.quat_inv(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))

    dist_rew = reward_dist

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = torch.exp(-0.2 * (dist_rew * dist_reward_scale + rot_dist)) - action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(
        torch.abs(goal_dist) <= 0.03,
        torch.ones_like(reset_goal_buf),
        reset_goal_buf,
    )
    success_buf = torch.where(
        success_buf == 0,
        torch.where(
            torch.abs(goal_dist) <= 0.03,
            torch.ones_like(success_buf),
            success_buf,
        ),
        success_buf,
    )

    right_hand_base_dist = torch.norm(right_hand_base_pos - torch.tensor([0.0, 0.0, 0.5], dtype=torch.float, device=right_hand_base_pos.device), p=2, dim=-1)
    left_hand_base_dist = torch.norm(left_hand_base_pos - torch.tensor([0.0, -0.8, 0.5], dtype=torch.float, device=left_hand_base_pos.device), p=2, dim=-1)


    # Reward for throwing the object
    # thrown = (diff_xy[:, 1] >= -0.25) & (diff_xy[:, 1] <= -0.1) & (object_pos[:, 2] >= 0.4)
    # reward = torch.where(thrown, reward + throw_bonus, reward)

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.1, reward - fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(right_hand_base_dist >= 0.1, torch.ones_like(resets), resets)
    resets = torch.where(left_hand_base_dist >= 0.1, torch.ones_like(resets), resets)

    # Reset because of terminate or fall or success
    resets = torch.where(episode_length_buf >= max_episode_length, torch.ones_like(resets), resets)
    resets = torch.where(success_buf >= 1, torch.ones_like(resets), resets)

    return reward, resets, goal_resets, success_buf

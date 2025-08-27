"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import time

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler


@configclass
class Args:
    """Arguments for the static scene."""

    ## Handlers
    sim: Literal[
        "isaacsim",
        "isaacgym",
        "genesis",
        "pybullet",
        "sapien2",
        "sapien3",
        "mujoco",
        "mjx",
    ] = "mujoco"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

robot = RobotCfg(
    name="new_robot_h1",
    num_joints=26,
    usd_path="roboverse_data/robots/h1/usd/h1.usd",
    mjcf_path="roboverse_data/robots/h1/mjcf/h1.xml",
    urdf_path="roboverse_data/robots/h1/urdf/h1.urdf",
    enabled_gravity=True,
    fix_base_link=False,
    enabled_self_collisions=False,
    isaacgym_flip_visual_attachments=False,
    collapse_fixed_joints=True,
    actuators={
        "left_hip_yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_roll": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "left_knee": BaseActuatorCfg(stiffness=300, damping=6),
        "left_ankle": BaseActuatorCfg(stiffness=40, damping=2),
        "right_hip_yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_roll": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "right_knee": BaseActuatorCfg(stiffness=300, damping=6),
        "right_ankle": BaseActuatorCfg(stiffness=40, damping=2),
        "torso": BaseActuatorCfg(stiffness=300, damping=6),
        "left_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_roll": BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_yaw": BaseActuatorCfg(stiffness=100, damping=2),
        "left_elbow": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_roll": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_yaw": BaseActuatorCfg(stiffness=100, damping=2),
        "right_elbow": BaseActuatorCfg(stiffness=100, damping=2),
    },
    joint_limits={
        "left_hip_yaw": (-0.43, 0.43),
        "left_hip_roll": (-0.43, 0.43),
        "left_hip_pitch": (-3.14, 2.53),
        "left_knee": (-0.26, 2.05),
        "left_ankle": (-0.87, 0.52),
        "right_hip_yaw": (-0.43, 0.43),
        "right_hip_roll": (-0.43, 0.43),
        "right_hip_pitch": (-3.14, 2.53),
        "right_knee": (-0.26, 2.05),
        "right_ankle": (-0.87, 0.52),
        "torso": (-2.35, 2.35),
        "left_shoulder_pitch": (-2.87, 2.87),
        "left_shoulder_roll": (-0.34, 3.11),
        "left_shoulder_yaw": (-1.3, 4.45),
        "left_elbow": (-1.25, 2.61),
        "right_shoulder_pitch": (-2.87, 2.87),
        "right_shoulder_roll": (-3.11, 0.34),
        "right_shoulder_yaw": (-4.45, 1.3),
        "right_elbow": (-1.25, 2.61),
    },
    control_type={
        "left_hip_yaw": "position",
        "left_hip_roll": "position",
        "left_hip_pitch": "position",
        "left_knee": "position",
        "left_ankle": "position",
        "right_hip_yaw": "position",
        "right_hip_roll": "position",
        "right_hip_pitch": "position",
        "right_knee": "position",
        "right_ankle": "position",
        "torso": "position",
        "left_shoulder_pitch": "position",
        "left_shoulder_roll": "position",
        "left_shoulder_yaw": "position",
        "left_elbow": "position",
        "right_shoulder_pitch": "position",
        "right_shoulder_roll": "position",
        "right_shoulder_yaw": "position",
        "right_elbow": "position",
    },
    default_joint_positions={
        "left_hip_yaw": 0.0,
        "left_hip_roll": 0.0,
        "left_hip_pitch": -0.4,
        "left_knee": 0.8,
        "left_ankle": -0.4,
        "right_hip_yaw": 0.0,
        "right_hip_roll": 0.0,
        "right_hip_pitch": -0.4,
        "right_knee": 0.8,
        "right_ankle": -0.4,
        "torso": 0.0,
        "left_shoulder_pitch": 0.0,
        "left_shoulder_roll": 0,
        "left_shoulder_yaw": 0.0,
        "left_elbow": 0.0,
        "right_shoulder_pitch": 0.0,
        "right_shoulder_roll": 0.0,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.0,
    },
)
# initialize scenario
scenario = ScenarioCfg(
    robots=[robot],
    simulator=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(name="camera_0", pos=(2.0, -1.0, 1.49), look_at=(0.0, -0.5, 0.89))]

# add objects
scenario.objects = [
    RigidObjCfg(
        name="cube",
        scale=(1, 1, 1),
        physics=PhysicStateType.RIGIDBODY,
    ),
    PrimitiveSphereCfg(
        name="sphere",
        radius=0.1,
        color=[0.0, 0.0, 1.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    RigidObjCfg(
        name="bbq_sauce",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
    ArticulationObjCfg(
        name="box_base",
        fix_base_link=True,
        usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
        urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
        mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
    ),
]


log.info(f"Using simulator: {args.sim}")
handler = get_handler(scenario)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.0, -0.37, 0.86]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
        },
        "robots": {
            "franka_shadow_left": {
                "pos": torch.tensor([0.0, -1.336, 0.0]),
                "rot": torch.tensor([0.7071, 0, 0, 0.7071]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.1415926,
                    "panda_joint7": -2.356194,
                    "WRJ2": 0.0,
                    "WRJ1": 0.0,
                    "FFJ4": 0.0,
                    "FFJ3": 0.0,
                    "FFJ2": 0.0,
                    "FFJ1": 0.0,
                    "MFJ4": 0.0,
                    "MFJ3": 0.0,
                    "MFJ2": 0.0,
                    "MFJ1": 0.0,
                    "RFJ4": 0.0,
                    "RFJ3": 0.0,
                    "RFJ2": 0.0,
                    "RFJ1": 0.0,
                    "LFJ5": 0.0,
                    "LFJ4": 0.0,
                    "LFJ3": 0.0,
                    "LFJ2": 0.0,
                    "LFJ1": 0.0,
                    "THJ5": 0.0,
                    "THJ4": 0.0,
                    "THJ3": 0.0,
                    "THJ2": 0.0,
                    "THJ1": 0.0,
                },
            },
            "franka_shadow_right": {
                "pos": torch.tensor([0.0, 0.336, 0.0]),
                "rot": torch.tensor([0.7071, 0, 0, -0.7071]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.1415928,
                    "panda_joint7": -2.356194,
                    "WRJ2": 0.0,
                    "WRJ1": 0.0,
                    "FFJ4": 0.0,
                    "FFJ3": 0.0,
                    "FFJ2": 0.0,
                    "FFJ1": 0.0,
                    "MFJ4": 0.0,
                    "MFJ3": 0.0,
                    "MFJ2": 0.0,
                    "MFJ1": 0.0,
                    "RFJ4": 0.0,
                    "RFJ3": 0.0,
                    "RFJ2": 0.0,
                    "RFJ1": 0.0,
                    "LFJ5": 0.0,
                    "LFJ4": 0.0,
                    "LFJ3": 0.0,
                    "LFJ2": 0.0,
                    "LFJ1": 0.0,
                    "THJ5": 0.0,
                    "THJ4": 0.0,
                    "THJ3": 0.0,
                    "THJ2": 0.0,
                    "THJ1": 0.0,
                },
            },
        },
    }
]
num_robo_dof = {}
for robot in scenario.robots:
    num_robo_dof[robot.name] = robot.num_joints
dof_num = 0
for robot in scenario.robots:
    dof_num += num_robo_dof[robot.name]
joint_low_limits = {}
joint_high_limits = {}
for robot in scenario.robots:
    joint_low_limits[robot.name] = torch.zeros(num_robo_dof[robot.name], device="cuda:0")
    joint_high_limits[robot.name] = torch.zeros(num_robo_dof[robot.name], device="cuda:0")
    for i, joint_name in enumerate(robot.joint_limits.keys()):
        if robot.actuators[joint_name].fully_actuated:
            joint_low_limits[robot.name][i] = robot.joint_limits[joint_name][0]
            joint_high_limits[robot.name][i] = robot.joint_limits[joint_name][1]
            i += 1
obs, extras = env.reset(states=init_states)
for robot in scenario.robots:
    robot.update_state(obs)
os.makedirs("get_started/output", exist_ok=True)


## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/2_add_new_robot_{args.sim}.mp4")
obs_saver.add(obs)
start_time = time.time()
step = 0
for _ in range(100):
    log.debug(f"Step {step}")
    # actions = [
    #     {
    #         robot.name: {
    #             "dof_pos_target": {
    #                 joint_name: (
    #                     torch.rand(1).item() * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
    #                     + robot.joint_limits[joint_name][0]
    #                 )
    #                 for joint_name in robot.joint_limits.keys()
    #                 if robot.actuators[joint_name].fully_actuated
    #             }
    #         }
    #         for robot in scenario.robots
    #     }
    #     for _ in range(scenario.num_envs)
    # ]
    left_pos_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    left_pos_err[:, 1] = 0.0
    left_pos_err[:, 2] = 0.0
    left_rot_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    left_dpose = torch.cat([left_pos_err, left_rot_err], -1).unsqueeze(-1)
    left_targets = scenario.robots[0].control_arm_ik(left_dpose, args.num_envs, "cuda:0")

    right_pos_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    right_pos_err[:, 1] = 0.0
    right_pos_err[:, 2] = 0.0
    right_rot_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    right_dpose = torch.cat([right_pos_err, right_rot_err], -1).unsqueeze(-1)
    right_targets = scenario.robots[1].control_arm_ik(right_dpose, args.num_envs, "cuda:0")

    left_ft_pos_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    left_ft_pos_err[..., 1] = 0.03
    right_ft_pos_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    right_ft_pos_err[..., 1] = -0.03
    left_ft_rot_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    right_ft_rot_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    left_dof_pos = scenario.robots[0].control_hand_ik(left_ft_pos_err, left_ft_rot_err)
    right_dof_pos = scenario.robots[1].control_hand_ik(right_ft_pos_err, right_ft_rot_err)

    actions = torch.zeros((args.num_envs, dof_num), device="cuda:0")
    num_dof = 0
    for robot in scenario.robots:
        arm_dof_idx = [i + num_dof for i in robot.arm_dof_idx]
        hand_dof_idx = [i + num_dof for i in robot.hand_dof_idx]
        actions[:, arm_dof_idx] = left_targets if robot.name == "franka_shadow_left" else right_targets
        actions[:, hand_dof_idx] = left_dof_pos if robot.name == "franka_shadow_left" else right_dof_pos
        num_dof += num_robo_dof[robot.name]
    obs, reward, success, time_out, extras = env.step(actions)
    for robot in scenario.robots:
        robot.update_state(obs)
    obs_saver.add(obs)
    step += 1
    if step % 10 == 0:
        log.info(f"Step {step}, Time Elapsed: {time.time() - start_time:.2f} seconds")
        start_time = time.time()

obs_saver.save()

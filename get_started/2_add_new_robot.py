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


def quat_to_rotvec(q):  # q: (..., 4), in (w, x, y, z)
    q = q / q.norm(p=2, dim=-1, keepdim=True)  # normalize
    angle = 2.0 * torch.acos(torch.clamp(q[..., 0], -1.0, 1.0))
    sin_half_angle = torch.sqrt(1 - q[..., 0] ** 2)
    axis = q[..., 1:] / (sin_half_angle.unsqueeze(-1) + 1e-6)
    return axis * angle.unsqueeze(-1)


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
    sim_params=sim_params,
    decimation=1,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(name="camera_0", pos=(-1.7, 0.0, 1.275), look_at=(0.0, 0.0, 0.675))]

# add objects
scenario.objects = [
    # RigidObjCfg(
    #     name="cube1",
    #     scale=(1, 1, 1),
    #     physics=PhysicStateType.RIGIDBODY,
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/cube_multicolor.urdf",
    #     usd_path="roboverse_data/assets/bidex/objects/usd/cube_multicolor.usd",
    #     default_density=500.0,
    #     use_vhacd=True,
    #     fix_base_link=True,
    # ),
    # RigidObjCfg(
    #     name="cube2",
    #     scale=(0.2, 0.2, 0.2),
    #     physics=PhysicStateType.RIGIDBODY,
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/cube_multicolor.urdf",
    #     usd_path="roboverse_data/assets/bidex/objects/usd/cube_multicolor.usd",
    #     default_density=500.0,
    #     use_vhacd=True,
    # ),
    PrimitiveCubeCfg(
        name="table",
        size=(0.5, 1.0, 0.6),
        disable_gravity=True,
        fix_base_link=True,
        flip_visual_attachments=True,
        friction=0.5,
        color=[0.8, 0.8, 0.8],
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
            # "cube1": {"pos": torch.tensor([0.0, 0.1, 1.612]), "rot": torch.tensor([0.5, 0.5, 0.5, -0.5])},
            # "cube2": {"pos": torch.tensor([0.0, -0.1, 0.725]), "rot": torch.tensor([0.5, 0.5, 0.5, -0.5])},
            "table": {
                "pos": torch.tensor([0, 0.0, 0.3]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            # "door": {
            #     "pos": torch.tensor([0, 0.0, 0.75]),
            #     "rot": torch.tensor([0.0, 0.0, 1.0, 0.0]),
            #     "dof_pos": {
            #         "joint_1": 0.0,
            #         "joint_2": 0.0,
            #     },
            # },
            # "pot": {
            #     "pos": torch.tensor([0, -0.6, 0.765]),
            #     # "pos": torch.tensor([0, -0.6, 0.815]),
            #     "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            #     "dof_pos": {
            #         "joint_0": 0.0,  # Initial position of the switch
            #     },
            # },
            # "cup": {
            #     "pos": torch.tensor([0, 0.0, 0.685]),
            #     "rot": torch.tensor([0.707, 0.0, 0.0, 0.707]),
            #     "dof_pos": {
            #         "joint_0": 0.0,  # Initial position of the switch
            #     },
            # },
            # "scissor": {
            #     "pos": torch.tensor([0, 0.0, 0.6075]),
            #     "rot": torch.tensor([0.707, 0.0, 0.0, 0.707]),
            #     "dof_pos": {
            #         "joint_0": -0.59,  # Initial position of the switch
            #     },
            # },
            # "pen": {
            #     "pos": torch.tensor([0, 0.0, 0.612]),
            #     "rot": torch.tensor([0.5, 0.5, 0.5, -0.5]),
            #     "dof_pos": {
            #         "joint_0": 0.0,  # Initial position of the switch
            #     },
            # },
            # "kettle": {
            #     "pos": torch.tensor([0.0, 0.0, 0.675]),
            #     "rot": torch.tensor([0.707, 0.0, 0.0, 0.707]),
            #     "dof_pos": {
            #         "joint_0": 0.0,
            #     },
            # },
            "bucket": {
                "pos": torch.tensor([0.0, 0.2, 0.355]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
        },
        "robots": {
            "franka_shadow_right": {
                "pos": torch.tensor([0.68, 0.2, 0.0]),
                "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "dof_pos": {
                    "FFJ1": 0.0,
                    "FFJ2": 0.0,
                    "FFJ3": 0.0,
                    "FFJ4": 0.0,
                    "LFJ1": 0.0,
                    "LFJ2": 0.0,
                    "LFJ3": 0.0,
                    "LFJ4": 0.0,
                    "LFJ5": 0.0,
                    "MFJ1": 0.0,
                    "MFJ2": 0.0,
                    "MFJ3": 0.0,
                    "MFJ4": 0.0,
                    "RFJ1": 0.0,
                    "RFJ2": 0.0,
                    "RFJ3": 0.0,
                    "RFJ4": 0.0,
                    "THJ1": 0.0,
                    "THJ2": 0.0,
                    "THJ3": 0.0,
                    "THJ4": 0.0,
                    "THJ5": 0.0,
                    "WRJ1": 0.0,
                    "WRJ2": 0.0,
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.1415928,
                    "panda_joint7": 2.35619445,
                },
            },
            "franka_shadow_left": {
                "pos": torch.tensor([0.68, -0.2, 0.0]),
                "rot": torch.tensor([0, 0, 0, 1]),
                "dof_pos": {
                    "FFJ1": 0.0,
                    "FFJ2": 0.0,
                    "FFJ3": 0.0,
                    "FFJ4": 0.0,
                    "LFJ1": 0.0,
                    "LFJ2": 0.0,
                    "LFJ3": 0.0,
                    "LFJ4": 0.0,
                    "LFJ5": 0.0,
                    "MFJ1": 0.0,
                    "MFJ2": 0.0,
                    "MFJ3": 0.0,
                    "MFJ4": 0.0,
                    "RFJ1": 0.0,
                    "RFJ2": 0.0,
                    "RFJ3": 0.0,
                    "RFJ4": 0.0,
                    "THJ1": 0.0,
                    "THJ2": 0.0,
                    "THJ3": 0.0,
                    "THJ4": 0.0,
                    "THJ5": 0.0,
                    "WRJ1": 0.0,
                    "WRJ2": 0.0,
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 3.1415928,
                    "panda_joint7": -0.785398,
                },
            },
            # "franka_allegro_right": {
            #     "pos": torch.tensor([0.68, 0.2, 0.0]),
            #     "rot": torch.tensor([0.0, 0.0, 0.0, 1.0]),
            #     "dof_pos": {
            #         "joint_0": 0.0,
            #         "joint_1": 0.0,
            #         "joint_2": 0.0,
            #         "joint_3": 0.0,
            #         "joint_4": 0.0,
            #         "joint_5": 0.0,
            #         "joint_6": 0.0,
            #         "joint_7": 0.0,
            #         "joint_8": 0.0,
            #         "joint_9": 0.0,
            #         "joint_10": 0.0,
            #         "joint_11": 0.0,
            #         "joint_12": 0.0,
            #         "joint_13": 0.0,
            #         "joint_14": 1.64,
            #         "joint_15": 0.0,
            #         "panda_joint1": 0.0,
            #         "panda_joint2": -0.785398,
            #         "panda_joint3": 0.0,
            #         "panda_joint4": -2.356194,
            #         "panda_joint5": 0.0,
            #         "panda_joint6": 3.1415928,
            #         "panda_joint7": -2.356194,
            #     },
            # },
            # "franka_allegro_left": {
            #     "pos": torch.tensor([0.68, -0.2, 0.0]),
            #     "rot": torch.tensor([0, 0, 0, 1]),
            #     "dof_pos": {
            #         "joint_0": 0.0,
            #         "joint_1": 0.0,
            #         "joint_2": 0.0,
            #         "joint_3": 0.0,
            #         "joint_4": 0.0,
            #         "joint_5": 0.0,
            #         "joint_6": 0.0,
            #         "joint_7": 0.0,
            #         "joint_8": 0.0,
            #         "joint_9": 0.0,
            #         "joint_10": 0.0,
            #         "joint_11": 0.0,
            #         "joint_12": 0.0,
            #         "joint_13": 0.0,
            #         "joint_14": 1.64,
            #         "joint_15": 0.0,
            #         "panda_joint1": 0.0,
            #         "panda_joint2": -0.785398,
            #         "panda_joint3": 0.0,
            #         "panda_joint4": -2.356194,
            #         "panda_joint5": 0.0,
            #         "panda_joint6": 3.1415928,
            #         "panda_joint7": -2.356194,
            #     },
            # },
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
obj_name = "bucket"


## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/2_add_new_robot_{args.sim}.mp4")
obs_saver.add(obs)
start_time = time.time()
step = 0
for _ in range(50):
    # while True:
    # log.debug(f"Step {step}")
    # left_pos = scenario.robots[0].wrist_pos
    # left_pos_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    # left_pos_err[:, 0] = -0.55 - left_pos[:, 0]
    # left_pos_err[:, 1] = -0.2 - left_pos[:, 1]
    # left_pos_err[:, 2] = 0.6 - left_pos[:, 2]
    # print("left_pos:", left_pos)
    # left_rot = scenario.robots[0].wrist_rot
    # print("left_rot:", left_rot)
    # print("left_dof:", scenario.robots[0].dof_pos)
    # left_target_rot = torch.tensor([-1, 0.0, 0.0, 0.0], device="cuda:0").unsqueeze(0).repeat(args.num_envs, 1)
    # left_rot_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    # # left_rot_err = quat_mul(left_target_rot, quat_inv(left_rot))
    # # left_rot_err = quat_to_rotvec(left_rot_err)
    # left_dpose = torch.cat([left_pos_err, left_rot_err], -1)
    # left_targets = scenario.robots[0].control_arm_ik(left_dpose, args.num_envs, "cuda:0")
    # print("left_targets:", left_targets)

    # right_pos = scenario.robots[1].wrist_pos
    # right_rot = scenario.robots[1].wrist_rot
    # right_pos_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    # right_pos_err[:, 1] = 0.0
    # right_pos_err[:, 2] = 0.6 - right_pos[:, 2]
    # right_rot_err = torch.zeros((args.num_envs, 3), device="cuda:0")
    # right_dpose = torch.cat([right_pos_err, right_rot_err], -1)
    # right_targets = scenario.robots[1].control_arm_ik(right_dpose, args.num_envs, "cuda:0")
    # print("right_pos:", right_pos)
    # print("right_rot:", right_rot)
    # print("right_dof:", scenario.robots[1].dof_pos)
    # print("right_targets:", right_targets)

    # left_ft_pos_err = torch.zeros((args.num_envs, 4, 3), device="cuda:0")
    # left_ft_pos_err[..., 1] = 0.0
    # right_ft_pos_err = torch.zeros((args.num_envs, 4, 3), device="cuda:0")
    # right_ft_pos_err[..., 1] = -0.0
    # left_ft_rot_err = torch.zeros((args.num_envs, 4, 3), device="cuda:0")
    # right_ft_rot_err = torch.zeros((args.num_envs, 4, 3), device="cuda:0")
    # left_ft_pos_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    # left_ft_pos_err[..., 1] = 0.0
    # right_ft_pos_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    # right_ft_pos_err[..., 1] = -0.0
    # left_ft_rot_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    # right_ft_rot_err = torch.zeros((args.num_envs, 5, 3), device="cuda:0")
    # left_dof_pos = scenario.robots[0].control_hand_ik(left_ft_pos_err, left_ft_rot_err)
    # right_dof_pos = scenario.robots[1].control_hand_ik(right_ft_pos_err, right_ft_rot_err)

    actions = torch.zeros((args.num_envs, dof_num), device="cuda:0")
    num_dof = 0
    ###############################################
    # scissor_pos = obs.objects[obj_name].root_state[:, :3]
    # scissor_rot = obs.objects[obj_name].root_state[:, 3:7]
    # r_link_idx = obs.objects[obj_name].body_names.index("link_1")
    # r_link_pos = obs.objects[obj_name].body_state[:, r_link_idx, :3]
    # print("r_link pos:", r_link_pos)
    # l_link_idx = obs.objects[obj_name].body_names.index("link_1")
    # l_link_pos = obs.objects[obj_name].body_state[:, l_link_idx, :3]
    # print("l_link pos:", l_link_pos)
    # x_unit_tensor = torch.tensor([1.0, 0.0, 0.0], device="cuda:0").unsqueeze(0)
    # y_unit_tensor = torch.tensor([0.0, 1.0, 0.0], device="cuda:0").unsqueeze(0)
    # z_unit_tensor = torch.tensor([0.0, 0.0, 1.0], device="cuda:0").unsqueeze(0)
    # scissor_right_handle_pos = r_link_pos
    # scissor_right_handle_rot = obs.objects[obj_name].body_state[:, r_link_idx, 3:7]
    # scissor_right_handle_pos = scissor_right_handle_pos + quat_apply(scissor_right_handle_rot, x_unit_tensor * 0.15)
    # scissor_right_handle_pos = scissor_right_handle_pos + quat_apply(scissor_right_handle_rot, z_unit_tensor * -0.1)
    # scissor_left_handle_pos = l_link_pos
    # scissor_left_handle_rot = obs.objects[obj_name].body_state[:, l_link_idx, 3:7]
    # scissor_left_handle_pos = scissor_left_handle_pos + quat_apply(scissor_left_handle_rot, x_unit_tensor * -0.2)
    # scissor_left_handle_pos = scissor_left_handle_pos + quat_apply(scissor_left_handle_rot, z_unit_tensor * -0.1)
    # print("scissor_right_handle_pos:", scissor_right_handle_pos)
    # print("scissor_left_handle_pos:", scissor_left_handle_pos)
    # print("scissor_pos:", scissor_pos)
    ###################################################
    # r_link_idx = obs.objects[obj_name].body_names.index("link_0")
    # r_link_pos = obs.objects[obj_name].body_state[:, r_link_idx, :3]
    # print("r_link pos:", r_link_pos)
    # l_link_idx = obs.objects[obj_name].body_names.index("base")
    # l_link_pos = obs.objects[obj_name].body_state[:, l_link_idx, :3]
    # print("l_link pos:", l_link_pos)
    # scissor_right_handle_pos = r_link_pos
    # scissor_right_handle_rot = obs.objects[obj_name].body_state[:, r_link_idx, 3:7]
    # scissor_left_handle_pos = l_link_pos
    # scissor_left_handle_rot = obs.objects[obj_name].body_state[:, l_link_idx, 3:7]
    # scissor_right_handle_pos = obs.objects[obj_name].root_state[:, :3]
    # scissor_right_handle_rot = obs.objects[obj_name].root_state[:, 3:7]
    # scissor_left_handle_pos = scissor_right_handle_pos
    # scissor_left_handle_rot = scissor_right_handle_rot
    # right_pos = torch.tensor([0.0, 0.2, 0.655], device="cuda:0").unsqueeze(0).repeat(args.num_envs, 1)
    # left_pos = torch.tensor([0.0, 0.2, 0.655], device="cuda:0").unsqueeze(0).repeat(args.num_envs, 1)
    # right_offset = quat_apply(quat_inv(scissor_right_handle_rot), right_pos - scissor_right_handle_pos)
    # left_offset = quat_apply(quat_inv(scissor_left_handle_rot), left_pos - scissor_left_handle_pos)
    # print("right_offset:", right_offset)
    # print("left_offset:", left_offset)
    # print("scissor pos:", obs.objects[obj_name].root_state[:, :3])
    ################################################
    # for robot in scenario.robots:
    #     arm_dof_idx = [i + num_dof for i in robot.arm_dof_idx]
    #     hand_dof_idx = [i + num_dof for i in robot.hand_dof_idx]
    #     actions[:, arm_dof_idx] = left_targets if robot.name == "franka_allegro_left" else right_targets
    #     actions[:, hand_dof_idx] = left_dof_pos if robot.name == "franka_allegro_left" else right_dof_pos
    # actions[:, arm_dof_idx] = left_targets if robot.name == "franka_shadow_left" else right_targets
    # actions[:, hand_dof_idx] = left_dof_pos if robot.name == "franka_shadow_left" else right_dof_pos
    # num_dof += num_robo_dof[robot.name]
    obs, reward, success, time_out, extras = env.step(actions)
    for robot in scenario.robots:
        robot.update_state(obs)
    obs_saver.add(obs)
    step += 1
    if step % 10 == 0:
        log.info(f"Step {step}, Time Elapsed: {time.time() - start_time:.2f} seconds")
        start_time = time.time()

obs_saver.save()

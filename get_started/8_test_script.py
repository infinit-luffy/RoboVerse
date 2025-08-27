"""This script provides a minimal example of loading dexterous hand."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import time

from get_started.utils import ObsSaver
from metasim.cfg.objects import RigidObjCfg
from metasim.cfg.robots import ShadowHandCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.cfg.simulator_params import SimParamCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""

    ## Handlers
    # TODO currently, only support for isaacgym. Adding support for other simulators.
    sim: Literal["isaaclab", "isaacgym", "genesis", "pyrep", "pybullet", "sapien", "sapien3", "mujoco", "blender"] = (
        "isaacgym"
    )

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)
sim_params = SimParamCfg()
sim_params.dt = 1.0 / 60.0
sim_params.bounce_threshold_velocity = 0.2
sim_params.contact_offset = 0.002
sim_params.num_velocity_iterations = 0
sim_params.num_threads = 4
sim_params.use_gpu_pipeline = True
sim_params.use_gpu = True
sim_params.substeps = 2


# initialize scenario
scenario = ScenarioCfg(
    robots=[
        ShadowHandCfg(name="shadow_hand_right", angular_damping=0.01, fix_base_link=False),
        ShadowHandCfg(name="shadow_hand_left", angular_damping=0.01, fix_base_link=False),
    ],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
    sim_params=sim_params,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(-1.5, -1.5, 1.5), look_at=(0.0, -0.5, 0.6))]

scenario.objects = [
    RigidObjCfg(
        name="cube",
        scale=(1, 1, 1),
        physics=PhysicStateType.RIGIDBODY,
        urdf_path="roboverse_data/assets/bidex/objects/urdf/cube_multicolor.urdf",
        usd_path="roboverse_data/assets/bidex/objects/usd/cube_multicolor/object.usd",
        default_density=500.0,
        use_vhacd=True,
    ),
    # ArticulationObjCfg(
    #     name="switch1",
    #     fix_base_link=True,
    #     # disable_gravity=True,
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/switch_mobility.urdf",
    # ),
    # ArticulationObjCfg(
    #     name="switch2",
    #     fix_base_link=True,
    #     # disable_gravity=True,
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/switch_mobility.urdf",
    # ),
    # ArticulationObjCfg(
    #     name="scissor",
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/scissor_mobility.urdf",
    #     default_density=500.0,
    #     friction=1.0,
    #     use_vhacd=False,
    #     fix_base_link=False,
    # ),
    # ArticulationObjCfg(
    #     name="bottle",
    #     fix_base_link=False,
    #     disable_gravity=True,
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/bottle_mobility.urdf",
    # ),
    # RigidObjCfg(
    #     name="cube2",
    #     physics=PhysicStateType.RIGIDBODY,
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/cube_multicolor.urdf"
    # ),
    # ArticulationObjCfg(
    #     name="pen",
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/pen_mobility.urdf",
    #     default_density=500.0,
    #     friction=1.0,
    # ),
    # ArticulationObjCfg(
    #     name="bucket",
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/bucket_mobility.urdf",
    #     default_density=500.0,
    #     use_vhacd=True,
    #     override_com=True,
    #     override_inertia=True,
    #     use_mesh_materials=True,
    #     mesh_normal_mode="vertex",
    # ),
    # ArticulationObjCfg(
    #     name="kettle",
    #     urdf_path="roboverse_data/assets/bidex/objects/urdf/kettle_mobility.urdf",
    #     default_density=500.0,
    #     use_vhacd=True,
    #     override_com=True,
    #     override_inertia=True,
    #     use_mesh_materials=True,
    #     mesh_normal_mode="vertex",
    #     vhacd_resolution=400000,  # Use a higher resolution for better collision detection
    #     friction=1.0,
    # ),
    # PrimitiveCubeCfg(
    #     name="table",
    #     size=(0.5, 1.0, 0.5),
    #     disable_gravity=True,
    #     fix_base_link=True,
    #     flip_visual_attachments=True,
    #     physics=PhysicStateType.RIGIDBODY,
    #     color=[0.8, 0.8, 0.8],
    # )
]

log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            # "switch1": {
            #     "pos": torch.tensor([0.0, 0.2, 0.65]),
            #     "rot": torch.tensor([0, -0.7071, 0, 0.7071]),
            #     "dof_pos": {
            #         "joint_0": 0.5585,  # Initial position of the switch
            #     }
            # },
            # "switch2": {
            #     "pos": torch.tensor([0.0, -0.2, 0.65]),
            #     "rot": torch.tensor([0, -0.7071, 0, 0.7071]),
            #     "dof_pos": {
            #         "joint_0": 0.5585,  # Initial position of the switch
            #     }
            # },
            "cube": {
                "pos": torch.tensor([0.0, -0.55, 0.54]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            # "cube2": {
            #     "pos": torch.tensor([0.0, -0.2, 0.6]),
            #     "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            # },
            # "scissor": {
            #     "pos": torch.tensor([0.0, 0.0, 0.6075]),
            #     "rot": torch.tensor([0.707, 0.0, 0.0, 0.707]),
            #     "dof_pos": {
            #         "joint_0": -0.59,  # Initial position of the switch
            #     }
            # },
            # "cup": {
            #     "pos": torch.tensor([0.0, 0.2, 0.8]),
            #     "rot": torch.tensor([0, -0.7071, 0, 0.7071]),
            #     "dof_pos": {
            #         "joint_0": 0.0,  # Initial position of the switch
            #     }
            # },
            # "bottle": {
            #     "pos": torch.tensor([0, -0.6, 0.7]),
            #     "rot": torch.tensor([0.945, 0.0, -0.327, 0.018]),
            #     "dof_pos": {
            #         "joint_0": 0.0,
            #         "joint_2": 0.0,  # Initial position of the switch
            #     }
            # },
            "table": {
                "pos": torch.tensor([0.0, 0.0, 0.275]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            # "kettle": {
            #     "pos": torch.tensor([0.0, 0.0, 0.5]),
            #     "rot": torch.tensor([0.707, 0.0, 0.0, 0.707]),
            # },
            # "bucket": {
            #     "pos": torch.tensor([0.0, -0.3, 0.5]),
            #     "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            # },
        },
        "robots": {
            "shadow_hand_right": {
                "pos": torch.tensor([0.0, 0.0, 0.5]),
                "rot": torch.tensor([0.0, 0.0, -0.707, 0.707]),
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
                "pos": torch.tensor([0.0, -1.15, 0.5]),
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
    for _ in range(args.num_envs)
]
obs, extras = env.reset(states=init_states)
os.makedirs("get_started/output", exist_ok=True)

## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/8_shadowhand_loading_{args.sim}.mp4")
obs_saver.add(obs, single_env=True)

step = 0
robot_joint_limits = {}
for robot in scenario.robots:
    robot_joint_limits.update(robot.joint_limits)
start_time = time.time()
for _ in range(100):
    log.debug(f"Step {step}")
    actions = [
        {
            robot.name: {
                "dof_pos_target": {
                    # joint_name: (
                    #     np.random.rand()  # Randomly sample joint positions
                    #     * (robot.joint_limits[joint_name][1] - robot.joint_limits[joint_name][0])
                    #     + robot.joint_limits[joint_name][0]
                    # )
                    joint_name: 0.0
                    for joint_name in robot.joint_limits.keys()
                    if robot.actuators[joint_name].fully_actuated
                }
            }
            for robot in scenario.robots
        }
        for _ in range(scenario.num_envs)
    ]
    # print(actions[0]["dof_pos_target"])
    # from ipdb import set_trace
    # set_trace()  # Debugging point to inspect actions
    obs, reward, success, time_out, extras = env.step(actions)
    obs_saver.add(obs, single_env=True)
    step += 1
    if step % 10 == 0:
        log.info(f"Step {step}, Time Elapsed: {time.time() - start_time:.2f} seconds")
        start_time = time.time()

obs_saver.save()
exit(0)

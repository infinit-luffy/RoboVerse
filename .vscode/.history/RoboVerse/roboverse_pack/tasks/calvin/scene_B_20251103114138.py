from __future__ import annotations

import pickle
import xml.etree.ElementTree as ET

import gymnasium as gym

from metasim.scenario.objects import ArticulationObjCfg, RigidObjCfg
from metasim.scenario.robot import BaseActuatorCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.base import BaseTaskEnv
from metasim.task.registry import register_task
from metasim.utils.ik_solver import setup_ik_solver
from metasim.utils.math import quat_from_euler_np
from metasim.utils.tensor_util import array_to_tensor
from roboverse_pack.robots.franka_with_gripper_extension_cfg import FrankaWithGripperExtensionCfg

all_joint_names = {
    "franka": [
        "panda_finger_joint1",
        "panda_finger_joint2",
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
    "table": ["base__button", "base__switch", "base__slide", "base__drawer"],
}


@register_task("calvin.base_table_B")
class BaseCalvinTableTask_B(BaseTaskEnv):
    scenario = ScenarioCfg(
        robots=[
            FrankaWithGripperExtensionCfg(
                name="franka",
                default_position=[-0.34, -0.46, 0.24],
                default_orientation=[1, 0, 0, 0],
                actuators={
                    "panda_joint1": BaseActuatorCfg(velocity_limit=2.175, torque_limit=87, stiffness=280, damping=10),
                    "panda_joint2": BaseActuatorCfg(velocity_limit=2.175, torque_limit=87, stiffness=280, damping=10),
                    "panda_joint3": BaseActuatorCfg(velocity_limit=2.175, torque_limit=87, stiffness=280, damping=10),
                    "panda_joint4": BaseActuatorCfg(velocity_limit=2.175, torque_limit=87, stiffness=280, damping=10),
                    "panda_joint5": BaseActuatorCfg(velocity_limit=2.61, torque_limit=12.0, stiffness=200, damping=5),
                    "panda_joint6": BaseActuatorCfg(velocity_limit=2.61, torque_limit=12.0, stiffness=200, damping=5),
                    "panda_joint7": BaseActuatorCfg(velocity_limit=2.61, torque_limit=12.0, stiffness=200, damping=5),
                    "panda_finger_joint1": BaseActuatorCfg(
                        velocity_limit=0.2, torque_limit=20.0, is_ee=True, stiffness=30000, damping=1000
                    ),
                    "panda_finger_joint2": BaseActuatorCfg(
                        velocity_limit=0.2, torque_limit=20.0, is_ee=True, stiffness=30000, damping=1000
                    ),
                },
                default_joint_positions={
                    "panda_joint1": -1.21779206,
                    "panda_joint2": 1.03987646,
                    "panda_joint3": 2.11978261,
                    "panda_joint4": -2.34205014,
                    "panda_joint5": -0.87015947,
                    "panda_joint6": 1.64119353,
                    "panda_joint7": 0.55344866,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
                control_type="joint_position",
                fix_base_link=True,
                urdf_path="/home/dyz/RoboVerse/calvin/calvin_env/calvin_env/data/franka_panda/panda_longer_finger.urdf",
                usd_path=None,
                mjcf_path=None,
                mjx_mjcf_path=None,
            )
        ],
        objects=[
            ArticulationObjCfg(
                name="table",
                scale=0.8,
                default_position=[0, 0, 0],
                default_orientation=[1, 0, 0, 0],
                fix_base_link=True,
                urdf_path="/home/dyz/RoboVerse/calvin/calvin_env/calvin_env/data/calvin_table_B/urdf/calvin_table_B.urdf",
            ),
            RigidObjCfg(
                name="pink_cube",
                scale=0.8,
                default_position=[1.28661989e-01, -3.77756105e-02, 4.59989266e-01 + 0.01],
                default_orientation=quat_from_euler_np(1.10200730e-04, 3.19760378e-05, -3.94522179e-01),
                fix_base_link=False,
                urdf_path="/home/dyz/RoboVerse/calvin/calvin_env/calvin_env/data/blocks/block_pink_middle.urdf",
            ),
            RigidObjCfg(
                name="blue_cube",
                scale=0.8,
                default_position=[-2.83642665e-01, 8.05351014e-02, 4.60989238e-01 + 0.01],
                default_orientation=quat_from_euler_np(-1.10251078e-05, -5.25663348e-05, -9.06438129e-01),
                fix_base_link=False,
                urdf_path="/home/dyz/RoboVerse/calvin/calvin_env/calvin_env/data/blocks/block_blue_big.urdf",
            ),
            RigidObjCfg(
                name="red_cube",
                scale=0.8,
                default_position=[2.32403619e-01, -4.04295856e-02, 4.59990009e-01 + 0.01],
                default_orientation=quat_from_euler_np(4.12287744e-08, -8.05700103e-09, -2.17741510e00),
                fix_base_link=False,
                urdf_path="/home/dyz/RoboVerse/calvin/calvin_env/calvin_env/data/blocks/block_red_small.urdf",
            ),
        ],
        decimation=8,
    )

    traj_filepath = "roboverse_pack/tasks/calvin/env_B_out/episode_chunk_36_632862_639064/trajectory_env_B_4370_v2.pkl"

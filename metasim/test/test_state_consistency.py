import math

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass



# from isaaclab.app import AppLauncher

# launch omniverse app
# simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch

# import isaacsim.core.utils.stage as stage_utils
# import pytest
# from isaacsim.core.api.simulation_context import SimulationContext

# import isaaclab.sim as sim_utils
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
# from isaaclab.utils.math import random_orientation
# from isaaclab.utils.timer import Timer

import pytest
import rootutils
import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
# from metasim.sim.sim_context import HandlerContext
from metasim.utils.state import state_tensor_to_nested

rootutils.setup_root(__file__, pythonpath=True)
from roboverse_pack.robots.franka_cfg import FrankaCfg


def assert_close(a, b, atol=1e-3):
    if isinstance(a, torch.Tensor):
        assert torch.allclose(a, b, atol=atol), f"a: {a} != b: {b}"
    elif isinstance(a, float):
        assert math.isclose(a, b, abs_tol=atol), f"a: {a} != b: {b}"
    else:
        raise ValueError(f"Unsupported type: {type(a)}")


def get_test_parameters():
    """Generate test parameters with different num_envs for different simulators."""
    # MuJoCo only supports num_envs=1 due to simulator limitations
    # Other simulators can test with multiple environments
    isaacsim_params = [("isaacsim", num_envs) for num_envs in [1, 2, 4]]
    isaacgym_params = [("isaacgym", num_envs) for num_envs in [1, 2, 4]]
    genesis_params = [("genesis", num_envs) for num_envs in [1, 2, 4]]
    mujoco_params = [("mujoco", 1)]
    mujoco_params = [("sapien3", 1)]
    mujoco_params = [("pybullet", 1)]
    return mujoco_params + isaacsim_params + isaacgym_params + genesis_params


@pytest.mark.parametrize("sim,num_envs", get_test_parameters())
def test_consistency(sim, num_envs):
    # print(f"Testing {sim} with {num_envs} envs")
    scenario = ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveCubeCfg(
                name="cube", size=(0.1, 0.1, 0.1), color=[1.0, 0.0, 0.0], physics=PhysicStateType.RIGIDBODY
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
        ],
        robots=[FrankaCfg()],
    )
    init_states = [
        {
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.3, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "sphere": {
                    "pos": torch.tensor([0.4, -0.6, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "bbq_sauce": {
                    "pos": torch.tensor([0.7, -0.3, 0.14]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "box_base": {
                    "pos": torch.tensor([0.5, 0.2, 0.1]),
                    "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                    "dof_pos": {"box_joint": 0.0},
                },
            },
            "robots": {
                "franka": {
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785398,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356194,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.570796,
                        "panda_joint7": 0.785398,
                        "panda_finger_joint1": 0.04,
                        "panda_finger_joint2": 0.04,
                    },
                },
            },
        }
    ] * num_envs

    # with HandlerContext(scenario) as handler:
    from metasim.utils.setup_util import get_sim_handler_class
    from metasim.constants import SimType
    # env = get_sim_handler_class(SimType(sim))(scenario)
    env_class = get_sim_handler_class(SimType(sim))
    env = env_class(scenario)
    env.launch()
    env.set_states(init_states)
    states = state_tensor_to_nested(env, env.get_states())
    for i in range(num_envs):
        assert_close(states[i]["objects"]["cube"]["pos"], init_states[i]["objects"]["cube"]["pos"])
        assert_close(states[i]["objects"]["sphere"]["pos"], init_states[i]["objects"]["sphere"]["pos"])
        assert_close(states[i]["objects"]["bbq_sauce"]["pos"], init_states[i]["objects"]["bbq_sauce"]["pos"])
        assert_close(states[i]["objects"]["box_base"]["pos"], init_states[i]["objects"]["box_base"]["pos"])
        assert_close(states[i]["objects"]["box_base"]["rot"], init_states[i]["objects"]["box_base"]["rot"])
        assert_close(states[i]["robots"]["franka"]["pos"], init_states[i]["robots"]["franka"]["pos"])
        assert_close(states[i]["robots"]["franka"]["rot"], init_states[i]["robots"]["franka"]["rot"])
        assert_close(
            states[i]["objects"]["box_base"]["dof_pos"]["box_joint"],
            init_states[i]["objects"]["box_base"]["dof_pos"]["box_joint"],
        )
        for k in states[i]["robots"]["franka"]["dof_pos"].keys():
            assert_close(
                states[i]["robots"]["franka"]["dof_pos"][k],
                init_states[i]["robots"]["franka"]["dof_pos"][k],
                )
    env.close()
    # print(f"Testing {sim} with {num_envs} envs passed")


# if __name__ == "__main__":
#     # 直接运行时，可以指定要测试的模拟器和环境数量
#     import sys
    
#     # 默认参数
#     sim = "mujoco"
#     num_envs = 1
    
#     # 从命令行获取参数
#     if len(sys.argv) > 1:
#         sim = sys.argv[1]
#     if len(sys.argv) > 2:
#         num_envs = int(sys.argv[2])
    
#     print(f"Testing {sim} with {num_envs} envs...")
#     test_consistency(sim, num_envs)
#     print(f"✅ Test passed for {sim} with {num_envs} envs!")

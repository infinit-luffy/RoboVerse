try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


# from isaaclab.app import AppLauncher

# launch omniverse app
# simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

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
from loguru import logger as log

from metasim.constants import PhysicStateType
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg

# from metasim.sim.sim_context import HandlerContext
from metasim.utils.state import state_tensor_to_nested

rootutils.setup_root(__file__, pythonpath=True)
from metasim.test.test_utils import assert_close, get_test_parameters
from roboverse_pack.robots.franka_cfg import FrankaCfg
from metasim.scenario.simulator_params import SimParamCfg

@pytest.mark.parametrize("sim,num_envs", get_test_parameters())
def test_consistency(sim, num_envs):
    scenario = ScenarioCfg(
        simulator=sim,
        num_envs=num_envs,
        headless=True,
        objects=[
            PrimitiveCubeCfg(
                name="cube",
                size=(0.1, 0.1, 0.1),
                color=[1.0, 0.0, 0.0],
                physics=PhysicStateType.RIGIDBODY,
                default_position=[0, 0, 10.0]
            ),
        ],
        robots=[FrankaCfg()],
        sim_params=SimParamCfg(
            dt=0.001
        ),
    )

    from metasim.constants import SimType
    from metasim.utils.setup_util import get_sim_handler_class

    env_class = get_sim_handler_class(SimType(sim))
    env = env_class(scenario)
    env.launch()

    env.close()


if __name__ == "__main__":
    # 直接运行时，可以指定要测试的模拟器和环境数量
    import sys

    # 默认参数
    sim = "mujoco"
    num_envs = 1

    # 从命令行获取参数
    if len(sys.argv) > 1:
        sim = sys.argv[1]
    if len(sys.argv) > 2:
        num_envs = int(sys.argv[2])

    log.info(f"Testing {sim} with {num_envs} envs...")
    test_consistency(sim, num_envs)
    log.success(f"✅ Test passed for {sim} with {num_envs} envs!")

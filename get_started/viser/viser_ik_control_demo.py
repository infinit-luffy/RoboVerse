"""This script demonstrates IK control using viser visualization."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    RigidObjCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.hf_util import check_and_download_recursive
from metasim.utils.setup_util import get_handler


@configclass
class Args:
    """Arguments for the IK control demo."""

    robot: str = "franka"

    ## Handlers
    sim: Literal[
        "isaacsim",
        "isaacgym",
        "isaaclab",
        "genesis",
        "pybullet",
        "sapien2",
        "sapien3",
        "mujoco",
    ] = "mujoco"

    ## Others
    num_envs: int = 1
    headless: bool = True  # Use viser for visualization, not simulator's viewer
    solver: Literal["curobo", "pyroki"] = "pyroki"

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


def extract_states_from_init(init_states, key):
    """
    key: "objects" or "robots"
    Return: dict[name] = {"pos": ..., "rot": ..., "dof_pos": ...}
    """
    result = {}
    if init_states and len(init_states) > 0:
        state = init_states[0]
        if key in state:
            for name, item in state[key].items():
                state_dict = {}
                if "pos" in item and item["pos"] is not None:
                    state_dict["pos"] = (
                        item["pos"].cpu().numpy().tolist() if hasattr(item["pos"], "cpu") else list(item["pos"])
                    )
                if "rot" in item and item["rot"] is not None:
                    state_dict["rot"] = (
                        item["rot"].cpu().numpy().tolist() if hasattr(item["rot"], "cpu") else list(item["rot"])
                    )
                if "dof_pos" in item and item["dof_pos"] is not None:
                    state_dict["dof_pos"] = item["dof_pos"]
                result[name] = state_dict
    return result


def main():
    args = tyro.cli(Args)

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )

    # add cameras
    scenario.cameras = [
        PinholeCameraCfg(
            name="camera",
            width=1024,
            height=1024,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
        )
    ]

    # add minimal objects for visual reference
    scenario.objects = [
        PrimitiveCubeCfg(
            name="reference_cube",
            size=(0.05, 0.05, 0.05),
            color=[0.0, 1.0, 0.0],  # Green for reference
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
                "reference_cube": {
                    "pos": torch.tensor([0.5, -0.2, 0.05]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
                "bbq_sauce": {
                    "pos": torch.tensor([0.6, -0.4, 0.14]),
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
    ]

    handler.set_states(init_states * scenario.num_envs)

    # ========================================================================
    # viser visualization with IK control
    # ========================================================================
    from get_started.viser.viser_util import ViserVisualizer

    # ========================================================================
    # Download URDF files before visualization
    # ========================================================================
    def download_urdf_files(scenario):
        """Download URDF files for all objects and robots in the scenario."""
        urdf_paths = []

        # Collect URDF paths from objects
        for obj in scenario.objects:
            if hasattr(obj, "urdf_path") and obj.urdf_path:
                urdf_paths.append(obj.urdf_path)

        # Collect URDF paths from robots
        for robot in scenario.robots:
            if hasattr(robot, "urdf_path") and robot.urdf_path:
                urdf_paths.append(robot.urdf_path)

        # Download URDF files and all related mesh files recursively
        if urdf_paths:
            log.info(f"Downloading {len(urdf_paths)} URDF files and all related meshes...")
            check_and_download_recursive(urdf_paths, n_processes=16)
            log.info("URDF files and meshes download completed!")

    # Download URDF files before visualization
    download_urdf_files(scenario)

    # initialize the viser server
    visualizer = ViserVisualizer(port=8080)
    visualizer.add_grid()
    visualizer.add_frame("/world_frame")

    # extract states from objects and robots
    default_object_states = extract_states_from_init(init_states, "objects")
    default_robot_states = extract_states_from_init(init_states, "robots")

    # visualize all objects and robots
    visualizer.visualize_scenario_items(scenario.objects, default_object_states)
    visualizer.visualize_scenario_items(scenario.robots, default_robot_states)

    # scene info string
    scene_info = ["The IK control demo includes:"]
    for obj in scenario.objects:
        scene_info.append(f"  • {obj.name} ({type(obj).__name__})")
    for robot in scenario.robots:
        scene_info.append(f"  • {robot.name} ({type(robot).__name__})")

    # print for debugging
    log.info("\n".join(scene_info))

    # Setup IK solver for the robot
    robot_config = scenario.robots[0]
    robot_name = robot_config.name
    success = visualizer.setup_ik_solver(robot_name, robot_config, handler)

    if success:
        log.info(f"IK solver successfully setup for robot {robot_name}")
    else:
        log.error(f"Failed to setup IK solver for robot {robot_name}")
        return

    # Enable camera controls
    visualizer.enable_camera_controls(
        initial_position=[1.5, -1.5, 1.5],
        render_width=1024,
        render_height=1024,
        look_at_position=[0, 0, 0],
        initial_fov=71.28,
    )

    # Enable IK control
    visualizer.enable_ik_control()

    # Optional: Enable joint control as well for comparison
    visualizer.enable_joint_control()

    log.info("IK Control Demo is ready!")
    log.info("Instructions:")
    log.info("1. Open http://localhost:8080 in your browser")
    log.info("2. In the 'IK Control' panel, click 'Setup IK Control'")
    log.info("3. Adjust the target position sliders (X, Y, Z)")
    log.info("4. Adjust the target orientation sliders (Quat W, X, Y, Z)")
    log.info("5. Watch the RED SPHERE and RGB AXES move - they show target position & orientation")
    log.info("6. Click 'Solve & Apply IK' to move the robot to the target")
    log.info("7. You can also use the 'Joint Control' panel for direct joint control")
    log.info("8. The green cube is just a reference object in the scene")
    log.info("")
    log.info("Visual Guide:")
    log.info("  • RED SPHERE = Target position")
    log.info("  • RGB AXES at target = Target orientation (Red=X, Green=Y, Blue=Z)")

    # Keep the demo running
    while True:
        pass


if __name__ == "__main__":
    main()

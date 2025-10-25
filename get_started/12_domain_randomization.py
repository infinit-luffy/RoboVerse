"""Domain Randomization Demo with Trajectory Replay - Final Version

Replays close_box task trajectories with domain randomization applied.

## Scene Setup (ALL LEVELS):
- Table height: 0.7m
- Objects placed ON table surface (z = 0.7 + object_height/2)
- Room: 10m x 10m x 5m (walls enclose scene)
- Table: 1.8m x 1.8m (smaller than room)
- Lights: VERY BRIGHT (enclosed room)
- SceneRandomizer creates table WITH PHYSICS COLLISION

## Randomization Levels:
- Level 0: No randomization (baseline) - Scene with FIXED textured materials
- Level 1: Object properties (physics + pose) - Scene with FIXED textured materials
- Level 2: + Visual randomization (scene materials + lights) - Scene materials RANDOMIZE
- Level 3: + Camera randomization - Scene materials RANDOMIZE
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import os
import time
from typing import Literal

import numpy as np
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.randomization import (
    CameraPresets,
    CameraRandomizer,
    LightPresets,
    LightRandomizer,
    MaterialPresets,
    MaterialRandomizer,
    ObjectPresets,
    ObjectRandomizer,
    SceneRandomizer,
)
from metasim.randomization.presets.scene_presets import ScenePresets
from metasim.randomization.scene_randomizer import SceneMaterialPoolCfg
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg, SphereLightCfg
from metasim.task.registry import get_task_class
from metasim.utils import configclass
from metasim.utils.demo_util import get_traj
from metasim.utils.obs_utils import ObsSaver

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


###########################################################
## Utils for Trajectory Replay
###########################################################
def get_actions(all_actions, action_idx: int, num_envs: int):
    """Get actions for all environments at a given step."""
    envs_actions = all_actions[:num_envs]
    actions = [
        env_actions[action_idx] if action_idx < len(env_actions) else env_actions[-1] for env_actions in envs_actions
    ]
    return actions


def get_runout(all_actions, action_idx: int):
    """Check if all trajectories have run out of actions."""
    runout = all([action_idx >= len(all_actions[i]) for i in range(len(all_actions))])
    return runout


def create_env(args):
    """Create task environment."""
    task_name = "close_box"
    task_cls = get_task_class(task_name)

    table_height = 0.7

    # Camera - adjust for table
    camera = PinholeCameraCfg(
        name="main_camera",
        width=1024,
        height=1024,
        pos=(2.0, -2.0, 2.0),  # Higher and further back for table view
        look_at=(0.0, 0.0, table_height + 0.05),  # Look at objects on table
    )

    # Lights - CRITICAL: Enclosed room needs MUCH more light
    # All levels have enclosed room, use bright lighting
    lights = [
        DistantLightCfg(
            name="main_light",
            intensity=10000.0,  # 10x brighter for enclosed room
            color=(1.0, 1.0, 0.98),
            polar=30.0,
            azimuth=45.0,
            is_global=True,
        ),
        SphereLightCfg(
            name="fill_light",
            intensity=8000.0,  # Very bright fill
            color=(1.0, 1.0, 1.0),
            radius=0.6,
            pos=(2.0, 2.0, 3.5),  # Inside room (walls at ±5m)
            is_global=True,  # IMPORTANT: Penetrate walls for enclosed room
        ),
        SphereLightCfg(
            name="back_light",
            intensity=6000.0,  # Strong back light
            color=(0.98, 0.98, 1.0),
            radius=0.5,
            pos=(-2.0, -2.0, 3.5),  # Inside room, opposite corner
            is_global=True,  # IMPORTANT: Penetrate walls for enclosed room
        ),
        SphereLightCfg(
            name="table_light",
            intensity=5000.0,  # Direct table illumination
            color=(1.0, 1.0, 0.95),
            radius=0.4,
            pos=(0.0, 0.0, 2.5),  # Centered above table
            is_global=True,  # IMPORTANT: Penetrate walls for enclosed room
        ),
    ]

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        lights=lights,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)

    return env


def get_init_states(level, num_envs):
    """Get initial states for objects and robot based on level."""

    # Object dimensions
    box_base_height = 0.15  # Approximate height of box_base

    # All levels: Objects ON TABLE
    table_surface_z = 0.7  # Table surface height

    # Objects placed on table surface
    # For box_base: center_z = surface_z + half_height
    objects = {
        "box_base": {
            "pos": torch.tensor([-0.2, 0.0, table_surface_z + box_base_height / 2]),  # 0.775m
            "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
            "dof_pos": {"box_joint": 0.0},
        },
    }

    # Robot also on table
    robot = {
        "franka": {
            "pos": torch.tensor([0.0, -0.4, table_surface_z]),  # Base on table
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
    }

    return [{"objects": objects, "robots": robot}] * num_envs


def initialize_randomizers(handler, args):
    """Initialize all randomizers based on randomization level."""
    randomizers = {
        "object": [],
        "material": [],
        "light": [],
        "camera": [],
        "scene": None,
    }

    level = args.level
    log.info("=" * 70)
    log.info(f"Randomization Level: {level}")
    log.info("=" * 70)

    # All levels: Create tabletop workspace scene
    log.info("\n[Scene Setup] Tabletop Workspace")
    log.info("-" * 70)

    # Get base scene configuration
    scene_cfg = ScenePresets.tabletop_workspace(
        room_size=10.0,
        wall_height=5.0,
        table_size=(1.8, 1.8, 0.1),
        table_height=0.7,
    )

    # Level 0-1: Use fixed materials (single material from pool, no randomization)
    # Level 2+: Full material randomization
    if level < 1:
        # Enable material application but use fixed materials (first from pool)
        # if scene_cfg.floor:
        #     scene_cfg.floor.material_randomization = True
        # if scene_cfg.walls:
        #     scene_cfg.walls.material_randomization = True
        # if scene_cfg.ceiling:
        #     scene_cfg.ceiling.material_randomization = True
        # if scene_cfg.table:
        #     scene_cfg.table.material_randomization = True

        # Override material pools with single fixed material
        scene_cfg.floor_materials = SceneMaterialPoolCfg(
            material_paths=["roboverse_data/materials/arnold/Carpet/Carpet_Beige.mdl"],
            selection_strategy="sequential",
        )
        scene_cfg.wall_materials = SceneMaterialPoolCfg(
            material_paths=["roboverse_data/materials/arnold/Masonry/Stucco.mdl"],
            selection_strategy="sequential",
        )
        scene_cfg.ceiling_materials = SceneMaterialPoolCfg(
            material_paths=["roboverse_data/materials/arnold/Architecture/Ceiling_Tiles.mdl"],
            selection_strategy="sequential",
        )
        scene_cfg.table_materials = SceneMaterialPoolCfg(
            material_paths=["roboverse_data/materials/arnold/Wood/Plywood.mdl"],
            selection_strategy="sequential",
        )
        log.info("  [OK] Tabletop workspace (FIXED materials with textures, no randomization)")
    else:
        log.info("  [OK] Tabletop workspace (materials WILL randomize)")

    scene_rand = SceneRandomizer(scene_cfg, seed=args.seed)
    scene_rand.bind_handler(handler)
    randomizers["scene"] = scene_rand

    log.info("    - Room: 10m x 10m x 5m")
    log.info("    - Table: 1.8m x 1.8m at z=0.7m (WITH COLLIDER)")

    # # Level 0+: Object randomization (currently disabled - only box_base used)
    # if level >= 0:
    log.info("\n[Level 1] Object Randomization (Physics + Pose)")
    log.info("-" * 70)
    log.info("  [SKIP] Object randomization (only box_base, no DR needed)")

    box_rand = ObjectRandomizer(
        ObjectPresets.heavy_object("box_base"),
        seed=args.seed,
    )
    box_rand.cfg.pose.rotation_range = (0, 0)
    box_rand.cfg.pose.position_range[2] = (0, 0)
    box_rand.bind_handler(handler)
    # only randomize once for position of box_base
    box_rand()

    # Level 1+: Visual randomization
    # if level >= 1:
    log.info("\n[Level 1] Visual Randomization (Materials + Lights)")
    log.info("-" * 70)

    log.info("  Materials:")
    box_mat = MaterialRandomizer(
        MaterialPresets.wood_object("box_base", use_mdl=True),
        seed=args.seed,
    )
    box_mat.bind_handler(handler)
    randomizers["material"].append(box_mat)
    log.info("    [OK] box_base: wood (MDL)")

    # Light randomization - all 4 lights in enclosed room
    log.info("  Lights:")

    # level 2+: lighting and reflection randomization
    # if level >= 2:
    # Main light: allow full randomization
    main_light_rand = LightRandomizer(
        LightPresets.indoor_ambient("main_light", randomization_mode="combined"),
        seed=args.seed,
    )
    main_light_rand.bind_handler(handler)
    randomizers["light"].append(main_light_rand)

    # Other lights: ONLY randomize intensity and color
    # DO NOT randomize position - walls will block lights if they move outside room
    for light_name in ["fill_light", "back_light", "table_light"]:
        light_rand = LightRandomizer(
            LightPresets.outdoor_daylight(light_name, randomization_mode="intensity_only"),
            seed=args.seed,
        )
        light_rand.bind_handler(handler)
        randomizers["light"].append(light_rand)

    log.info(f"    [OK] {len(randomizers['light'])} lights configured")

    # Level 3+: Camera randomization
    # if level >= 3:
    log.info("\n[Level 3] Camera Randomization (Viewpoint Variation)")
    log.info("-" * 70)

    camera_rand = CameraRandomizer(
        CameraPresets.surveillance_camera("main_camera", randomization_mode="combined"),
        seed=args.seed,
    )
    camera_rand.bind_handler(handler)
    randomizers["camera"].append(camera_rand)
    log.info("  [OK] main_camera: surveillance preset")

    log.info("\n" + "=" * 70)
    return randomizers


def apply_randomization(randomizers, level):
    """Apply all active randomizers."""

    # Scene randomizer - all levels (but only level 2+ randomize materials)
    if randomizers["scene"]:
        randomizers["scene"]()

    # Object randomization - level 0+
    if level >= 0:
        for rand in randomizers["object"]:
            rand()

    # Material and light randomization - level 1+
    if level >= 1:
        for rand in randomizers["material"]:
            rand()

    if level >= 2:
        for rand in randomizers["light"]:
            rand()

    # Camera randomization - level 3+
    if level >= 3:
        for rand in randomizers["camera"]:
            rand()


def get_states(all_states, action_idx: int, num_envs: int):
    """Get states for all environments at a given step."""
    envs_states = all_states[:num_envs]
    states = [env_states[action_idx] if action_idx < len(env_states) else env_states[-1] for env_states in envs_states]
    return states


def run_replay_with_randomization(env, randomizers, init_state, all_actions, all_states, args):
    """Replay trajectory with periodic randomization.

    Supports two replay modes:
    1. Action-based (default): Replay using actions (allows randomization)
    2. State-based (--object_states): Directly set object states (deterministic)
    """
    os.makedirs("get_started/output", exist_ok=True)
    video_path = f"get_started/output/12_dr_level{args.level}_{args.sim}.mp4"
    obs_saver = ObsSaver(video_path=video_path)

    log.info("\n" + "=" * 70)
    log.info("Running Trajectory Replay with Domain Randomization")
    log.info("=" * 70)
    log.info(f"Replay mode: {'State-based (deterministic)' if args.object_states else 'Action-based (with DR)'}")
    log.info(f"Video: {video_path}")

    if not args.object_states:
        log.info(f"Randomization interval: every {args.randomize_interval} steps")

    traj_length = len(all_actions[0]) if all_actions else (len(all_states[0]) if all_states else 0)
    log.info(f"Trajectory length: {traj_length} steps")

    # Apply initial randomization (especially scene setup)
    log.info("\nInitial randomization (scene setup)...")
    apply_randomization(randomizers, args.level)

    log.info("Custom materials applied successfully!")

    # Reset environment
    tic = time.time()
    obs, extras = env.reset(states=[init_state] * args.num_envs)

    toc = time.time()
    log.trace(f"Time to reset: {toc - tic:.2f}s")

    # Main replay loop
    step = 0
    num_envs = env.scenario.num_envs

    while True:
        log.debug(f"Step {step}")

        tic = time.time()

        # Action-based replay: use actions with domain randomization

        # Apply randomization periodically
        if step % args.randomize_interval == 0 and step > 0:
            log.info(f"\nStep {step}: Applying randomizations")
            apply_randomization(randomizers, args.level)

        # Get actions from trajectory
        actions = get_actions(all_actions, step, num_envs)

        # Execute step
        obs, reward, success, time_out, extras = env.step(actions)

        # Check termination conditions
        if success.any():
            log.info(f"Env {success.nonzero().squeeze(-1).tolist()} succeeded!")

        if time_out.any():
            log.info(f"Env {time_out.nonzero().squeeze(-1).tolist()} timed out!")

        if success.all() or time_out.all():
            log.info("All environments terminated!")
            break

        toc = time.time()
        log.trace(f"Time to step: {toc - tic:.2f}s")

        # Save observation
        tic = time.time()
        obs_saver.add(obs)
        toc = time.time()
        log.trace(f"Time to save obs: {toc - tic:.2f}s")

        # Check if trajectory ended
        check_array = all_states if args.object_states else all_actions
        if get_runout(check_array, step + 1):
            log.info("Run out of trajectory data, stopping")
            break

        step += 1

    obs_saver.save()
    log.info(f"\n[SAVED] Video saved to: {video_path}")


def main():
    @configclass
    class Args:
        sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaacsim"
        renderer: Literal["isaacsim", "isaacgym", "genesis", "pybullet", "mujoco", "sapien2", "sapien3"] | None = None
        robot: str = "franka"
        scene: str | None = None
        num_envs: int = 1
        headless: bool = False
        seed: int | None = 42

        level: Literal[0, 1, 2, 3] = 1
        """Randomization level (ALL levels have same tabletop scene):
        0 - Baseline (no DR) - Scene with FIXED textured materials
        1 - Object only - Scene with FIXED textured materials
        2 - + Visual (scene materials + lights) - Scene materials RANDOMIZE
        3 - + Camera - Scene materials RANDOMIZE
        """

        randomize_interval: int = 10

        object_states: bool = False
        """If True, replay using object states (deterministic, no DR applied).
        If False (default), replay using actions (allows domain randomization)."""

    args = tyro.cli(Args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    log.info("=" * 70)
    log.info("Domain Randomization Demo with Trajectory Replay - Final Version")
    log.info("=" * 70)
    log.info("\nConfiguration:")
    log.info(f"  Simulator: {args.sim}")
    log.info(f"  Robot: {args.robot}")
    log.info(f"  Seed: {args.seed}")
    log.info(f"  Level: {args.level}")

    log.info("\n  SCENE SETUP (ALL LEVELS):")
    log.info("      - Objects placed ON TABLE (z=0.7m)")
    log.info("      - Room: 10m x 10m with walls and ceiling")
    log.info("      - Table: 1.8m x 1.8m WITH PHYSICS COLLISION")
    log.info("      - Lights: 4 sources, VERY BRIGHT for enclosed room")

    if args.level < 2:
        log.info("\n  Note: Scene materials are FIXED with textures (not randomized)")
    else:
        log.info("\n  Note: Scene materials WILL be randomized")

    # Create environment
    tic = time.time()
    env = create_env(args)
    toc = time.time()
    log.trace(f"Time to launch: {toc - tic:.2f}s")

    # Get handler for randomization
    handler = env.handler

    # Load trajectory data
    traj_filepath = env.traj_filepath
    log.info(f"\nLoading trajectory from: {traj_filepath}")

    tic = time.time()
    assert os.path.exists(traj_filepath), f"Trajectory file: {traj_filepath} does not exist."
    init_states, all_actions, all_states = get_traj(traj_filepath, env.scenario.robots[0], handler)
    # z axis lifting by table height
    init_state = init_states[0]
    for obj_name, obj_state in init_state["objects"].items():
        obj_state["pos"][2] += 0.7

    for robot_name, robot_state in init_state["robots"].items():
        robot_state["pos"][2] += 0.7

    env.handler.set_states(init_states, env_ids=list(range(args.num_envs)))

    toc = time.time()
    log.trace(f"Time to load data: {toc - tic:.2f}s")
    log.info(f"Loaded {len(all_actions[0]) if all_actions else 0} actions from trajectory")

    # Initialize randomizers
    randomizers = initialize_randomizers(handler, args)

    # Show replay mode information
    if args.object_states:
        log.info("\n" + "!" * 70)
        log.info("WARNING: Using state-based replay mode")
        log.info("Domain randomization will NOT be applied (deterministic replay)")
        log.info("!" * 70)

    # Run replay with randomization
    run_replay_with_randomization(env, randomizers, init_state, all_actions, all_states, args)

    # Cleanup
    env.close()
    if args.sim == "isaacsim":
        env.handler.simulation_app.close()

    log.info("\n" + "=" * 70)
    log.info("Demo Completed!")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

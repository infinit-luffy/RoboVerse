"""Domain Randomization Demo - Final Version

Comprehensive demonstration of all 5 randomizers with 5 randomization levels.

## Randomization Levels:
- Level 0: No randomization (baseline) - Ground
- Level 1: Object properties (physics + pose) - Ground
- Level 2: + Visual (materials + lights) - Ground
- Level 3: + Camera randomization - Ground
- Level 4: + Scene with TABLE - Objects ON TABLE

## Critical Notes for Level 4:
- Table height: 0.7m
- Objects placed ON table surface (z = 0.7 + object_height/2)
- Room: 10m x 10m x 5m (walls enclose scene)
- Table: 1.8m x 1.8m (smaller than room)
- Lights: MUCH BRIGHTER (walls absorb light)
- SceneRandomizer creates table WITH PHYSICS COLLISION
"""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import os
from typing import Literal

import numpy as np
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

from metasim.constants import PhysicStateType
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
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg, SphereLightCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


def create_scenario(args) -> ScenarioCfg:
    """Create simulation scenario - configuration changes based on level."""
    scenario = ScenarioCfg(
        robots=["franka"],
        num_envs=args.num_envs,
        simulator=args.sim,
        headless=args.headless,
    )

    has_table = args.level >= 0
    table_height = 0.7 if has_table else 0.0

    # Camera - adjust for table
    if has_table:
        scenario.cameras = [
            PinholeCameraCfg(
                name="main_camera",
                width=1024,
                height=1024,
                pos=(2.5, -2.5, 2.5),  # Higher and further back for table view
                look_at=(0.0, 0.0, table_height + 0.15),  # Look at objects on table
            )
        ]
    else:
        scenario.cameras = [
            PinholeCameraCfg(
                name="main_camera",
                width=1024,
                height=1024,
                pos=(1.5, -1.5, 1.5),
                look_at=(0.0, 0.0, 0.0),
            )
        ]

    # Lights - CRITICAL: Enclosed room needs MUCH more light
    if has_table:
        # Level 4: Enclosed room (10m x 10m x 5m)
        # Walls at ±5m, so lights must be within ±4m to stay inside
        # Use multiple bright lights for enclosed space
        scenario.lights = [
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
    else:
        # Level 0-3: Open space - normal lighting
        scenario.lights = [
            DistantLightCfg(
                name="main_light",
                intensity=1000.0,
                color=(1.0, 1.0, 1.0),
                polar=45.0,
                azimuth=30.0,
                is_global=True,
            ),
            SphereLightCfg(
                name="fill_light",
                intensity=500.0,
                color=(0.9, 0.9, 1.0),
                radius=0.3,
                pos=(0.0, 0.0, 2.5),
                is_global=False,
            ),
        ]

    # Objects
    scenario.objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.1, 0.1, 0.1),
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        PrimitiveSphereCfg(
            name="sphere",
            radius=0.1,
            color=[0.0, 0.0, 1.0],
            physics=PhysicStateType.RIGIDBODY,
        ),
        ArticulationObjCfg(
            name="box_base",
            fix_base_link=True,
            usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
            urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
            mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
        ),
    ]

    return scenario


def get_init_states(level, num_envs):
    """Get initial states for objects and robot based on level."""

    # Object dimensions
    cube_size = 0.1
    sphere_radius = 0.1
    box_base_height = 0.15  # Approximate height of box_base

    if level >= 0:
        # Level 4: Objects ON TABLE
        table_surface_z = 0.7  # Table surface height

        # Objects placed on table surface
        # For cube: center_z = surface_z + half_size
        # For sphere: center_z = surface_z + radius (sphere sits on surface)
        # For box_base: center_z = surface_z + half_height

        objects = {
            "cube": {
                "pos": torch.tensor([0.3, 0.05, table_surface_z + cube_size / 2]),  # 0.75m
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "sphere": {
                "pos": torch.tensor([0.3, 0.15, table_surface_z + sphere_radius]),  # 0.8m
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "box_base": {
                "pos": torch.tensor([-0.2, 0.0, table_surface_z + box_base_height / 2]),  # 0.775m
                "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                "dof_pos": {"box_joint": 0.0},
            },
        }

        # Robot also on table (if base at table height)
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
    else:
        # Level 0-3: Objects ON GROUND
        objects = {
            "cube": {
                "pos": torch.tensor([0.3, -0.2, cube_size / 2]),  # 0.05m
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "sphere": {
                "pos": torch.tensor([0.4, -0.6, sphere_radius]),  # 0.1m
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "box_base": {
                "pos": torch.tensor([0.5, 0.2, box_base_height / 2]),  # 0.075m
                "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),
                "dof_pos": {"box_joint": 0.0},
            },
        }

        # Robot on ground
        robot = {
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

    # if level == 0:
    #     log.info("Level 0: Baseline - No randomization")
    #     return randomizers

    # Level 0+: Object randomization
    if level >= 0:
        log.info("\n[Level 1] Object Randomization (Physics + Pose)")
        log.info("-" * 70)

        cube_rand = ObjectRandomizer(
            ObjectPresets.grasping_target("cube"),
            seed=args.seed,
        )
        cube_rand.bind_handler(handler)
        randomizers["object"].append(cube_rand)
        log.info("  [OK] cube: grasping_target preset")

        sphere_rand = ObjectRandomizer(
            ObjectPresets.bouncy_object("sphere"),
            seed=args.seed,
        )
        sphere_rand.bind_handler(handler)
        randomizers["object"].append(sphere_rand)
        log.info("  [OK] sphere: bouncy_object preset")

        # Level 2+: Visual randomization
        if level >= 1:
            log.info("\n[Level 2] Visual Randomization (Materials + Lights)")
            log.info("-" * 70)

            log.info("  Materials:")
            cube_mat = MaterialRandomizer(
                MaterialPresets.wood_object("cube", use_mdl=True),
                seed=args.seed,
            )
            cube_mat.bind_handler(handler)
            randomizers["material"].append(cube_mat)
            log.info("    [OK] cube: wood (MDL)")

            sphere_mat = MaterialRandomizer(
                MaterialPresets.rubber_object("sphere"),
                seed=args.seed,
            )
            sphere_mat.bind_handler(handler)
            randomizers["material"].append(sphere_mat)
            log.info("    [OK] sphere: rubber (PBR)")

            box_mat = MaterialRandomizer(
                MaterialPresets.wood_object("box_base", use_mdl=True),
                seed=args.seed,
            )
            box_mat.bind_handler(handler)
            randomizers["material"].append(box_mat)
            log.info("    [OK] box_base: wood (MDL)")

            # Light randomization
            log.info("  Lights:")
            main_light_rand = LightRandomizer(
                LightPresets.outdoor_daylight("main_light", randomization_mode="combined"),
                seed=args.seed,
            )
            main_light_rand.bind_handler(handler)
            randomizers["light"].append(main_light_rand)

            # Use different strategies for Level 4 vs open space
            if level >= 2:
                # For enclosed room: ONLY randomize intensity and color
                # DO NOT randomize position - walls will block lights if they move
                fill_light_rand = LightRandomizer(
                    LightPresets.outdoor_daylight("fill_light", randomization_mode="intensity_only"),
                    seed=args.seed,
                )
                fill_light_rand.bind_handler(handler)
                randomizers["light"].append(fill_light_rand)

                back_light_rand = LightRandomizer(
                    LightPresets.outdoor_daylight("back_light", randomization_mode="intensity_only"),
                    seed=args.seed,
                )
                back_light_rand.bind_handler(handler)
                randomizers["light"].append(back_light_rand)

                table_light_rand = LightRandomizer(
                    LightPresets.outdoor_daylight("table_light", randomization_mode="intensity_only"),
                    seed=args.seed,
                )
                table_light_rand.bind_handler(handler)
                randomizers["light"].append(table_light_rand)
            else:
                # For open space, use normal presets
                fill_light_rand = LightRandomizer(
                    LightPresets.indoor_ambient("fill_light", randomization_mode="combined"),
                    seed=args.seed,
                )
                fill_light_rand.bind_handler(handler)
                randomizers["light"].append(fill_light_rand)

            log.info(f"    [OK] {len(randomizers['light'])} lights configured")

        # Level 3+: Camera randomization
        if level >= 3:
            log.info("\n[Level 3] Camera Randomization (Viewpoint Variation)")
            log.info("-" * 70)

            camera_rand = CameraRandomizer(
                CameraPresets.surveillance_camera("main_camera", randomization_mode="combined"),
                seed=args.seed,
            )
            camera_rand.bind_handler(handler)
            randomizers["camera"].append(camera_rand)
            log.info("  [OK] main_camera: surveillance preset")

        # Level 3: Scene randomization with TABLE
        if level >= 2:
            log.info("\n[Level 3] Scene Randomization (Tabletop Workspace)")
            log.info("-" * 70)

            scene_cfg = ScenePresets.tabletop_workspace(
                room_size=10.0,
                wall_height=5.0,
                table_size=(1.8, 1.8, 0.1),  # Smaller than room
                table_height=0.7,
            )
            scene_rand = SceneRandomizer(scene_cfg, seed=args.seed)
            scene_rand.bind_handler(handler)
            randomizers["scene"] = scene_rand
            log.info("  [OK] Tabletop workspace with PHYSICS COLLISION")
            log.info("    - Room: 10m x 10m x 5m")
            log.info("    - Table: 1.8m x 1.8m at z=0.7m (WITH COLLIDER)")
            log.info("    - Materials: ~300 (table), ~150 each (floor/walls/ceiling)")

    log.info("\n" + "=" * 70)
    return randomizers


def apply_randomization(randomizers, level):
    """Apply all active randomizers."""

    if level >= 0:
        for rand in randomizers["object"]:
            rand()

    if level >= 1:
        for rand in randomizers["material"]:
            rand()
        for rand in randomizers["light"]:
            rand()

    if level >= 2:
        if randomizers["scene"]:
            randomizers["scene"]()

    if level >= 3:
        for rand in randomizers["camera"]:
            rand()



def run_simulation(handler, randomizers, args):
    """Run simulation with periodic randomization."""
    os.makedirs("get_started/output", exist_ok=True)
    video_path = f"get_started/output/12_dr_level{args.level}_{args.sim}.mp4"
    obs_saver = ObsSaver(video_path=video_path)

    log.info("\n" + "=" * 70)
    log.info("Running Simulation")
    log.info("=" * 70)
    log.info(f"Video: {video_path}")
    log.info(f"Randomization interval: every {args.randomize_interval} steps")
    log.info(f"Total steps: {args.num_steps}")

    for step in range(args.num_steps):
        handler.simulate()
        obs = handler.get_states(mode="tensor")

        obs_saver.add(obs)

        if step % args.randomize_interval == 0 and step > 0:
            log.info(f"\nStep {step}: Applying randomizations")
            apply_randomization(randomizers, args.level)

    obs_saver.save()
    log.info(f"\n[SAVED] Video saved to: {video_path}")


def main():
    @configclass
    class Args:
        sim: Literal["isaacsim"] = "isaacsim"
        num_envs: int = 1
        headless: bool = False
        seed: int | None = 42

        level: Literal[0, 1, 2, 3] = 1
        """Randomization level:
        0 - Baseline (no DR) - Ground
        1 - Object only - Ground
        2 - + Visual - Ground
        3 - + Camera - Ground
        4 - + Scene (TABLE) - Objects ON TABLE
        """

        num_steps: int = 100
        randomize_interval: int = 10

    args = tyro.cli(Args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    log.info("=" * 70)
    log.info("Domain Randomization Demo - Final Version")
    log.info("=" * 70)
    log.info("\nConfiguration:")
    log.info(f"  Simulator: {args.sim}")
    log.info(f"  Seed: {args.seed}")
    log.info(f"  Level: {args.level}")

    if args.level >= 3:
        log.info("\n  LEVEL 3 SPECIAL SETUP:")
        log.info("      - Objects placed ON TABLE (z=0.7m)")
        log.info("      - Room: 10m x 10m with walls and ceiling")
        log.info("      - Table: 1.8m x 1.8m WITH PHYSICS COLLISION")
        log.info("      - Lights: 4 sources, VERY BRIGHT for enclosed room")
    else:
        log.info("\n  Note: Objects on GROUND (z~0.05-0.1m)")

    # Create scenario
    scenario = create_scenario(args)
    handler = get_handler(scenario)

    if args.level >= 0:
        log.info("\n[Level 0 Setup] Creating table first...")
        # Create scene randomizer early to build table
        scene_cfg = ScenePresets.tabletop_workspace(
            room_size=10.0,
            wall_height=5.0,
            table_size=(1.8, 1.8, 0.1),
            table_height=0.7,
        )
        # if args.level >= 3:
        scene_rand_early = SceneRandomizer(scene_cfg, seed=args.seed)
        scene_rand_early.bind_handler(handler)
        scene_rand_early()  # Create table NOW
        scene_rand_early()
        log.info("[OK] Table created (with physics collision)")

        # Let table settle
        for _ in range(0):
            handler.simulate()
        log.info("[OK] Table stable")

    # Now place objects (table exists if Level 3)
    init_states = get_init_states(args.level, scenario.num_envs)
    handler.set_states(init_states)
    log.info(f"\n[OK] Objects initialized at {'TABLE' if args.level >= 3 else 'GROUND'} level")

    # Stabilize physics
    log.info("\nStabilizing physics...")
    stabilize_steps = 20 if args.level >= 3 else 10
    for _ in range(stabilize_steps):
        handler.simulate()
    log.info("[OK] Physics stable")

    # Log object positions after stabilization
    if args.level >= 3:
        log.info("\n  Verifying object positions on table:")
        try:
            obs = handler.get_states(mode="tensor")
            if hasattr(obs, "objects") and obs.objects is not None:
                for obj_name in ["cube", "sphere", "box_base"]:
                    if obj_name in obs.objects:
                        pos = obs.objects[obj_name]["pos"][0]
                        expected = "0.75-0.8" if obj_name in ["cube", "sphere", "box_base"] else "0.7"
                        status = "[OK]" if pos[2] >= 0.7 else "[WARN]"
                        log.info(f"    {status} {obj_name}: z={pos[2]:.3f}m (expected ~{expected}m)")
        except Exception as e:
            log.warning(f"  Could not verify positions: {e}")

    # Initialize randomizers (scene already created for Level 3)
    randomizers = initialize_randomizers(handler, args)

    # For Level 3, use the already-created scene randomizer
    if args.level >= 3:
        randomizers["scene"] = scene_rand_early

    # Run simulation
    run_simulation(handler, randomizers, args)

    handler.close()

    log.info("\n" + "=" * 70)
    log.info("Demo Completed!")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

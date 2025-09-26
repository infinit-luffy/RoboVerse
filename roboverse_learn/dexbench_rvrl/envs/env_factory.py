"""Unified environment factory for creating different types of environments."""

from __future__ import annotations

from typing import Literal

from roboverse_learn.dexbench_rvrl.envs.base import BaseVecEnv

SEED_SPACING = 1_000_000


def create_vector_env(
    env_id: str,
    args=None,
    env_cfg=None,
) -> BaseVecEnv:
    """
    Create a vectorized environment.

    Args:
        env_id: Environment identifier
        obs_mode: Observation type
        num_envs: Number of parallel environments
        seed: Seed for the environment
        action_repeat: Action repeat for the environment
        image_size: Image size for RGB observation. Only used when :param:`obs_mode` is "rgb". Type is (width, height) tuple.
        device: Device to run on (for IsaacLab)

    Returns:
        Vectorized environment
    """
    if env_id.startswith("dexbench/"):
        from .dex_env import DexEnv

        envs = DexEnv(env_id.replace("dexbench/", ""), args, env_cfg)
        return envs
    else:
        raise ValueError(f"Unknown environment: {env_id}")

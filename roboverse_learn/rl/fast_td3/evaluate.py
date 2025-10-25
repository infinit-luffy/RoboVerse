from __future__ import annotations

import os
import sys
import argparse
from typing import Any
import yaml

import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ensure repository root is on sys.path for local package imports
import rootutils

rootutils.setup_root(__file__, pythonpath=True)

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch
import numpy as np
from loguru import logger as log
from torch.amp import autocast

from roboverse_learn.rl.fast_td3.fttd3_module import Actor, EmpiricalNormalization
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.task.registry import get_task_class


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint from file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    log.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def evaluate(
    env,
    actor,
    obs_normalizer,
    num_episodes: int,
    device: torch.device,
    amp_enabled: bool = False,
    amp_device_type: str = "cpu",
    amp_dtype: torch.dtype = torch.float16,
    render: bool = False,
    video_path: str = None,
) -> dict:
    """
    Evaluate the policy for a specified number of episodes.

    Args:
        env: The environment to evaluate on
        actor: The policy network
        obs_normalizer: Observation normalizer
        num_episodes: Number of episodes to run
        device: Device to run evaluation on
        amp_enabled: Whether to use automatic mixed precision
        amp_device_type: Device type for AMP
        amp_dtype: Data type for AMP
        render: Whether to render and save video
        video_path: Path to save rendered video

    Returns:
        Dictionary with evaluation statistics
    """
    actor.eval()
    obs_normalizer.eval()

    num_eval_envs = env.num_envs
    episode_returns = []
    episode_lengths = []
    episode_successes = []

    frames = [] if render else None

    episodes_completed = 0
    current_returns = torch.zeros(num_eval_envs, device=device)
    current_lengths = torch.zeros(num_eval_envs, device=device)
    done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

    obs, info = env.reset()
    if render:
        frames.append(env.render())

    max_steps = env.max_episode_steps * num_episodes

    for step in range(max_steps):
        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = obs_normalizer(obs)
            actions = actor(norm_obs)

        next_obs, rewards, terminated, time_out, infos = env.step(actions.float())
        dones = terminated | time_out

        if render:
            frames.append(env.render())

        # Update episode statistics
        current_returns = torch.where(~done_masks, current_returns + rewards, current_returns)
        current_lengths = torch.where(~done_masks, current_lengths + 1, current_lengths)

        # Check for newly completed episodes
        newly_done = dones & ~done_masks
        if newly_done.any():
            for i in range(num_eval_envs):
                if newly_done[i]:
                    episode_returns.append(current_returns[i].item())
                    episode_lengths.append(current_lengths[i].item())

                    # Check for success if available in info
                    if "success" in infos:
                        episode_successes.append(infos["success"][i].item())

                    episodes_completed += 1

                    # Reset stats for this env
                    current_returns[i] = 0
                    current_lengths[i] = 0

            done_masks = torch.logical_or(done_masks, dones)

        # Stop if we've completed enough episodes
        if episodes_completed >= num_episodes:
            break

        # Reset done_masks if all envs are done
        if done_masks.all():
            done_masks.fill_(False)
            obs, info = env.reset()
        else:
            obs = next_obs

    # Save video if rendering
    if render and frames and video_path:
        import imageio.v2 as iio
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        iio.mimsave(video_path, frames, fps=30)
        log.info(f"Saved evaluation video to {video_path}")

    # Compute statistics
    stats = {
        "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
        "std_return": np.std(episode_returns) if episode_returns else 0.0,
        "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "std_length": np.std(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": len(episode_returns),
    }

    if episode_successes:
        stats["success_rate"] = np.mean(episode_successes)

    return stats


def main():
    parser = argparse.ArgumentParser(description='FastTD3 Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                       help='Render and save video')
    parser.add_argument('--video_path', type=str, default='output/eval_rollout.mp4',
                       help='Path to save video')
    parser.add_argument('--device_rank', type=int, default=0,
                       help='GPU device rank')
    parser.add_argument('--num_envs', type=int, default=None,
                       help='Number of parallel environments (default: from checkpoint config)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode')
    args = parser.parse_args()

    # Load checkpoint
    device = torch.device("cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)

    # Get configuration from checkpoint
    config = checkpoint.get("config", {})

    # Override device based on availability
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_rank}")
        torch.cuda.set_device(args.device_rank)
    elif torch.backends.mps.is_available():
        device = torch.device(f"mps:{args.device_rank}")

    log.info(f"Using device: {device}")
    log.info(f"Checkpoint global step: {checkpoint.get('global_step', 'unknown')}")

    # Get task configuration
    task_name = config.get("task")
    if not task_name:
        raise ValueError("Task name not found in checkpoint config")

    # Setup environment
    task_cls = get_task_class(task_name)
    num_envs = args.num_envs if args.num_envs is not None else config.get("num_envs", 1)

    # Configure cameras for rendering if needed
    cameras = []
    if args.render:
        cameras = [
            PinholeCameraCfg(
                width=config.get("video_width", 1024),
                height=config.get("video_height", 1024),
                pos=(4.0, -4.0, 4.0),
                look_at=(0.0, 0.0, 0.0),
            )
        ]

    scenario = task_cls.scenario.update(
        robots=config.get("robots", ["franka"]),
        simulator=config.get("sim", "mujoco"),
        num_envs=num_envs,
        headless=args.headless or not args.render,
        cameras=cameras,
    )

    env = task_cls(scenario, device=device)

    # Get dimensions
    n_obs = env.num_obs
    n_act = env.num_actions

    # Create actor and normalizer
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=num_envs,
        device=device,
        init_scale=config.get("init_scale", 0.1),
        hidden_dim=config.get("actor_hidden_dim", 256),
    )

    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)

    # Load weights
    actor.load_state_dict(checkpoint["actor_state_dict"])
    if checkpoint.get("obs_normalizer_state"):
        obs_normalizer.load_state_dict(checkpoint["obs_normalizer_state"])

    # Setup AMP
    amp_enabled = config.get("amp", False) and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bf16" else torch.float16

    # Run evaluation
    log.info(f"Evaluating for {args.num_episodes} episodes...")
    stats = evaluate(
        env=env,
        actor=actor,
        obs_normalizer=obs_normalizer,
        num_episodes=args.num_episodes,
        device=device,
        amp_enabled=amp_enabled,
        amp_device_type=amp_device_type,
        amp_dtype=amp_dtype,
        render=args.render,
        video_path=args.video_path if args.render else None,
    )

    # Print results
    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info(f"  Episodes: {stats['num_episodes']}")
    log.info(f"  Mean Return: {stats['mean_return']:.4f} ± {stats['std_return']:.4f}")
    log.info(f"  Mean Length: {stats['mean_length']:.4f} ± {stats['std_length']:.4f}")
    if "success_rate" in stats:
        log.info(f"  Success Rate: {stats['success_rate']:.2%}")
    log.info("=" * 50)

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()

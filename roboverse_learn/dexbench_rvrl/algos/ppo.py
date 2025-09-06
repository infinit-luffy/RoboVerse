from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import copy
import os
import time
from collections import deque
from datetime import datetime

import rootutils
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from dexbench_rvrl.algos.module import ActorCritic, ActorCritic_RGB
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax


########################################################
## Standalone utils
########################################################
class RollingMeter:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)

    def update(self, rewards: torch.Tensor):
        self.deque.extend(rewards.cpu().numpy().tolist())

    @property
    def len(self) -> int:
        return len(self.deque)

    @property
    def mean(self) -> float:
        return np.mean(self.deque).item()

    @property
    def std(self) -> float:
        return np.std(self.deque).item()


########################################################
## Networks
########################################################
class PPO:
    def __init__(
        self,
        env,
        train_cfg,
        device="cpu",
        log_dir="run",
        model_dir=None,
        is_testing=False,
        print_log=True,
        wandb_run=None,
    ):
        self.device = device

        ## env info
        self.action_dim = np.prod(env.single_action_space.shape)
        self.obs_dim = np.prod(env.single_observation_space.shape)
        self.obs_type = getattr(env, "obs_type", "state")
        self.use_prio = getattr(env, "use_prio", True)
        if self.use_prio:
            self.state_shape = getattr(env, "proprio_shape", None)
        else:
            self.state_shape = getattr(env, "proceptual_shape", None)
        self.img_h = getattr(env, "img_h", None)
        self.img_w = getattr(env, "img_w", None)

        ## learn cfg
        self.train_cfg = copy.deepcopy(train_cfg)
        learn_cfg = self.train_cfg["learn"]
        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.step_size = learn_cfg["optim_stepsize"]
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.nsteps = learn_cfg["nsteps"]
        self.learning_rate = learn_cfg["optim_stepsize"]
        self.num_envs = learn_cfg["num_envs"]
        self.num_learning_epochs = learn_cfg.get("num_learning_epochs", 8)
        self.max_iterations = learn_cfg.get("max_iterations", 100000)
        self.log_interval = learn_cfg.get("log_interval", 1)

        ## model cfg
        self.model_cfg = self.train_cfg["policy"]

        ## ppo components
        self.env = env
        if self.obs_type == "state":
            print("Using state observation")
            self.actor_critic = ActorCritic(
                self.obs_dim,
                self.action_dim,
                self.init_noise_std,
                self.model_cfg,
            )
        elif self.obs_type == "rgb":
            print("Using RGB observation")
            self.actor_critic = ActorCritic_RGB(
                self.obs_dim,
                self.action_dim,
                self.init_noise_std,
                self.model_cfg,
                state_shape=self.state_shape,
                img_h=self.img_h,
                img_w=self.img_w,
            )
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type}")

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate, eps=1e-5)

        self.obss = torch.zeros((self.nsteps + 1, self.num_envs) + self.obs_dim, device=device)
        self.actions = torch.zeros((self.nsteps + 1, self.num_envs) + self.action_dim, device=device)
        self.log_probs = torch.zeros((self.nsteps + 1, self.num_envs), device=device)
        self.mu = torch.zeros((self.nsteps + 1, self.num_envs) + self.action_dim, device=device)
        self.sigma = torch.zeros((self.nsteps + 1, self.num_envs) + self.action_dim, device=device)
        self.values = torch.zeros((self.nsteps + 1, self.num_envs), device=device)
        self.rewards = torch.zeros((self.nsteps + 1, self.num_envs), device=device)
        self.advantages = torch.zeros((self.nsteps + 1, self.num_envs), device=device)
        self.returns = torch.zeros((self.nsteps + 1, self.num_envs), device=device)
        self.dones = torch.zeros((self.nsteps + 1, self.num_envs), device=device)

        self.episode_rewards = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_lengths = RollingMeter(learn_cfg.get("window_size", 100))
        self.cur_rewards_sum = torch.zeros(self.num_envs, device=device)
        self.cur_episode_length = torch.zeros(self.num_envs, device=device)

        ## PPO params
        self.batch_size = self.num_envs * self.nsteps
        self.mini_batch_size = self.batch_size // learn_cfg.get("num_minibatches", 32)
        self.num_iterations = self.total_timesteps // self.batch_size
        self.clip_param = learn_cfg.get("clip_param", 0.2)
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg.get("entropy_coef", 0.0)
        self.gamma = learn_cfg.get("gamma", 0.99)
        self.lam = learn_cfg.get("lam", 0.95)
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)

        ## logging
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.print_log = print_log
        self.wandb_run = wandb_run
        self.is_testing = is_testing
        self.current_learning_epoch = 0
        self.global_step = 0
        self.tot_time = 0

        self.actor_critic.to(self.device)

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self):
        if self.is_testing:
            assert self.model_dir is not None
            self.test(self.model_dir)
            print(f"Loaded model from {self.model_dir}")
        elif self.model_dir is not None:
            self.load(self.model_dir)
            print(f"Loaded model from {self.model_dir}")
        obs = self.env.reset()
        self.obss[0] = obs
        self.dones[0] = torch.zeros(self.num_envs).to(self.device)
        if not self.is_testing:
            for iteration in tqdm(range(self.current_learning_iteration, self.max_iterations)):
                ## collect rollout
                start_time = time.time()
                ep_infos = []
                for t in range(self.nsteps):
                    self.global_step += self.num_envs

                    with torch.no_grad():
                        action, log_prob, value, mu, sigma = self.actor_critic.act(self.obss[t])

                    next_obs, reward, terminated, truncated, infos = self.env.step(action)

                    next_done = torch.logical_or(terminated, truncated)
                    self.actions[t].copy_(action)
                    self.log_probs[t].copy_(log_prob)
                    self.mu[t].copy_(mu)
                    self.sigma[t].copy_(sigma)
                    self.values[t].copy_(value.view(-1))
                    self.rewards[t].copy_(reward.view(-1))
                    self.obss[t + 1].copy_(next_obs)
                    self.dones[t + 1].copy_(next_done)

                    obs.copy_(next_obs)

                    self.cur_rewards_sum += reward
                    self.cur_episode_length += 1
                    self.episode_rewards.update(self.cur_rewards_sum[next_done])
                    self.episode_lengths.update(self.cur_episode_length[next_done])
                    self.cur_rewards_sum[next_done] = 0
                    self.cur_episode_length[next_done] = 0

                _, _, last_values, _, _ = self.actor_critic.act(self.obss[self.nsteps])
                self.values[self.nsteps].copy_(last_values)
                collection_time = time.time() - start_time
                self.tot_time += collection_time

                mean_episode_length = self.episode_lengths.mean if self.episode_lengths.len > 0 else 0
                mean_episode_reward = self.episode_rewards.mean if self.episode_rewards.len > 0 else 0

                compute_start_time = time.time()
                self.compute_returns()
                mean_value_loss, mean_surrogate_loss = self.update()
                compute_time = time.time() - compute_start_time
                self.tot_time += compute_time
                if self.print_log:
                    self.log(locals())
                if (iteration + 1) % self.log_interval == 0 or iteration == self.max_iterations - 1:
                    self.save(os.path.join(self.log_dir, f"model_{iteration + 1}.pt"))
                ep_infos.clear()

    def compute_returns(self):
        with torch.no_grad():
            self.advantages[self.nsteps] = 0
            for t in reversed(range(self.nsteps)):
                delta = self.rewards[t] + self.gamma * self.values[t + 1] * (1 - self.dones[t + 1]) - self.values[t]
                self.advantages[t] = delta + self.gamma * self.lam * self.advantages[t + 1] * (1 - self.dones[t + 1])
            self.returns = self.advantages + self.values
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def update(self):
        ## log
        mean_value_loss = 0
        mean_surrogate_loss = 0

        ## flatten and shuffle
        random_indices = torch.randperm(self.batch_size)
        b_obs = self.obss[:-1].view(-1, self.obs_dim)[random_indices]
        b_log_probs = self.log_probs[:-1].view(-1)[random_indices]
        b_mu = self.mu[:-1].view(-1, self.action_dim)[random_indices]
        b_sigma = self.sigma[:-1].view(-1, self.action_dim)[random_indices]
        b_actions = self.actions[:-1].view(-1, self.action_dim)[random_indices]
        b_returns = self.returns[:-1].view(-1)[random_indices]
        b_advantages = self.advantages[:-1].view(-1)[random_indices]
        b_values = self.values[:-1].view(-1)[random_indices]

        ## optimize
        for epoch in range(self.num_learning_epochs):
            for start in range(0, self.batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size

                new_log_probs, entropy, new_value, new_mu, new_sigma = self.actor_critic.evaluate(
                    b_obs[start:end], b_actions[start:end]
                )

                # KL
                if self.desired_kl is not None and self.schedule == "adaptive":
                    kl = torch.sum(
                        new_sigma
                        - b_sigma
                        + (torch.square(b_sigma.exp()) + torch.square(b_mu - new_mu)) / (2.0 * new_sigma.exp().square())
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                # surrogate loss
                logratio = new_log_probs - b_log_probs[start:end]
                ratio = logratio.exp()

                mb_advantages = b_advantages[start:end]

                surr_loss1 = mb_advantages * ratio
                surr_loss2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                surr_loss = -torch.min(surr_loss1, surr_loss2).mean()

                # value loss
                value_loss = 0.5 * (new_value - b_returns[start:end]).pow(2).mean()

                # total loss
                loss = surr_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surr_loss.item()

        num_updates = self.batch_size * self.num_learning_epochs / self.mini_batch_size
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    def log(self, locs, width=80, pad=35):
        iteration_time = locs["collection_time"] + locs["compute_time"]
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""
                if key == "total_succ_rate":
                    locs["mean_succ_rate"] = value.item()
        mean_std = self.actor_critic.log_std.exp().mean()

        fps = int(self.nsteps * self.num_envs / iteration_time)
        str = f" \033[1m Learning iteration {locs['iteration']}/{locs['max_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
                # f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
                f"""{"Mean reward:":>{pad}} {locs["mean_episode_reward"]:.2f}\n"""
                f"""{"Mean episode length:":>{pad}} {locs["mean_episode_length"]:.2f}\n"""
                f"""{"Mean reward/step:":>{pad}} {locs["mean_reward"]:.2f}\n"""
                f"""{"Mean episode length/episode:":>{pad}} {locs["mean_trajectory_length"]:.2f}\n"""
            )
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{str.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
                # f"""{"Value function loss:":>{pad}} {locs["mean_value_loss"]:.4f}\n"""
                f"""{"Surrogate loss:":>{pad}} {locs["mean_surrogate_loss"]:.4f}\n"""
                f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
                f"""{"Mean reward/step:":>{pad}} {locs["mean_reward"]:.2f}\n"""
                f"""{"Mean episode length/episode:":>{pad}} {locs["mean_trajectory_length"]:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.global_step}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {self.tot_time / (locs["iteration"] + 1) * (locs["max_iterations"] - locs["iteration"]):.1f}s\n"""
        )
        print(log_string)
        if self.wandb_run is not None:
            log_data = {
                "Loss/surrogate": locs["mean_surrogate_loss"],
                "Policy/noise_std": mean_std.item(),
                "Train/mean_reward_per_step": locs["mean_reward"],
                "Train/mean_episode_length_per_episode": locs["mean_trajectory_length"],
                "Train/mean_succ_rate": locs["mean_succ_rate"],
                "Train/fps": fps,
                "timesteps_total": self.global_step,
                "iteration_time": iteration_time,
                "iteration": locs["iteration"],
                "Train/mean_reward": locs["mean_episode_reward"],
                "Train/mean_episode_length": locs["mean_episode_length"],
            }

            self.wandb_run.log(log_data)

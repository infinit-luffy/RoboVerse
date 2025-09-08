from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import copy
import os
import time
from collections import deque
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as log
from tensordict import TensorDict
from torch import Tensor

from roboverse_learn.dexbench_rvrl.algos.sac.module import Actor, QNet
from roboverse_learn.dexbench_rvrl.algos.sac.storage import ReplayBuffer
from roboverse_learn.dexbench_rvrl.utils.reproducibility import enable_deterministic_run
from roboverse_learn.dexbench_rvrl.utils.timer import timer


class RollingMeter:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)

    def update(self, rewards):
        if isinstance(rewards, torch.Tensor):
            self.deque.extend(rewards.cpu().numpy().tolist())
        elif isinstance(rewards, float):
            self.deque.append(rewards)

    @property
    def len(self) -> int:
        return len(self.deque)

    @property
    def mean(self) -> float:
        return np.mean(self.deque).item()

    @property
    def std(self) -> float:
        return np.std(self.deque).item()


class SAC:
    def __init__(
        self,
        env,
        train_cfg: dict,
        device="cpu",
        log_dir="run",
        model_dir=None,
        is_testing=False,
        print_log=True,
        wandb_run=None,
    ):
        self.device = device

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
        self.env = env

        # learn cfg
        self.train_cfg = copy.deepcopy(train_cfg)
        if train_cfg.get("deterministic", False):
            enable_deterministic_run()
        learn_cfg = self.train_cfg["learn"]
        self.actor_lr = learn_cfg.get("actor_lr", 3e-4)
        self.critic_lr = learn_cfg.get("critic_lr", 1e-3)
        self.gamma = learn_cfg.get("gamma", 0.99)
        self.tau = learn_cfg.get("tau", 0.005)
        self.alpha = learn_cfg.get("alpha", 0.2)
        self.policy_frequency = learn_cfg.get("policy_frequency", 2)
        self.target_network_frequency = learn_cfg.get("target_network_frequency", 2)
        self.num_envs = env.num_envs
        self.batch_size = learn_cfg["batch_size"]
        self.max_iterations = learn_cfg.get("max_iterations", 100000)
        self.log_interval = learn_cfg.get("log_interval", 1)
        self.print_interval = learn_cfg.get("print_interval", 10)
        self.prefill = learn_cfg.get("prefill", 5000)
        self.max_grad_norm = learn_cfg.get("max_grad_norm", None)

        self.model_cfg = self.train_cfg.get("policy", None)

        ## Replay buffer
        self.buffer = ReplayBuffer(
            self.obs_dim,
            self.action_dim,
            device,
            self.num_envs,
            learn_cfg.get("buffer_size", 5000),
        )

        ## SAC components
        self.actor = Actor(self.obs_dim, self.action_dim, self.model_cfg).to(device)
        self.qf1 = QNet(self.obs_dim, self.action_dim, self.model_cfg).to(device)
        self.qf2 = QNet(self.obs_dim, self.action_dim, self.model_cfg).to(device)
        self.qf1_target = QNet(self.obs_dim, self.action_dim, self.model_cfg).to(device)
        self.qf2_target = QNet(self.obs_dim, self.action_dim, self.model_cfg).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        for param in self.qf1_target.parameters():
            param.requires_grad = False
        for param in self.qf2_target.parameters():
            param.requires_grad = False
        self.qf1_params = TensorDict(dict(self.qf1.named_parameters(), batch_size=()))
        self.qf2_params = TensorDict(dict(self.qf2.named_parameters(), batch_size=()))
        self.qf1_target_params = TensorDict(dict(self.qf1_target.named_parameters(), batch_size=()))
        self.qf2_target_params = TensorDict(dict(self.qf2_target.named_parameters(), batch_size=()))

        ## Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(chain(self.qf1.parameters(), self.qf2.parameters()), lr=self.critic_lr)

        ## Logging
        self.episode_rewards = RollingMeter(learn_cfg.get("window_size", 100))
        self.episode_lengths = RollingMeter(learn_cfg.get("window_size", 100))
        self.actor_losses = RollingMeter(learn_cfg.get("window_size", 100))
        self.critic_losses = RollingMeter(learn_cfg.get("window_size", 100))
        self.q_values = RollingMeter(learn_cfg.get("window_size", 100))
        self.cur_rewards_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.cur_episode_length = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self.model_dir = model_dir
        self.log_dir = log_dir
        self.print_log = print_log
        self.wandb_run = wandb_run
        self.is_testing = is_testing
        self.global_step = 0
        self.prev_global_step = 0
        self.current_learning_iteration = 0

    def test(self, path):
        actor_path = os.path.join(path, "actor.pt")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            log.info(f"Loaded model from {actor_path}")
        qf1_path = os.path.join(path, "qf1.pt")
        if os.path.exists(qf1_path):
            self.qf1.load_state_dict(torch.load(qf1_path, map_location=self.device))
            log.info(f"Loaded model from {qf1_path}")
        qf2_path = os.path.join(path, "qf2.pt")
        if os.path.exists(qf2_path):
            self.qf2.load_state_dict(torch.load(qf2_path, map_location=self.device))
            log.info(f"Loaded model from {qf2_path}")
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.actor.eval()
        self.qf1.eval()
        self.qf2.eval()
        log.info("Testing mode")

    def load(self, path):
        self.current_learning_iteration = int(path.split("_")[-1])
        actor_path = os.path.join(path, "actor.pt")
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            log.info(f"Loaded model from {actor_path}")
        qf1_path = os.path.join(path, "qf1.pt")
        if os.path.exists(qf1_path):
            self.qf1.load_state_dict(torch.load(qf1_path, map_location=self.device))
            log.info(f"Loaded model from {qf1_path}")
        qf2_path = os.path.join(path, "qf2.pt")
        if os.path.exists(qf2_path):
            self.qf2.load_state_dict(torch.load(qf2_path, map_location=self.device))
            log.info(f"Loaded model from {qf2_path}")
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.actor.train()
        self.qf1.train()
        self.qf2.train()
        log.info("Training mode")

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.qf1.state_dict(), os.path.join(path, "qf1.pt"))
        torch.save(self.qf2.state_dict(), os.path.join(path, "qf2.pt"))
        log.info(f"Saved model to {path}")

    def run(self):
        if self.is_testing:
            assert self.model_dir is not None, "model_dir must be specified in testing mode"
            self.test(self.model_dir)
        elif self.model_dir is not None:
            self.load(self.model_dir)
        obs = self.env.reset().clone()
        self.start_time = time.time()
        if not self.is_testing:
            ep_infos = []
            for iteration in range(self.current_learning_iteration, self.max_iterations):
                with timer("time/iteration"):
                    with torch.inference_mode(), timer("time/roll_out"):
                        if self.global_step < self.prefill:
                            action = torch.rand((self.num_envs, self.action_dim), device=self.device) * 2 - 1
                        else:
                            action, _ = self.actor.get_action(obs)
                        next_obs, reward, terminated, truncated, info = self.env.step(action)
                        done = torch.logical_or(terminated, truncated)
                        self.buffer.add(obs, action, reward, next_obs, done, terminated)
                        obs.copy_(next_obs)

                        ep_infos.append(info)
                        self.cur_rewards_sum += reward
                        self.cur_episode_length += 1
                        self.global_step += self.num_envs

                        if done.any():
                            self.episode_rewards.update(self.cur_rewards_sum[done])
                            self.episode_lengths.update(self.cur_episode_length[done])
                            self.cur_rewards_sum[done] = 0
                            self.cur_episode_length[done] = 0

                    self.global_step += self.num_envs
                    mean_reward = self.buffer.mean_reward()
                    # update the model
                    if self.global_step >= self.prefill:
                        with timer("time/sample_data"):
                            data = self.buffer.sample(self.batch_size)
                        with timer("time/update_model"):
                            self.update_q(data)
                            if iteration % self.policy_frequency == 0:
                                self.update_actor(data)
                            if iteration % self.target_network_frequency == 0:
                                with torch.no_grad():
                                    self.qf1_target_params.lerp_(self.qf1_params.data, self.tau)
                                    self.qf2_target_params.lerp_(self.qf2_params.data, self.tau)

                ## log
                if (iteration + 1) % self.log_interval == 0 or iteration == self.max_iterations - 1:
                    log_dir = os.path.join(self.log_dir, f"model_{iteration + 1}")
                    os.makedirs(log_dir, exist_ok=True)
                    self.save(log_dir)
                if (iteration + 1) % self.print_interval == 0 & self.print_log:
                    self.log(locals())
                    ep_infos.clear()

    def update_q(self, data: dict[str, Tensor]) -> TensorDict:
        obs = data["observation"]
        action = data["action"]
        next_obs = data["next_observation"]
        reward = data["reward"]
        done = data["done"]

        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(next_obs)
            qf1_next_target = self.qf1_target(next_obs, next_action)
            qf2_next_target = self.qf2_target(next_obs, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_prob
            next_q = reward + (1 - done) * self.gamma * min_qf_next_target

        q1 = self.qf1(obs, action)
        q2 = self.qf2(obs, action)
        q1_loss = F.mse_loss(q1, next_q)
        q2_loss = F.mse_loss(q2, next_q)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(chain(self.qf1.parameters(), self.qf2.parameters()), self.max_grad_norm)
        self.critic_optimizer.step()
        q_mean = 0.5 * (q1.mean() + q2.mean()).detach().item()
        self.q_values.update(q_mean)
        self.critic_losses.update(critic_loss.detach().item())

    def update_actor(self, data: dict[str, Tensor]) -> TensorDict:
        obs = data["observation"]
        action, log_prob = self.actor.get_action(obs)
        q1 = self.qf1(obs, action)
        q2 = self.qf2(obs, action)
        min_q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - min_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.actor_losses.update(actor_loss.detach().item())

    def log(self, locs, width=80, pad=35):
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

        str = f" \033[1m Learning iteration {locs['iteration']}/{self.max_iterations} \033[0m "
        log_string = (
            f"""{"#" * width}\n"""
            f"""{str.center(width, " ")}\n\n"""
        )
        log_data = {
            "Train/mean_succ_rate": locs["mean_succ_rate"] if "mean_succ_rate" in locs else 0,
            "timesteps_total": self.global_step,
            "iteration": locs["iteration"],
        }

        if not timer.disabled:
            time_ = timer.compute()
            fps = (self.global_step - self.prev_global_step) / time_["time/iteration"]
            update_time = time_["time/update_model"]
            roll_out_time = time_["time/roll_out"]
            iteration_time = time_["time/iteration"]
            time_str = f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection {roll_out_time:.3f}s, learning {update_time:.3f}s)\n"""
            log_data["Train/fps"] = fps
            log_data["Time/roll_out"] = roll_out_time
            log_data["Time/update_model"] = update_time
            log_data["Time/iteration"] = iteration_time
            log_string += time_str
            timer.reset()
            self.prev_global_step = self.global_step

        if self.episode_lengths.len > 0:
            episode_length = self.episode_lengths.mean
            episode_reward = self.episode_rewards.mean
            log_string += f"""{"Mean episode length:":>{pad}} {episode_length:.1f}\n"""
            log_string += f"""{"Mean episode reward:":>{pad}} {episode_reward:.3f}\n"""
            log_data["Train/mean_episode_length"] = episode_length
            log_data["Train/mean_reward"] = episode_reward

        log_string += f"""{"Mean reward per step:":>{pad}} {locs["mean_reward"]:.3f}\n"""
        log_data["Train/mean_reward_per_step"] = locs["mean_reward"]

        if self.actor_losses.len > 0:
            actor_loss = self.actor_losses.mean
            q_value = self.q_values.mean
            critic_loss = self.critic_losses.mean
            log_string += f"""{"Mean actor loss:":>{pad}} {actor_loss:.3f}\n"""
            log_string += f"""{"Mean critic loss:":>{pad}} {critic_loss:.3f}\n"""
            log_string += f"""{"Mean Q value:":>{pad}} {q_value:.3f}\n"""
            log_data["Loss/mean_actor_loss"] = actor_loss
            log_data["Loss/mean_critic_loss"] = critic_loss
            log_data["Loss/mean_q_value"] = q_value

        tot_time = time.time() - self.start_time
        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.global_step}\n"""
            f"""{"Total time:":>{pad}} {tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {tot_time / (locs["iteration"] + 1) * (self.max_iterations - locs["iteration"]):.1f}s\n"""
        )
        print(log_string)
        if self.wandb_run is not None:
            self.wandb_run.log(log_data)

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution


class Actor(nn.Module):
    def __init__(self, obs_shape, actions_shape, model_cfg):
        super().__init__()

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            activation = get_activation(model_cfg["activation"])

        actor_layers = []
        actor_layers.append(nn.Linear(obs_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for dim in range(len(actor_hidden_dim)):
            if dim == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[dim], actions_shape * 2))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[dim], actor_hidden_dim[dim + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        print(self.actor)
        actor_weights = [np.sqrt(2)] * (len(actor_hidden_dim))
        actor_weights.append(0.01)
        self.init_weights(self.actor, actor_weights)

        self.log_std_min = model_cfg.get("log_std_min", -20)
        self.log_std_max = model_cfg.get("log_std_max", 2)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self, obs):
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, obs):
        # assume action is bounded in [-1, 1], which is the value range of tanh. Otherwise we need to add an affine transform

        ## Option 1
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        action_dist = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )  # ! use cache_size=1 to avoid atanh which could cause nan
        action_dist = Independent(action_dist, 1)
        action = action_dist.rsample()
        return action, action_dist.log_prob(action).unsqueeze(-1)

        ## Option 2
        # mean, log_std = self(obs)
        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # action = torch.tanh(x_t)
        # log_prob = normal.log_prob(x_t)
        # log_prob -= torch.log((1 - action**2) + 1e-6)  # ! 1e-6 to avoid nan
        # log_prob = log_prob.sum(1, keepdim=True)
        # return action, log_prob


class QNet(nn.Module):
    def __init__(self, obs_shape, actions_shape, model_cfg):
        super().__init__()
        if model_cfg is None:
            q_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            q_hidden_dim = model_cfg["q_hid_sizes"]
            activation = get_activation(model_cfg["activation"])

        q_layers = []
        q_layers.append(nn.Linear(np.prod(obs_shape) + np.prod(actions_shape), q_hidden_dim[0]))
        q_layers.append(activation)
        for dim in range(len(q_hidden_dim)):
            if dim == len(q_hidden_dim) - 1:
                q_layers.append(nn.Linear(q_hidden_dim[dim], 1))
            else:
                q_layers.append(nn.Linear(q_hidden_dim[dim], q_hidden_dim[dim + 1]))
                q_layers.append(activation)
        self.qnet = nn.Sequential(*q_layers)
        print(self.qnet)

        q_weights = [np.sqrt(2)] * (len(q_hidden_dim))
        q_weights.append(1.0)
        self.init_weights(self.qnet, q_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.qnet(x)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg):
        super().__init__()

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            critic_hidden_dim = model_cfg["vf_hid_sizes"]
            activation = get_activation(model_cfg["activation"])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(obs_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for dim in range(len(actor_hidden_dim)):
            if dim == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[dim], actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[dim], actor_hidden_dim[dim + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(obs_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for dim in range(len(critic_hidden_dim)):
            if dim == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[dim], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[dim], critic_hidden_dim[dim + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones((actions_shape,)))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self):
        raise NotImplementedError

    def act(self, observations):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(observations)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
        )

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, actions):
        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


class ActorCritic_RGB(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, state_shape, img_h=None, img_w=None):
        super().__init__()

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
            self.fix_img_encoder = False
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            critic_hidden_dim = model_cfg["vf_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
            self.fix_img_encoder = model_cfg.get("fix_img_encoder", False)

        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.img_h = img_h if img_h is not None else 256
        self.img_w = img_w if img_w is not None else 256
        self.num_img = (obs_shape[0] - state_shape) // (3 * self.img_h * self.img_w)

        # img encoder
        self.encoder_type = model_cfg.get("encoder_type", "resnet")
        if self.encoder_type == "resnet":
            self.visiual_encoder = torchvision.models.resnet18(pretrained=True)
            self.visual_feature_dim = self.visiual_encoder.fc.in_features
            del self.visiual_encoder.fc
            self.visiual_encoder.fc = nn.Identity()
            if self.fix_img_encoder:
                for param in self.visiual_encoder.parameters():
                    param.requires_grad = False
        elif self.encoder_type == "cnn":
            self.visiual_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                test_data = torch.zeros(1, 3, self.img_h, self.img_w)
                self.visual_feature_dim = self.visiual_encoder(test_data).shape[1]
            if self.fix_img_encoder:
                for param in self.visiual_encoder.parameters():
                    param.requires_grad = False
        else:
            raise NotImplementedError
        self.fc_shape = self.visual_feature_dim * self.num_img + state_shape

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.fc_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for dim in range(len(actor_hidden_dim)):
            if dim == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[dim], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[dim], actor_hidden_dim[dim + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.fc_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for dim in range(len(critic_hidden_dim)):
            if dim == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[dim], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[dim], critic_hidden_dim[dim + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self):
        raise NotImplementedError

    def act(self, observations):
        img = observations[:, self.state_shape :].view(-1, 3, self.img_h, self.img_w)
        # import cv2
        # import numpy as np
        # img0 = img[0].permute(1, 2, 0).cpu().numpy()  # Get the first environment's camera image
        # img_uint8 = (img0 * 255).astype(np.uint8) if img0.dtype != np.uint8 else img0
        # img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("camera0_image.png", img_bgr)
        # exit(0)
        if self.fix_img_encoder:
            with torch.no_grad():
                img_features = self.visiual_encoder(img)
        else:
            img_features = self.visiual_encoder(img)  # (batch_size * num_img, visual_feature_dim)
        img_features_flatten = img_features.view(
            observations.shape[0], -1
        )  # (batch_size, num_img * visual_feature_dim)
        state = observations[:, : self.state_shape]  # (batch_size, state_shape)
        observations = torch.cat((img_features_flatten, state), dim=-1)

        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(observations)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
        )

    def act_inference(self, observations):
        img = observations[:, self.state_shape :].view(-1, 3, self.img_h, self.img_w)
        if self.fix_img_encoder:
            with torch.no_grad():
                img_features = self.visiual_encoder(img)
        else:
            img_features = self.visiual_encoder(img)  # (batch_size * num_img, visual_feature_dim)
        img_features_flatten = img_features.view(
            observations.shape[0], -1
        )  # (batch_size, num_img * visual_feature_dim)
        state = observations[:, : self.state_shape]  # (batch_size, state_shape)
        observations = torch.cat((img_features_flatten, state), dim=-1)

        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, actions):
        img = observations[:, self.state_shape :].view(-1, 3, self.img_h, self.img_w)
        if self.fix_img_encoder:
            with torch.no_grad():
                img_features = self.visiual_encoder(img)
        else:
            img_features = self.visiual_encoder(img)  # (batch_size * num_img, visual_feature_dim)
        img_features_flatten = img_features.view(
            observations.shape[0], -1
        )  # (batch_size, num_img * visual_feature_dim)
        state = observations[:, : self.state_shape]  # (batch_size, state_shape)
        observations = torch.cat((img_features_flatten, state), dim=-1)

        actions_mean = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


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

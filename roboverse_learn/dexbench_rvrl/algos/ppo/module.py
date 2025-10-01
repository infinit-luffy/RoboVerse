import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, obs_type, obs_shape, actions_shape, initial_std, model_cfg, img_h=None, img_w=None):
        super().__init__()

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
            if "rgb" in obs_type:
                self.fix_img_encoder = False
                self.fix_actor_img_encoder = True
                self.visual_feature_dim = 512
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            critic_hidden_dim = model_cfg["vf_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
            if "rgb" in obs_type:
                self.fix_img_encoder = model_cfg.get("fix_img_encoder", False)
                self.fix_actor_img_encoder = model_cfg.get("fix_actor_img_encoder", True)
                self.visual_feature_dim = model_cfg.get("visual_feature_dim", 512)

        self.obs_shape = obs_shape
        self.obs_key = list(obs_shape.keys())
        self.state_key = [key for key in obs_shape.keys() if "state" in key]
        self.state_shape = sum([sum(obs_shape[key]) for key in self.state_key])
        self.visual_feature_dim = 0 if "rgb" not in obs_type else self.visual_feature_dim
        self.num_img = 0

        if "rgb" in obs_type:
            self.img_h = img_h if img_h is not None else 256
            self.img_w = img_w if img_w is not None else 256
            self.img_key = [key for key in obs_shape.keys() if "rgb" in key]
            assert len(self.img_key) == 1, "only support one rgb observation, shape 3xhxw"
            self.num_channel = [obs_shape[key][0] for key in self.img_key]
            self.num_img = len(self.img_key)

            # img encoder
            self.encoder_type = model_cfg.get("encoder_type", "resnet")
            if self.encoder_type == "resnet":
                self.visual_encoder = torchvision.models.resnet18(pretrained=True)
                self.visual_feature_dim = self.visual_encoder.fc.in_features
                del self.visual_encoder.fc
                self.visual_encoder.fc = nn.Identity()
                if self.fix_img_encoder:
                    for param in self.visual_encoder.parameters():
                        param.requires_grad = False
            elif self.encoder_type == "cnn":
                stages = model_cfg.get("stages", 5)
                input_dim = self.num_channel[0]

                kernel_size = model_cfg.get("kernel_size", [4])
                if isinstance(kernel_size, int):
                    kernel_size = [kernel_size] * stages
                elif isinstance(kernel_size, list):
                    if len(kernel_size) == 1:
                        kernel_size = kernel_size * stages
                    else:
                        assert len(kernel_size) == stages, "kernel_size should be an int or list of length stages"

                stride = model_cfg.get("stride", [2])
                if isinstance(stride, int):
                    stride = [stride] * stages
                elif isinstance(stride, list):
                    if len(stride) == 1:
                        stride = stride * stages
                    else:
                        assert len(stride) == stages, "stride should be an int or list of length stages"

                depth = model_cfg.get("depth", [32])
                if isinstance(depth, int):
                    depth = [depth] * stages
                elif isinstance(depth, list):
                    if len(depth) == 1:
                        depth = depth * stages
                    else:
                        assert len(depth) == stages, "depth should be an int or list of length stages"

                self.visual_encoder = []
                for i in range(stages):
                    padding = (kernel_size[i] - 1) // stride[i]
                    self.visual_encoder.append(
                        nn.Conv2d(
                            input_dim,
                            depth[i],
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding,
                            bias=False,
                        )
                    )
                    self.visual_encoder.append(nn.ReLU())
                    input_dim = depth[i]

                self.visual_encoder.append(nn.Flatten())
                self.visual_encoder = nn.Sequential(*self.visual_encoder)

                with torch.no_grad():
                    test_data = torch.zeros(1, self.num_channel[0], self.img_h, self.img_w)
                    out_dim = self.visual_encoder(test_data).shape[1]
                    self.visual_encoder.add_module("out", nn.Linear(out_dim, self.visual_feature_dim))
                    self.visual_encoder.add_module("out_activation", nn.ReLU())
                if self.fix_img_encoder:
                    for param in self.visual_encoder.parameters():
                        param.requires_grad = False
            else:
                raise NotImplementedError
            print(f"visual encoder: {self.visual_encoder}")
        self.fc_shape = self.visual_feature_dim * self.num_img + self.state_shape

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.fc_shape, actor_hidden_dim[0]))
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
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(actions_shape))

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
        feature = []
        for key in self.obs_key:
            if key in self.state_key:
                feature.append(observations[key])
            elif key in self.img_key:
                img = observations[key]
                # import cv2
                # import numpy as np

                # img0 = img[0].permute(1, 2, 0).cpu().numpy()  # Get the first environment's camera image
                # img_uint8 = (img0 * 255).astype(np.uint8) if img0.dtype != np.uint8 else img0
                # img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
                # cv2.imwrite("camera0_image.png", img_bgr)
                # exit(0)
                if self.fix_img_encoder:
                    with torch.no_grad():
                        img_features = self.visual_encoder(img)
                else:
                    img_features = self.visual_encoder(img)  # (batch_size * num_img, visual_feature_dim)
                img_features_flatten = img_features.view(
                    observations[key].shape[0], -1
                )  # (batch_size, num_img * visual_feature_dim)
                feature.append(img_features_flatten)

        feature = torch.cat(feature, dim=-1)

        actor_feature = feature.detach() if self.num_img > 0 and self.fix_actor_img_encoder else feature
        actions_mean = self.actor(actor_feature)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)
        actions = torch.tanh(actions)
        actions_log_prob -= torch.sum(
            torch.log(1 - actions * actions + 1e-6), dim=-1
        )  # Enforcing Action Bound, see appendix C of SAC paper

        value = self.critic(feature)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
        )

    def act_inference(self, observations):
        feature = []
        for key in self.obs_key:
            if key in self.state_key:
                feature.append(observations[key])
            elif key in self.img_key:
                img = observations[key]
                if self.fix_img_encoder or self.fix_actor_img_encoder:
                    with torch.no_grad():
                        img_features = self.visual_encoder(img)
                else:
                    img_features = self.visual_encoder(img)  # (batch_size * num_img, visual_feature_dim)
                img_features_flatten = img_features.view(
                    observations[key].shape[0], -1
                )  # (batch_size, num_img * visual_feature_dim)
                feature.append(img_features_flatten)

        feature = torch.cat(feature, dim=-1)

        actions_mean = self.actor(feature)
        actions_mean = torch.tanh(actions_mean)
        return actions_mean

    def evaluate(self, observations, actions):
        feature = []
        for key in self.obs_key:
            if key in self.state_key:
                feature.append(observations[key])
            elif key in self.img_key:
                img = observations[key]
                if self.fix_img_encoder:
                    with torch.no_grad():
                        img_features = self.visual_encoder(img)
                else:
                    img_features = self.visual_encoder(img)  # (batch_size * num_img, visual_feature_dim)
                img_features_flatten = img_features.view(
                    observations[key].shape[0], -1
                )  # (batch_size, num_img * visual_feature_dim)
                feature.append(img_features_flatten)

        feature = torch.cat(feature, dim=-1)

        actor_feature = feature.detach() if self.num_img > 0 and self.fix_actor_img_encoder else feature
        actions_mean = self.actor(actor_feature)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = torch.clamp(actions, -1 + 1e-6, 1 - 1e-6)
        pre_tanh_actions = 0.5 * (torch.log1p(actions) - torch.log1p(-actions))  # atanh
        actions_log_prob = distribution.log_prob(pre_tanh_actions)
        actions_log_prob -= torch.sum(
            torch.log(1 - actions * actions + 1e-6), dim=-1
        )  # Enforcing Action Bound, see appendix C of SAC paper
        entropy = distribution.entropy()

        value = self.critic(feature)

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

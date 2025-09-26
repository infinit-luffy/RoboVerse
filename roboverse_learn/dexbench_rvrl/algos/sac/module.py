import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(
        self,
        obs_type,
        obs_shape,
        actions_shape,
        model_cfg,
        visual_encoder=None,
        img_h=None,
        img_w=None,
    ):
        super().__init__()

        if "rgb" in obs_type:
            assert visual_encoder is not None, "Must provide visual encoder for RGB observations"

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
            if "rgb" in obs_type:
                self.fix_img_encoder = False
                self.fix_actor_img_encoder = True
                self.visual_feature_dim = visual_encoder.visual_feature_dim
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
            if "rgb" in obs_type:
                self.fix_img_encoder = model_cfg.get("fix_img_encoder", False)
                self.fix_actor_img_encoder = model_cfg.get("fix_actor_img_encoder", True)
                self.visual_feature_dim = visual_encoder.visual_feature_dim

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

            ## image encoder
            self.visual_encoder = visual_encoder
            self.visual_feature_dim = visual_encoder.visual_feature_dim

        self.fc_shape = self.visual_feature_dim * self.num_img + self.state_shape

        actor_layers = []
        actor_layers.append(nn.Linear(self.fc_shape, actor_hidden_dim[0]))
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

        feature = []
        for key in self.obs_key:
            if key in self.state_key:
                feature.append(obs[key])
            elif key in self.img_key:
                img = obs[key].view(-1, 3, self.img_h, self.img_w)
                if self.fix_img_encoder:
                    with torch.no_grad():
                        img_features = self.visual_encoder(img)
                else:
                    img_features = self.visual_encoder(img)  # (batch_size * num_img, visual_feature_dim)
                img_features_flatten = img_features.view(
                    obs[key].shape[0], -1
                )  # (batch_size, num_img * visual_feature_dim)
                feature.append(img_features_flatten)

        feature = torch.cat(feature, dim=-1)

        mean, log_std = self(feature)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - action**2) + 1e-6)  # ! 1e-6 to avoid nan
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class QNet(nn.Module):
    def __init__(
        self,
        obs_type,
        obs_shape,
        actions_shape,
        model_cfg,
        visual_encoder=None,
        img_h=None,
        img_w=None,
    ):
        super().__init__()

        if "rgb" in obs_type:
            assert visual_encoder is not None, "Must provide visual encoder for RGB observations"

        if model_cfg is None:
            q_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
            if "rgb" in obs_type:
                self.fix_img_encoder = False
                self.visual_feature_dim = visual_encoder.visual_feature_dim
        else:
            q_hidden_dim = model_cfg["q_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
            if "rgb" in obs_type:
                self.fix_img_encoder = model_cfg.get("fix_img_encoder", False)
                self.visual_feature_dim = visual_encoder.visual_feature_dim

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

            ## image encoder
            self.visual_encoder = visual_encoder
            self.visual_feature_dim = visual_encoder.visual_feature_dim

        self.fc_shape = self.visual_feature_dim * self.num_img + self.state_shape + actions_shape

        q_layers = []
        q_layers.append(nn.Linear(self.fc_shape, q_hidden_dim[0]))
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
        feature = []
        for key in self.obs_key:
            if key in self.state_key:
                feature.append(obs[key])
            elif key in self.img_key:
                img = obs[key].view(-1, 3, self.img_h, self.img_w)
                if self.fix_img_encoder:
                    with torch.no_grad():
                        img_features = self.visual_encoder(img)
                else:
                    img_features = self.visual_encoder(img)  # (batch_size * num_img, visual_feature_dim)
                img_features_flatten = img_features.view(
                    obs[key].shape[0], -1
                )  # (batch_size, num_img * visual_feature_dim)
                feature.append(img_features_flatten)
        feature = torch.cat(feature, dim=-1)

        x = torch.cat([feature, action], dim=-1)
        return self.qnet(x)


class RGB_Encoder(nn.Module):
    def __init__(self, model_cfg, img_h=None, img_w=None):
        super().__init__()
        self.encoder_type = model_cfg.get("encoder_type", "resnet")
        self.visual_feature_dim = model_cfg.get("visual_feature_dim", 512)
        self.img_h = img_h if img_h is not None else 256
        self.img_w = img_w if img_w is not None else 256

        if self.encoder_type == "resnet":
            self.encoder = torchvision.models.resnet18(pretrained=True)
            self.visual_feature_dim = self.encoder.fc.in_features
            del self.encoder.fc  # delete the original fully connected layer
            self.encoder.fc = nn.Identity()
            print("=> using resnet18 as visual encoder")
        elif self.encoder_type == "cnn":
            self.min_res = model_cfg.get("min_res", 4)
            stages = int(np.log2(self.img_h // self.min_res))
            kernel_size = model_cfg.get("kernel_size", 4)
            input_dim = self.num_channel[0]
            depth = model_cfg.get("depth", 32)
            output_dim = depth
            self.visual_encoder = []
            self.h = self.img_h
            self.w = self.img_w
            for i in range(stages):
                self.visual_encoder.append(
                    nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=2, padding=1)
                )
                self.visual_encoder.append(nn.ReLU())
                input_dim = output_dim
                output_dim = min(512, output_dim * 2)
                self.h = self.h // 2
                self.w = self.w // 2

            self.visual_encoder.append(nn.Flatten())
            self.visual_encoder = nn.Sequential(*self.visual_encoder)

            with torch.no_grad():
                test_data = torch.zeros(1, self.num_channel[0], self.img_h, self.img_w)
                out_dim = self.visual_encoder(test_data).shape[1]
                self.visual_encoder.add_module("out", nn.Linear(out_dim, self.visual_feature_dim))
                self.visual_encoder.add_module("out_activation", nn.ReLU())
            print("=> using custom cnn as visual encoder")
        else:
            raise NotImplementedError

    def forward(self, img):
        # import cv2
        # import numpy as np

        # img0 = img[0].permute(1, 2, 0).cpu().numpy()  # Get the first environment's camera image
        # img_uint8 = (img0 * 255).astype(np.uint8) if img0.dtype != np.uint8 else img0
        # img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("camera0_image.png", img_bgr)
        # exit(0)
        return self.encoder(img)


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

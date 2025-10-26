import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.distributions import Normal
import re
from torch.distributions.utils import _standard_normal
import torch.nn.functional as F

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)

class TruncatedNormal(Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, data):
        for key in data["observation"].keys():
            if "rgb" in key:
                data["observation"][key] = self.random_shift(data["observation"][key])
                data["next_observation"][key] = self.random_shift(data["next_observation"][key])
        return data
                
    def random_shift(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

    
class RGB_Encoder(nn.Module):
    def __init__(self, obs_type, obs_shape, model_cfg, img_h=None, img_w=None):
        super().__init__()
        self.encoder_type = model_cfg.get("encoder_type", "resnet")
        self.visual_feature_dim = model_cfg.get("visual_feature_dim", 512)
        self.img_h = img_h if img_h is not None else 256
        self.img_w = img_w if img_w is not None else 256
        self.img_key = [key for key in obs_shape.keys() if "rgb" in key]
        assert len(self.img_key) == 1, "only support one rgb observation, shape 3xhxw"
        self.num_channel = [obs_shape[key][0] for key in self.img_key]
        self.num_img = len(self.img_key)

        if self.encoder_type == "resnet":
            self.visual_encoder = torchvision.models.resnet18(pretrained=True)
            self.visual_feature_dim = self.encoder.fc.in_features
            del self.visual_encoder.fc  # delete the original fully connected layer
            self.visual_encoder.fc = nn.Identity()
            print("=> using resnet18 as visual encoder")
        elif self.encoder_type == "cnn":
            cnn_cfg = model_cfg.get("cnn", {})
            stages = cnn_cfg.get("stages", 5)
            input_dim = self.num_channel[0]

            kernel_size = cnn_cfg.get("kernel_size", [4])
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size] * stages
            elif isinstance(kernel_size, list):
                if len(kernel_size) == 1:
                    kernel_size = kernel_size * stages
                else:
                    assert len(kernel_size) == stages, "kernel_size should be an int or list of length stages"

            stride = cnn_cfg.get("stride", [2])
            if isinstance(stride, int):
                stride = [stride] * stages
            elif isinstance(stride, list):
                if len(stride) == 1:
                    stride = stride * stages
                else:
                    assert len(stride) == stages, "stride should be an int or list of length stages"

            depth = cnn_cfg.get("depth", [32])
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
            print("=> using custom cnn as visual encoder")
        else:
            raise NotImplementedError

    def forward(self, img):
        # import cv2
        # import numpy as np

        # img0 = img[0].permute(1, 2, 0).cpu().numpy()  # Get the first environment's camera image
        # img0 = img0[:, :, :3]  # Take only the first 3 channels (RGB)
        # img_uint8 = (img0 * 255).astype(np.uint8) if img0.dtype != np.uint8 else img0
        # img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("drq_image.png", img_bgr)
        # exit(0)
        if self.encoder_type == "cnn":
            img = img - 0.5
        return self.visual_encoder(img)
    
class Encoder(nn.Module):
    def __init__(self, obs_type, obs_shape, model_cfg, img_h=None, img_w=None):
        super().__init__()
        
        self.obs_type = obs_type
        if "rgb" in obs_type:
            self.visual_encoder = RGB_Encoder(
                obs_type, obs_shape, model_cfg, img_h, img_w
            )
            
        self.obs_shape = obs_shape
        self.obs_key = list(obs_shape.keys())
        self.state_key = [key for key in obs_shape.keys() if "state" in key]
        self.state_shape = sum([np.prod(obs_shape[key]) for key in self.state_key])
        self.model_cfg = model_cfg
        self.img_h = img_h
        self.img_w = img_w

        self.repr_dim = model_cfg.get("repr_dim", 1024)
        self.use_state = model_cfg.get("use_state", True)
        state_dim = 0
        if self.use_state:
            for key in obs_shape.keys():
                if "state" in key:
                    state_dim += np.prod(obs_shape[key])

        if self.visual_encoder is not None:
            visual_feature_dim = self.visual_encoder.visual_feature_dim * self.visual_encoder.num_img
        else:
            visual_feature_dim = 0

        self.repr_dim = visual_feature_dim + state_dim
        self.feature_dim = model_cfg.get("feature_dim", 1024)

        self.fc = nn.Sequential(
            nn.Linear(self.repr_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh(),
        )

        self.apply(weight_init)

    def forward(self, obs):
        features = []
        for key in self.obs_shape.keys():
            if "rgb" in key:
                img = obs[key]
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)  # Add batch dimension if missing
                img_repr = self.visual_encoder(img)
                features.append(img_repr)
            elif self.use_state and "state" in key:
                if len(obs[key].shape) == 1:
                    state = obs[key].unsqueeze(0)  # Add batch dimension if missing
                else:
                    state = obs[key]
                features.append(state)
        feature = torch.cat(features, dim=-1)
        feature = self.fc(feature)
        return feature



class Actor(nn.Module):
    def __init__(
        self,
        obs_type,
        obs_shape,
        actions_shape,
        model_cfg,
    ):
        super().__init__()

        if model_cfg is None:
            actor_hidden_dim = [256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
            
        self.feature_dim = model_cfg.get("feature_dim", 1024)
        actor_layers = []
        actor_layers.append(
            nn.Linear(self.feature_dim, actor_hidden_dim[0])
        )
        actor_layers.append(activation)
        for dim in range(len(actor_hidden_dim)):
            if dim == len(actor_hidden_dim) - 1:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[dim], actions_shape)
                )
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[dim], actor_hidden_dim[dim + 1])
                )
                actor_layers.append(activation)

        self.actor = nn.Sequential(*actor_layers)
        print(self.actor)
        
        self.apply(weight_init)

    def forward(self, feature, std):
        mean = self.actor(feature)
        mean = torch.tanh(mean)
        std = torch.ones_like(mean) * std

        dist = TruncatedNormal(mean, std)
        return dist


class Critic(nn.Module):
    def __init__(
        self,
        obs_type,
        obs_shape,
        actions_shape,
        model_cfg,
    ):
        super().__init__()
        
        if model_cfg is None:
            q_hidden_dim = [256, 256]
            activation = get_activation("selu")
        else:
            q_hidden_dim = model_cfg["q_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
            
        self.feature_dim = model_cfg.get("feature_dim", 1024)
        q_layers = []
        q_layers.append(
            nn.Linear(self.feature_dim + actions_shape, q_hidden_dim[0])
        )
        q_layers.append(activation)
        for dim in range(len(q_hidden_dim)):
            if dim == len(q_hidden_dim) - 1:
                q_layers.append(
                    nn.Linear(q_hidden_dim[dim], 1)
                )
            else:
                q_layers.append(
                    nn.Linear(q_hidden_dim[dim], q_hidden_dim[dim + 1])
                )
                q_layers.append(activation)

        self.Q1 = nn.Sequential(*q_layers)

        self.Q2 = nn.Sequential(*q_layers)

        self.apply(weight_init)

    def forward(self, feature, action):
        feature = torch.cat([feature, action], dim=-1)
        q1 = self.Q1(feature)
        q2 = self.Q2(feature)

        return q1, q2

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

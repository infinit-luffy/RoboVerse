from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as log
from rich.logging import RichHandler
from torch import Tensor
from torch.distributions import (
    Bernoulli,
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
)
from torch.distributions.utils import probs_to_logits

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class LayerNormChannelLast(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class MSEDistribution:
    """
    Copied from https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/utils/distribution.py#L196
    """

    def __init__(self, mode: Tensor, dims: int, agg: str = "sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]

    @property
    def mode(self) -> Tensor:
        return self._mode

    @property
    def mean(self) -> Tensor:
        return self._mode

    def log_prob(self, value: Tensor) -> Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class LayerNormGRUCell(nn.Module):
    """A GRU cell with a LayerNorm
    copied from https://github.com/Eclectic-Sheep/sheeprl/blob/4441dbf4bcd7ae0daee47d35fb0660bc1fe8bd4b/sheeprl/models/models.py#L331
    which was taken from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py#L317.

    This particular GRU cell accepts 3-D inputs, with a sequence of length 1, and applies
    a LayerNorm after the projection of the inputs.

    Args:
        input_size (int): the input size.
        hidden_size (int): the hidden state size
        bias (bool, optional): whether to apply a bias to the input projection.
            Defaults to True.
        batch_first (bool, optional): whether the first dimension represent the batch dimension or not.
            Defaults to False.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to nn.Identiy.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {}.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        layer_norm_cls: Callable[..., nn.Module] = nn.Identity,
        layer_norm_kw: dict[str, Any] = {},  # noqa: B006
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=self.bias)
        # Avoid multiple values for the `normalized_shape` argument
        layer_norm_kw.pop("normalized_shape", None)
        self.layer_norm = layer_norm_cls(3 * hidden_size, **layer_norm_kw)

    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {input.shape[int(self.batch_first)]}"
                )
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        assert input.dim() in (
            1,
            2,
        ), f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        hx = hx.unsqueeze(0) if not is_batched else hx

        input = torch.cat((hx, input), -1)
        x = self.linear(input)
        x = self.layer_norm(x)
        reset, cand, update = torch.chunk(x, 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        hx = update * cand + (1 - update) * hx

        if not is_batched:
            hx = hx.squeeze(0)
        elif is_3d:
            hx = hx.unsqueeze(0)

        return hx


# From https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py
@torch.jit.script
def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class TwoHotEncodingDistribution:
    """
    Copied from https://github.com/Eclectic-Sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/utils/distribution.py#L224
    """

    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
        transfwd: Callable[[Tensor], Tensor] = symlog,
        transbwd: Callable[[Tensor], Tensor] = symexp,
    ):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self._batch_shape = logits.shape[: len(logits.shape) - dims]
        self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)

    @property
    def mean(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    @property
    def mode(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x)
        # below in [-1, len(self.bins) - 1]
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        # above in [0, len(self.bins)]
        above = below + 1

        # above in [0, len(self.bins) - 1]
        above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        # below in [0, len(self.bins) - 1]
        below = torch.maximum(below, torch.zeros_like(below))

        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None]
            + F.one_hot(above, len(self.bins)) * weight_above[..., None]
        ).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (target * log_pred).sum(dim=self.dims)


class SafeBernoulli(Bernoulli):
    @property
    def mode(self) -> Tensor:
        mode = (self.probs >= 0.5).to(self.probs)
        return mode


class Moments(nn.Module):
    """
    Copied from https://github.com/Eclectic-Sheep/sheeprl/blob/419c7ce05b67b0fd89b62ae0b73b71b3f7a96514/sheeprl/algos/dreamer_v3/utils.py#L40
    """

    def __init__(
        self, decay: float = 0.99, max_: float = 1.0, percentile_low: float = 0.05, percentile_high: float = 0.95
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = torch.tensor(max_)
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor) -> Any:
        low = torch.quantile(x, self._percentile_low)
        high = torch.quantile(x, self._percentile_high)
        with torch.no_grad():  # ! stop tracing gradient, otherwise will cause memory leak
            self.low = self._decay * self.low + (1 - self._decay) * low
            self.high = self._decay * self.high + (1 - self._decay) * high
        invscale = torch.max(1 / self._max, self.high - self.low)
        return self.low.detach(), invscale.detach()


########################################################
## Networks
########################################################
# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L929
def init_weights(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
def uniform_init_weights(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


class Encoder(nn.Module):
    def __init__(self, model_cfg, state_shape, symlog_input=True, img_h=None, img_w=None):
        super().__init__()
        self.state_shape = state_shape
        self.symlog_input = symlog_input
        self.img_h = img_h
        self.img_w = img_w
        if img_h is not None and img_w is not None:
            self.min_res = model_cfg.get("min_res", 4)
            stages = int(np.log2(img_h) - np.log2(self.min_res))
            kernel_size = model_cfg.get("kernel_size", 4)
            input_dim = 3
            depth = model_cfg.get("depth", 16)
            output_dim = depth
            self.visual_encoder = []
            self.h = img_h
            self.w = img_w
            for stage in range(stages):
                self.visual_encoder.append(
                    nn.Conv2d(
                        input_dim,
                        output_dim,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=1,
                        bias=False,
                    )
                )
                self.visual_encoder.append(LayerNormChannelLast(output_dim, eps=1e-3))
                self.visual_encoder.append(nn.SiLU())
                input_dim = output_dim
                output_dim = min(512, output_dim * 2)
                self.h = self.h // 2
                self.w = self.w // 2
            self.visual_encoder = nn.Sequential(*self.visual_encoder)
            self.visual_encoder.apply(init_weights)
            self.visual_feature_dim = input_dim * self.h * self.w
        if model_cfg is None:
            hidden_dim = [256, 256, 256]
        else:
            hidden_dim = model_cfg.get("hidden_dim")
        self.mlp_encoder = []
        input_dim = np.prod(state_shape)
        for hdim in hidden_dim:
            self.mlp_encoder.append(nn.Linear(input_dim, hdim, bias=False))
            self.mlp_encoder.append(nn.LayerNorm(hdim, eps=1e-3))
            self.mlp_encoder.append(nn.SiLU())
            input_dim = hdim
        self.mlp_encoder = nn.Sequential(*self.mlp_encoder)
        self.mlp_encoder.apply(init_weights)
        self.mlp_feature_dim = input_dim
        self.output_dim = self.visual_feature_dim + self.mlp_feature_dim

    def forward(self, obs: Tensor) -> Tensor:
        B = obs.shape[0]
        vector_obs = obs[:, : self.state_shape]
        if self.img_h is not None and self.img_w is not None:
            image_obs = obs[:, self.state_shape :].reshape(B, 3, self.img_h, self.img_w) / 255.0 - 0.5
            visual_embedded = self.visual_encoder(image_obs).reshape(B, -1)
            vector_embedded = self.mlp_encoder(vector_obs)
            embedded_obs = torch.cat([visual_embedded, vector_embedded], dim=-1)
        else:
            if self.symlog_input:
                vector_obs = symlog(vector_obs)
            vector_embedded = self.mlp_encoder(vector_obs)
            embedded_obs = vector_embedded
        return embedded_obs.reshape(B, -1)  # flatten the last 3 dimensions C, H, W


class Decoder(nn.Module):
    def __init__(self, deterministic_size, stochastic_size, model_cfg, state_shape, img_h, img_w):
        super().__init__()
        self.state_shape = state_shape
        self.img_h = img_h
        self.img_w = img_w
        if self.img_h is not None and self.img_w is not None:
            self.min_res = model_cfg.get("min_res", 4)
            stages = int(np.log2(img_h) - np.log2(self.min_res))
            kernel_size = model_cfg.get("kernel_size", 4)
            depth = model_cfg.get("depth", 16)
            visual_feature_dim = self.min_res**2 * depth * 2 ** (stages - 1)
            self.linear_layer = nn.Linear(deterministic_size + stochastic_size, visual_feature_dim)
            self.linear_layer.apply(uniform_init_weights(1.0))
            input_dim = visual_feature_dim // (self.min_res * self.min_res)  # depth * 2**(stages-1)
            output_dim = input_dim // 2
            self.visual_decoder = []
            for stage in range(stages - 1):
                if stage != 0:
                    input_dim = 2 ** (stages - (stage - 1) - 2) * depth
                pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
                pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
                self.visual_decoder.append(
                    nn.ConvTranspose2d(
                        input_dim,
                        output_dim,
                        kernel_size,
                        2,
                        padding=(pad_h, pad_w),
                        output_padding=(outpad_h, outpad_w),
                        bias=False,
                    )
                )
                self.visual_decoder.append(LayerNormChannelLast(output_dim, eps=1e-3))
                self.visual_decoder.append(nn.SiLU())
                input_dim = output_dim
                output_dim = max(depth, output_dim // 2)
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            self.visual_decoder.append(
                nn.ConvTranspose2d(
                    input_dim,
                    3,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=True,
                )
            )
            self.visual_decoder = nn.Sequential(*self.visual_decoder)
            [m.apply(init_weights) for m in self.visual_decoder[:-1]]
            self.visual_decoder[-1].apply(uniform_init_weights(1.0))
        if model_cfg is None:
            hidden_dim = [256, 256, 256]
        else:
            hidden_dim = model_cfg.get("hidden_dim")
        self.mlp_decoder = []
        input_dim = deterministic_size + stochastic_size
        for hdim in hidden_dim:
            self.mlp_decoder.append(nn.Linear(input_dim, hdim, bias=False))
            self.mlp_decoder.append(nn.LayerNorm(hdim, eps=1e-3))
            self.mlp_decoder.append(nn.SiLU())
            input_dim = hdim
        self.mlp_decoder.append(nn.Linear(input_dim, np.prod(state_shape), bias=True))
        self.mlp_decoder = nn.Sequential(*self.mlp_decoder)
        [m.apply(init_weights) for m in self.mlp_decoder[:-1]]
        self.mlp_decoder[-1].apply(uniform_init_weights(0.0))

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([posterior, deterministic], dim=-1)
        if self.img_h is not None and self.img_w is not None:
            input_shape = x.shape
            x = x.flatten(0, 1)
            reconstructed_visual_obs = self.decoder(x)
            reconstructed_visual_obs = reconstructed_visual_obs.unflatten(0, input_shape[:2]) + 0.5
            reconstructed_vector_obs = self.mlp_decoder(x)
            reconstructed_obs = {
                "img": reconstructed_visual_obs,
                "state": reconstructed_vector_obs,
            }
        else:
            input_shape = x.shape
            x = x.flatten(0, 1)
            reconstructed_vector_obs = self.mlp_decoder(x)
            reconstructed_obs = reconstructed_vector_obs
            reconstructed_obs = {"state": reconstructed_obs.unflatten(0, input_shape[:2])}

        return reconstructed_obs


class RecurrentModel(nn.Module):
    def __init__(self, action_dim, deterministic_size, stochastic_size, model_cfg):
        super().__init__()
        if model_cfg is None:
            hidden_dim = 512
        else:
            hidden_dim = model_cfg.get("recurrent_hidden_dim", 512)
        self.mlp = nn.Sequential(
            nn.Linear(stochastic_size + action_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU(),
        )
        self.recurrent = LayerNormGRUCell(
            hidden_dim, deterministic_size, bias=False, layer_norm_cls=nn.LayerNorm, layer_norm_kw={"eps": 1e-3}
        )
        self.mlp.apply(init_weights)
        self.recurrent.apply(init_weights)

    def forward(self, state: Tensor, action: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=1)
        x = self.mlp(x)
        x = self.recurrent(x, deterministic)
        return x


class TransitionModel(nn.Module):
    def __init__(self, deterministic_size, stochastic_size, stochastic_length, stochastic_classes, model_cfg):
        super().__init__()
        if model_cfg is None:
            hidden_dim = 512
        else:
            hidden_dim = model_cfg.get("recurrent_hidden_dim", 512)
        self.net = nn.Sequential(
            nn.Linear(deterministic_size, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU(),
            nn.Linear(hidden_dim, stochastic_size),
        )
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(1.0))
        self.deterministic_size = deterministic_size
        self.stochastic_size = stochastic_size
        self.stochastic_length = stochastic_length
        self.stochastic_classes = stochastic_classes

    def forward(self, deterministic: Tensor) -> tuple[Distribution, Tensor]:
        logits = self.net(deterministic).view(-1, self.stochastic_length, self.stochastic_classes)
        logits = self._unimix(logits)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        return dist, logits.view(-1, self.stochastic_size)

    def _unimix(self, logits: Tensor) -> Tensor:
        probs = logits.softmax(dim=-1)
        uniform = torch.ones_like(probs) / self.stochastic_classes
        probs = 0.99 * probs + 0.01 * uniform
        logits = probs_to_logits(probs)
        return logits


class RepresentationModel(nn.Module):
    def __init__(
        self, deterministic_size, stochastic_size, stochastic_length, stochastic_classes, embedded_obs_size, model_cfg
    ):
        super().__init__()
        if model_cfg is None:
            hidden_dim = 1024
        else:
            hidden_dim = model_cfg.get("representation_hidden_dim", 1024)
        self.net = nn.Sequential(
            nn.Linear(embedded_obs_size + deterministic_size, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU(),
            nn.Linear(hidden_dim, stochastic_size),
        )
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(1.0))
        self.deterministic_size = deterministic_size
        self.stochastic_size = stochastic_size
        self.stochastic_length = stochastic_length
        self.stochastic_classes = stochastic_classes
        self.embedded_obs_size = embedded_obs_size

    def forward(self, embedded_obs: Tensor, deterministic: Tensor) -> tuple[Distribution, Tensor]:
        x = torch.cat([embedded_obs, deterministic], dim=1)
        logits = self.net(x).view(-1, self.stochastic_length, self.stochastic_classes)
        logits = self._unimix(logits)
        dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        return dist, logits.view(-1, self.stochastic_size)

    def _unimix(self, logits: Tensor) -> Tensor:
        probs = logits.softmax(dim=-1)
        uniform = torch.ones_like(probs) / self.stochastic_classes
        probs = 0.99 * probs + 0.01 * uniform
        logits = probs_to_logits(probs)
        return logits


class RewardPredictor(nn.Module):
    def __init__(self, deterministic_size, stochastic_size, bins, model_cfg):
        super().__init__()
        if model_cfg is None:
            hidden_dim = [512, 512]
        else:
            hidden_dim = model_cfg.get("value_hidden_dim", [512, 512])
        self.net = []
        input_dim = deterministic_size + stochastic_size
        for hdim in hidden_dim:
            self.net.append(nn.Linear(input_dim, hdim, bias=False))
            self.net.append(nn.LayerNorm(hdim, eps=1e-3))
            self.net.append(nn.SiLU())
            input_dim = hdim
        self.net.append(nn.Linear(input_dim, bins))
        self.net = nn.Sequential(*self.net)
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(0.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        input_shape = posterior.shape
        posterior = posterior.flatten(0, 1)
        deterministic = deterministic.flatten(0, 1)
        x = torch.cat([posterior, deterministic], dim=1)
        predicted_reward_bins = self.net(x)
        predicted_reward_bins = predicted_reward_bins.unflatten(0, input_shape[:2])
        return predicted_reward_bins


class ContinueModel(nn.Module):
    def __init__(self, deterministic_size, stochastic_size, model_cfg):
        super().__init__()
        if model_cfg is None:
            hidden_dim = [512, 512]
        else:
            hidden_dim = model_cfg.get("continue_hidden_dim", [512, 512])
        self.net = []
        input_dim = deterministic_size + stochastic_size
        for hdim in hidden_dim:
            self.net.append(nn.Linear(input_dim, hdim, bias=False))
            self.net.append(nn.LayerNorm(hdim, eps=1e-3))
            self.net.append(nn.SiLU())
            input_dim = hdim
        self.net.append(nn.Linear(input_dim, 1))
        [m.apply(init_weights) for m in self.net[:-1]]
        self.net[-1].apply(uniform_init_weights(1.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        input_shape = posterior.shape
        posterior = posterior.flatten(0, 1)
        deterministic = deterministic.flatten(0, 1)
        x = torch.cat([posterior, deterministic], dim=1)
        logits = self.net(x)
        return logits.unflatten(0, input_shape[:2])


class Actor(nn.Module):
    def __init__(self, action_dim, deterministic_size, stochastic_size, model_cfg):
        super().__init__()
        if model_cfg is None:
            hidden_dim = [256, 256, 256]
        else:
            hidden_dim = model_cfg.get("actor_hidden_dim", [256, 256, 256])
        self.actor = []
        input_dim = deterministic_size + stochastic_size
        for hdim in hidden_dim:
            self.actor.append(nn.Linear(input_dim, hdim, bias=False))
            self.actor.append(nn.LayerNorm(hdim, eps=1e-3))
            self.actor.append(nn.SiLU())
            input_dim = hdim
        self.actor.append(nn.Linear(input_dim, action_dim * 2))
        self.actor = nn.Sequential(*self.actor)
        [m.apply(init_weights) for m in self.actor[:-1]]
        self.actor[-1].apply(uniform_init_weights(1.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Distribution:
        x = torch.cat([posterior, deterministic], dim=-1)
        mean, std = self.actor(x).chunk(2, dim=-1)
        std_min, std_max = 0.1, 1
        mean = F.tanh(mean)
        std = std_min + (std_max - std_min) * F.sigmoid(std + 2.0)
        action_dist = Independent(Normal(mean, std), 1)
        return action_dist


class Critic(nn.Module):
    def __init__(self, deterministic_size, stochastic_size, bins, model_cfg):
        super().__init__()
        if model_cfg is None:
            hidden_dim = [512, 512]
        else:
            hidden_dim = model_cfg.get("value_hidden_dim", [512, 512])
        self.critic = []
        input_dim = deterministic_size + stochastic_size
        for hdim in hidden_dim:
            self.critic.append(nn.Linear(input_dim, hdim, bias=False))
            self.critic.append(nn.LayerNorm(hdim, eps=1e-3))
            self.critic.append(nn.SiLU())
            input_dim = hdim
        self.critic.append(nn.Linear(input_dim, bins))
        self.critic = nn.Sequential(*self.critic)
        [m.apply(init_weights) for m in self.critic[:-1]]
        self.critic[-1].apply(uniform_init_weights(0.0))

    def forward(self, posterior: Tensor, deterministic: Tensor) -> Tensor:
        x = torch.cat([posterior, deterministic], dim=-1)
        predicted_value_bins = self.critic(x)
        return predicted_value_bins

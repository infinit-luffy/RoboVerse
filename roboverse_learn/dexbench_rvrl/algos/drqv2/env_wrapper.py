from ...envs.dex_env import DexEnv
from collections import deque
import torch
from gymnasium import spaces
import numpy as np

class DrqDexEnv:
    """Dex Environment Wrapper for Drqv2 tasks.
    Stacks num_frames frames for each observation."""

    def __init__(self, num_frames: int, env: DexEnv):
        assert env.num_envs == 1, "DrqDexEnv only supports single environment."
        self._frames = deque([], maxlen=num_frames)
        # self._frames = {}
        # for key in obs_shape.keys():
        #     if "rgb" in key:
        #         self._frames[key] = torch.zeros((env.num_envs, num_frames, *obs_shape[key]), dtype=torch.float32)
        # self.env_mask = torch.zeros((env.num_envs,), dtype=torch.int8)
        self._num_frames = num_frames
        self.env = env
        self.single_observation_space = spaces.Dict({})
        for k, v in env.single_observation_space.spaces.items():
            if "rgb" in k:
                shape = (num_frames * v.shape[0], *v.shape[1:])
                self.single_observation_space.spaces[k] = spaces.Box(low=-5.0, high=5.0, shape=shape, dtype=np.float32)
            else:
                self.single_observation_space.spaces[k] = v
            
    def reset(self):
        observations = self.env.reset()
        for key, value in observations.items():
            observations[key] = value[0]
            if "rgb" in key:
                for _ in range(self._num_frames):
                    self._frames.append(observations[key])
                observations[key] = torch.cat(list(self._frames), dim=0)
        return observations

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(self._frames) == self._num_frames
        step_actions = actions.unsqueeze(0)  # Add batch dimension
        observations, rewards, dones, dones, info = self.env.step(step_actions)
        for key, value in observations.items():
            observations[key] = value[0]
            import cv2
            import numpy as np

            img0 = value[0].permute(1, 2, 0).cpu().numpy()  # Get the first environment's camera image
            img_uint8 = (img0 * 255).astype(np.uint8) if img0.dtype != np.uint8 else img0
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite("drq_image.png", img_bgr)
            # exit(0)
            if "rgb" in key:
                if dones[0]:
                    for _ in range(self._num_frames):
                        self._frames.append(observations[key])
                else:
                    self._frames.append(observations[key])
                observations[key] = torch.cat(list(self._frames), dim=0)
        return observations, rewards[0], dones[0], dones[0], info
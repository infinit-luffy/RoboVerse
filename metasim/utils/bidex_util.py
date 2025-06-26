"""Sub-module containing utilities for computing bidex shadow hand rewards."""

import torch

from metasim.utils.math import quat_from_angle_axis, quat_mul


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """Randomize the rotation of the object.

    Args:
        rand0 (tensor): Random value for rotation around x-axis
        rand1 (tensor): Random value for rotation around y-axis
        x_unit_tensor (tensor): Unit vector along x-axis
        y_unit_tensor (tensor): Unit vector along y-axis
    """
    return quat_mul(
        quat_from_angle_axis(rand0 * torch.pi, x_unit_tensor), quat_from_angle_axis(rand1 * torch.pi, y_unit_tensor)
    )

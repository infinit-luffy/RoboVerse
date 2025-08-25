"""Sub-module containing utilities for computing bidex shadow hand rewards."""

import jax
import jax.dlpack
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
import torch
import torch.utils.dlpack

from metasim.utils.math import quat_from_angle_axis, quat_mul


def torch_to_jax(t: torch.Tensor) -> jax.Array:
    """Turn a PyTorch Tensor into a JAX Array."""
    if not t.is_contiguous():
        t = t.contiguous()
    shape = t.shape
    t_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(torch.flatten(t)))
    return t_jax.reshape(shape)


def jax_to_torch(x: jax.Array) -> torch.Tensor:
    """Turn a JAX Array into a PyTorch Tensor."""
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))


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


def _solve_ik_single_env(
    robot: pk.Robot,
    init_q: jax.Array,  # (num_joints,)
    target_wxyz: jax.Array,  # (target_num, 4)
    target_position: jax.Array,  # (target_num, 3)
    target_link_indices: jax.Array,  # (target_num,)
) -> jax.Array:
    """Solve IK for a single environment and multiple targets(dex hand for instance).

    Args:
        robot: The robot model.
        init_q: Initial joint positions, shape (num_joints,).
        target_wxyz: Target orientations as quaternions, shape (target_num, 4).
        target_position: Target positions, shape (target_num, 3).
        target_link_indices: Indices of the joints to be targeted, shape (target_num,).
    """
    JointVar = robot.joint_var_cls

    target_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position)
    batch_axes = target_pose.get_batch_axes()
    robot_batched = jax._src.tree_util.tree_map(lambda x: x[None], robot)
    joint_var0 = JointVar(0).with_value(init_q)

    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot_batched,
            JointVar(jnp.full(batch_axes, 0)),
            target_pose,
            target_link_indices,
            pos_weight=50.0,
            ori_weight=2.5,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=JointVar.default_factory(),
            weight=1.0,
        ),
        pk.costs.limit_cost(
            robot,
            JointVar(0),
            jnp.array([100.0] * robot.joints.num_joints),
        ),
    ]

    sol = (
        jaxls.LeastSquaresProblem(factors, [JointVar(0)])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var0]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]  # (J,)

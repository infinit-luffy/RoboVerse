from __future__ import annotations

from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import torch

from metasim.utils import configclass, math
from metasim.utils.bidex_util import _solve_ik_single_env, jax_to_torch, torch_to_jax
from metasim.utils.state import TensorState

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class FrankaShadowHandLeftCfg(BaseRobotCfg):
    """Cfg for the franka with left shadow hand robot."""

    name: str = "franka_shadow_left"
    num_joints: int = 31
    num_arm_joints: int = 7
    fix_base_link: bool = True
    urdf_path: str = "roboverse_data/robots/franka_shadow_hand/urdf/franka_shadow_left.urdf"
    project_root: Path = Path(__file__).resolve().parents[3]
    hand_urdf_path: Path = (
        project_root / "roboverse_data" / "robots" / "franka_shadow_hand" / "urdf" / "shadow_left.urdf"
    )
    isaacgym_read_mjcf: bool = False
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    use_vhacd: bool = True
    dof_drive_mode: Literal["none", "position", "effort"] = "position"
    angular_damping: float = None
    linear_damping: float = None
    tendon_limit_stiffness: float = None
    tendon_damping: float = None
    friction = None  # Use default friction from MJCF
    robot_controller: Literal["ik", "dof_pos", "dof_effort"] = "ik"
    fingertips = ["ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal"]
    fingertips_offset = [0.0, 0.0, 0.02]
    num_fingertips: int = len(fingertips)
    observation_shape: int = num_joints * 2 + 7 + num_fingertips * 7
    wrist = "forearm"
    palm = "palm"
    shadow_hand_wrist_stiffness: float = 5
    shadow_hand_wrist_damping: float = 0.5
    shadow_hand_finger_stiffness: float = 1.0
    shadow_hand_finger_damping: float = 0.1
    vel_obs_scale: float = 0.2  # Scale for velocity observations
    force_torque_obs_scale: float = 10.0  # Scale for force and torque observations
    actuators: dict[str, BaseActuatorCfg] = {
        "panda_joint1": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint2": BaseActuatorCfg(stiffness=1e4, damping=1e3, velocity_limit=2.175),
        "panda_joint3": BaseActuatorCfg(stiffness=1e5, damping=5e3, velocity_limit=2.175),
        "panda_joint4": BaseActuatorCfg(stiffness=1e5, damping=1e4, velocity_limit=2.175),
        "panda_joint5": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint6": BaseActuatorCfg(stiffness=400, damping=50, velocity_limit=2.61),
        "panda_joint7": BaseActuatorCfg(stiffness=800, damping=50, velocity_limit=2.61),
        "WRJ2": BaseActuatorCfg(stiffness=shadow_hand_wrist_stiffness, damping=shadow_hand_wrist_damping),
        "WRJ1": BaseActuatorCfg(stiffness=shadow_hand_wrist_stiffness, damping=shadow_hand_wrist_damping),
        "FFJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "FFJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "FFJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "FFJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ5": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "LFJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "MFJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "MFJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "MFJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "MFJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "RFJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "RFJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "RFJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "RFJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ5": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ4": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ3": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ2": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
        "THJ1": BaseActuatorCfg(stiffness=shadow_hand_finger_stiffness, damping=shadow_hand_finger_damping),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "panda_joint1": (-2.8973, 2.8973),
        "panda_joint2": (-1.7628, 1.7628),
        "panda_joint3": (-2.8973, 2.8973),
        "panda_joint4": (-3.0718, -0.0698),
        "panda_joint5": (-2.8973, 2.8973),
        "panda_joint6": (-0.0175, 3.7525),
        "panda_joint7": (-2.8973, 2.8973),
        "WRJ2": (-0.489, 0.14),
        "WRJ1": (-0.698, 0.489),
        "FFJ4": (-0.349, 0.349),
        "FFJ3": (0, 1.571),
        "FFJ2": (0, 1.571),
        "FFJ1": (0, 1.571),
        "LFJ5": (0, 0.785),
        "LFJ4": (-0.349, 0.349),
        "LFJ3": (0, 1.571),
        "LFJ2": (0, 1.571),
        "LFJ1": (0, 1.571),
        "MFJ4": (-0.349, 0.349),
        "MFJ3": (0, 1.571),
        "MFJ2": (0, 1.571),
        "MFJ1": (0, 1.571),
        "RFJ4": (-0.349, 0.349),
        "RFJ3": (0, 1.571),
        "RFJ2": (0, 1.571),
        "RFJ1": (0, 1.571),
        "THJ5": (-1.047, 1.047),
        "THJ4": (0, 1.222),
        "THJ3": (-0.209, 0.209),
        "THJ2": (-0.524, 0.524),
        "THJ1": (0, 1.571),
    }

    # set False for visualization correction. Also see https://forums.developer.nvidia.com/t/how-to-flip-collision-meshes-in-isaac-gym/260779 for another example.
    isaacgym_flip_visual_attachments = False

    default_joint_positions: dict[str, float] = {
        "panda_joint1": 0.0,
        "panda_joint2": -0.785398,
        "panda_joint3": 0.0,
        "panda_joint4": -2.356194,
        "panda_joint5": 0.0,
        "panda_joint6": 3.1415926,
        "panda_joint7": -2.356194,
        "WRJ2": 0.0,
        "WRJ1": 0.0,
        "FFJ4": 0.0,
        "FFJ3": 0.0,
        "FFJ2": 0.0,
        "FFJ1": 0.0,
        "MFJ4": 0.0,
        "MFJ3": 0.0,
        "MFJ2": 0.0,
        "MFJ1": 0.0,
        "RFJ4": 0.0,
        "RFJ3": 0.0,
        "RFJ2": 0.0,
        "RFJ1": 0.0,
        "LFJ5": 0.0,
        "LFJ4": 0.0,
        "LFJ3": 0.0,
        "LFJ2": 0.0,
        "LFJ1": 0.0,
        "THJ5": 0.0,
        "THJ4": 0.0,
        "THJ3": 0.0,
        "THJ2": 0.0,
        "THJ1": 0.0,
    }

    def __post_init__(self):
        super().__post_init__()
        self.dof_names: list[str] = list(self.default_joint_positions.keys())
        self.hand_dof_names: list[str] = self.dof_names[self.num_arm_joints :]
        self.sorted_dof_names: list[str] = sorted(self.dof_names)
        self.joint_reindex_inverse = [self.sorted_dof_names.index(name) for name in self.dof_names]
        self.joint_reindex = [self.dof_names.index(name) for name in self.sorted_dof_names]
        self.hand_joint_sorted_idx = [self.sorted_dof_names.index(name) for name in sorted(self.hand_dof_names)]
        self.joint_limits_lower = [self.joint_limits[name][0] for name in self.dof_names]
        self.joint_limits_upper = [self.joint_limits[name][1] for name in self.dof_names]
        self.load_robot_for_ik()

    def update_state(self, envstates: TensorState):
        if not hasattr(self, "ft_index"):
            self.ft_index = [envstates.robots[self.name].body_names.index(name) for name in self.fingertips]
        if not hasattr(self, "wrist_index"):
            self.wrist_index = envstates.robots[self.name].body_names.index(self.wrist)
        if not hasattr(self, "palm_index"):
            self.palm_index = envstates.robots[self.name].body_names.index(self.palm)
        self.ft_states = envstates.robots[self.name].body_state[:, self.ft_index, :]
        if isinstance(self.joint_limits_lower, list):
            self.joint_limits_lower = torch.tensor(self.joint_limits_lower, device=self.ft_states.device)
            self.joint_limits_upper = torch.tensor(self.joint_limits_upper, device=self.ft_states.device)
        if isinstance(self.fingertips_offset, list):
            self.fingertips_offset = (
                torch.tensor(self.fingertips_offset, device=self.ft_states.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.ft_states.shape[0], self.num_fingertips, 1)
            )
        self.wrist_state = envstates.robots[self.name].body_state[:, self.wrist_index, :]
        if self.robot_controller == "ik":
            self.ft_pos = self.ft_states[:, :, :3]
            self.ft_rot = self.ft_states[:, :, 3:7]
            self.wrist_pos = self.wrist_state[:, :3]
            self.wrist_rot = self.wrist_state[:, 3:7]
            self.ft_relative_pos = math.quat_apply(
                math.quat_inv(self.wrist_rot).unsqueeze(1).repeat(1, self.num_fingertips, 1),
                self.ft_pos - self.wrist_pos.unsqueeze(1),
            )
            self.ft_relative_rot = math.quat_mul(
                self.ft_rot,
                math.quat_inv(self.wrist_rot).unsqueeze(1).repeat(1, self.num_fingertips, 1),
            )
        self.palm_state = envstates.robots[self.name].body_state[:, self.palm_index, :]
        self.dof_pos = envstates.robots[self.name].joint_pos
        self.dof_vel = envstates.robots[self.name].joint_vel
        self.dof_force = envstates.robots[self.name].joint_force

    def scale_hand_action(self, actions: torch.Tensor) -> torch.Tensor:
        if self.robot_controller != "dof_pos":
            raise ValueError("robot_controller must be 'dof_pos' to use scale_hand_action")
        if actions.shape[1] != self.num_joints - self.num_arm_joints:
            raise ValueError(
                f"action shape {actions.shape} does not match hand dof {self.num_joints - self.num_arm_joints}"
            )
        return math.unscale_transform(
            actions, self.joint_limits_lower[self.num_arm_joints :], self.joint_limits_upper[self.num_arm_joints :]
        )

    def control_arm_ik(self, dpose, num_envs: int, device: str):
        # Set controller parameters
        # IK params
        damping = 0.05
        # solve damped least squares
        j_eef = self.jacobian[:, self.wrist_index - 1, :, : self.num_arm_joints]
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=device) * (damping**2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
        u += self.dof_pos[:, self.joint_reindex_inverse[: self.num_arm_joints]]
        u = torch.clamp(
            u, self.joint_limits_lower[: self.num_arm_joints], self.joint_limits_upper[: self.num_arm_joints]
        )
        return u

    def load_robot_for_ik(self):
        import pyroki as pk
        import yourdfpy

        robot_urdf_path = self.hand_urdf_path

        def filename_handler(fname: str) -> str:
            base_path = robot_urdf_path.parent
            return yourdfpy.filename_handler_magic(fname, dir=base_path)

        try:
            urdf = yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"File {robot_urdf_path} not found.") from err

        # Create robot.
        self.hand = pk.Robot.from_urdf(urdf)
        self.solver = jax.jit(jax.vmap(_solve_ik_single_env, in_axes=(None, 0, 0, 0, None), out_axes=0))
        self.hand_dof_names_ik = self.hand.joints.names[1:]  # skip the first joint which is world joint
        self.ik_reindex = [self.hand_dof_names.index(name) for name in self.hand_dof_names_ik]
        self.ik_reindex_inverse = [self.hand_dof_names_ik.index(name) for name in self.hand_dof_names]

    def control_hand_ik(self, target_pos, target_rot):
        if self.robot_controller != "ik":
            raise ValueError("robot_controller must be 'ik' to use control_hand_ik")
        init_q = torch_to_jax(self.dof_pos[:, self.joint_reindex_inverse[self.num_arm_joints :]][:, self.ik_reindex])
        target_wxyz = torch_to_jax(
            math.quat_mul(
                self.ft_relative_rot,
                math.quat_from_euler_xyz(target_rot[..., 0], target_rot[..., 1], target_rot[..., 2]),
            )
        )
        target_position = torch_to_jax(target_pos + self.ft_relative_pos)
        if not hasattr(self, "target_link_indices"):
            self.target_link_indices = jnp.array(
                [self.hand.links.names.index(name) for name in self.fingertips], dtype=jnp.int32
            )

        solution = self.solver(
            self.hand,
            init_q,
            target_wxyz,
            target_position,
            self.target_link_indices,
        )
        hand_dof = jax_to_torch(solution)[:, self.ik_reindex_inverse]
        hand_dof = torch.clamp(
            hand_dof, self.joint_limits_lower[self.num_arm_joints :], self.joint_limits_upper[self.num_arm_joints :]
        )
        return hand_dof

    def observation(self):
        """
        Return the proceptive observation of the robot. The observation includes:
            - Arm joint positions and velocities
            - Hand joint positions and velocities
            - Wrist position and orientation
            - Fingertip positions and orientations
        """
        obs = torch.zeros((self.ft_states.shape[0], self.observation_shape), device=self.ft_states.device)
        obs[:, : self.num_joints] = self.dof_pos
        obs[:, self.num_joints : 2 * self.num_joints] = self.dof_vel * self.vel_obs_scale
        obs[:, 2 * self.num_joints : 2 * self.num_joints + 3] = self.wrist_pos
        obs[:, 2 * self.num_joints + 3 : 2 * self.num_joints + 7] = self.wrist_rot
        ft_pos = self.ft_pos + math.quat_apply(self.ft_rot, self.fingertips_offset)  # (num_envs, num_fingertips, 3)
        ft_state = torch.cat([ft_pos, self.ft_rot], dim=-1).view(self.ft_states.shape[0], -1)
        obs[:, 2 * self.num_joints + 7 :] = ft_state
        return obs

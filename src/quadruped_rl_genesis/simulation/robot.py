"""Unitree Go2 URDF loading, joint setup, and scene attachment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import genesis as gs
import torch
from genesis.utils.geom import inv_quat

from quadruped_rl_genesis.simulation.sensors import GO2_LINKS_TO_KEEP, resolve_feet


def _resolve_urdf_path(path_str: str) -> str:
    """Resolve a URDF path from config: absolute, cwd-relative, or repo-root-relative.

    Args:
        path_str (str): URDF path string to resolve.

    Returns:
        str: Resolved URDF path.

    Raises:
        FileNotFoundError: If the URDF path cannot be resolved.
    """
    path = Path(path_str)
    if path.is_file():
        return str(path.resolve())

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.is_file():
        return str(cwd_candidate.resolve())

    repo_root = Path(__file__).resolve().parents[3]
    repo_candidate = repo_root / path
    if repo_candidate.is_file():
        return str(repo_candidate.resolve())

    return path_str


@dataclass(frozen=True)
class Go2RobotSetup:
    """Resolved Go2 robot handles and default control state.

    Attributes:
        robot (Any): Genesis robot entity.
        motor_dof_indices (list[int]): DOF indices for the 12 motor joints.
        motor_joint_names (list[str]): Joint names in control order.
        default_dof_pos (torch.Tensor): Default joint positions for standing.
        base_init_pos (torch.Tensor): Initial base position.
        base_init_quat (torch.Tensor): Initial base orientation quaternion.
        inv_base_init_quat (torch.Tensor): Inverse of initial base quaternion.
    """

    robot: Any
    motor_dof_indices: list[int]
    motor_joint_names: list[str]
    default_dof_pos: torch.Tensor
    base_init_pos: torch.Tensor
    base_init_quat: torch.Tensor
    inv_base_init_quat: torch.Tensor
    foot_link_indices: list[int]
    foot_link_names: list[str]
    kp: float
    kd: float


def add_go2_robot(scene: Any, robot_config: dict[str, Any]) -> Any:
    """Attach the Go2 robot asset to a Genesis scene.

    Args:
        scene (Any): Genesis scene receiving the robot entity.
        robot_config (dict[str, Any]): Robot configuration with URDF path and
            initial base pose.

    Returns:
        Any: Genesis robot entity added to the scene.
    """
    base_init_pos = robot_config["base_init_pos"]
    base_init_quat = robot_config["base_init_quat"]

    return scene.add_entity(
        gs.morphs.URDF(
            file=_resolve_urdf_path(str(robot_config["urdf_path"])),
            pos=base_init_pos,
            quat=base_init_quat,
            links_to_keep=list(GO2_LINKS_TO_KEEP),
        )
    )


def build_go2_setup(
    *,
    robot: Any,
    robot_config: dict[str, Any],
    control_config: dict[str, Any],
    device: torch.device | str,
) -> Go2RobotSetup:
    """Resolve joint indices, default pose, and PD gains for the Go2 robot.

    Args:
        robot (Any): Genesis robot entity.
        robot_config (dict[str, Any]): Robot configuration payload.
        control_config (dict[str, Any]): Control configuration with PD gains.
        device (torch.device | str): Device used to allocate returned tensors.

    Returns:
        Go2RobotSetup: Structured robot handles and default tensors used by the
            navigation task.
    """
    motor_joint_names = list(robot_config["joint_names"])
    motor_dof_indices = [robot.get_joint(name).dof_start for name in motor_joint_names]
    kp = float(control_config["kp"])
    kd = float(control_config["kd"])
    robot.set_dofs_kp([kp] * int(robot_config["num_actions"]), motor_dof_indices)
    robot.set_dofs_kv([kd] * int(robot_config["num_actions"]), motor_dof_indices)
    feet_mounts = resolve_feet(robot)

    default_dof_pos = torch.tensor(
        [robot_config["default_joint_angles"][name] for name in motor_joint_names],
        device=device,
        dtype=gs.tc_float,
    )
    base_init_pos = torch.tensor(
        robot_config["base_init_pos"],
        device=device,
        dtype=gs.tc_float,
    )
    base_init_quat = torch.tensor(
        robot_config["base_init_quat"],
        device=device,
        dtype=gs.tc_float,
    )

    return Go2RobotSetup(
        robot=robot,
        motor_dof_indices=motor_dof_indices,
        motor_joint_names=motor_joint_names,
        default_dof_pos=default_dof_pos,
        base_init_pos=base_init_pos,
        base_init_quat=base_init_quat,
        inv_base_init_quat=inv_quat(base_init_quat),
        foot_link_indices=[int(mount.link_idx_local) for mount in feet_mounts],
        foot_link_names=[mount.link_name for mount in feet_mounts],
        kp=kp,
        kd=kd,
    )

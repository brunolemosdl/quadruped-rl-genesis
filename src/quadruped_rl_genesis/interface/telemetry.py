"""Structured metrics cards and Genesis viewer window geometry helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from genesis.utils.geom import quat_to_xyz

LAST_REWARDS_COUNT = 5

SectionData = list[tuple[str, list[tuple[str, str]]]]
CardData = tuple[SectionData, SectionData]

MinimapState = dict[str, float]


def get_genesis_window_geometry(env: Any) -> tuple[int, int, int, int] | None:
    """Return the Genesis viewer window bounds when the internal APIs expose them.

    Args:
        env (Any): Wrapped or vectorized environment that may expose ``task.scene``.

    Returns:
        tuple[int, int, int, int] | None: ``(x, y, width, height)`` in screen pixels,
            or ``None`` if geometry cannot be read.
    """
    try:
        task = getattr(env, "task", None)
        if task is None:
            return None
        scene = getattr(task, "scene", None)
        if scene is None:
            return None
        vis = getattr(scene, "visualizer", None)
        viewer = getattr(vis, "_viewer", None) if vis is not None else None
        if viewer is None:
            viewer = getattr(scene, "viewer", None)
        if viewer is None:
            return None
        pyrender_viewer = getattr(viewer, "_pyrender_viewer", None)
        if pyrender_viewer is None:
            return None
        x, y = pyrender_viewer.get_location()
        w, h = pyrender_viewer.get_size()
        return (int(x), int(y), int(w), int(h))
    except Exception:
        return None


def build_minimap_state(task: Any) -> MinimapState | None:
    """Build world-frame state for the visualization minimap (terrain, robot, goal).

    Args:
        task (Any): Navigation task with terrain, goal, and base pose buffers.

    Returns:
        MinimapState | None: Scalar map keyed by ``width_m``, ``length_m``, robot and
            goal coordinates, or ``None`` when terrain is invalid or read fails.
    """
    if task is None:
        return None
    try:
        size = tuple(getattr(task, "terrain_config", {}).get("size", (0, 0)))
        if len(size) < 2 or float(size[0]) <= 0 or float(size[1]) <= 0:
            return None
        goal_cfg = getattr(task, "goal_config", {}) or {}
        w_yaw = float(
            quat_to_xyz(task.base_quat[0:1], rpy=True, degrees=False)[0, 2].item()
        )
        return {
            "width_m": float(size[0]),
            "length_m": float(size[1]),
            "margin_m": float(goal_cfg.get("spawn_margin_m", 0.0)),
            "robot_x": float(task.base_pos[0, 0].item()),
            "robot_y": float(task.base_pos[0, 1].item()),
            "robot_yaw_rad": w_yaw,
            "goal_x": float(task.goal_pos[0, 0].item()),
            "goal_y": float(task.goal_pos[0, 1].item()),
        }
    except Exception:
        return None


def build_metrics_card(
    env: Any,
    step_reward: float,
    episode_return: float,
    step_index: int,
    episode_index: int,
    done: bool,
    info: dict[str, Any] | None,
    last_rewards: list[float],
    actions: np.ndarray | None = None,
) -> CardData:
    """Build metrics card data split into robot and RL columns.

    Args:
        env (Any): Environment or unwrapped env with task attribute.
        step_reward (float): Reward for the current step.
        episode_return (float): Cumulative return for the current episode.
        step_index (int): Current step index within the episode.
        episode_index (int): Current episode index (0-based).
        done (bool): Whether the episode has terminated.
        info (dict[str, Any] | None): Info dict from the last step, if any.
        last_rewards (list[float]): Recent step rewards for display.
        actions (np.ndarray | None): Action tensor sent to the robot.

    Returns:
        CardData: Tuple of (robot_sections, rl_sections) for the overlay columns.
    """
    robot_sections: SectionData = []
    rl_sections: SectionData = []
    task = getattr(env, "task", None)

    ep_rows = [
        ("Step", str(step_index)),
        ("Episode", str(episode_index + 1)),
        ("Reward (step)", f"{step_reward:+.3f}"),
        ("Episode return", f"{episode_return:+.2f}"),
    ]
    if last_rewards:
        last_str = " ".join(f"{r:+.2f}" for r in last_rewards[-5:])
        if len(last_str) > 28:
            last_str = last_str[:25] + "..."
        ep_rows.append(("Last 5 rewards", last_str))
    rl_sections.append(("Episode", ep_rows))

    if task is not None:
        try:
            sums = getattr(task, "episode_sums", {})
            if sums:
                rew_rows = [(name, f"{sums[name][0].item():+.2f}") for name in sums]
                rl_sections.append(("Rewards (cumulative)", rew_rows))
        except Exception:
            pass

    if done and info:
        ep = info.get("episode", {})
        reason = str(info.get("termination_reason", "?")).capitalize()
        term_rows = [("Termination", reason)]
        if "metric_success" in ep:
            term_rows.append(("Success", "Yes" if ep["metric_success"] else "No"))
        if "metric_mean_arc_progress_speed" in ep:
            term_rows.append(
                ("Arc progress", f"{ep['metric_mean_arc_progress_speed']:+.2f} m/s")
            )
        if "metric_final_goal_distance" in ep:
            term_rows.append(
                ("Final goal dist", f"{ep['metric_final_goal_distance']:.2f} m")
            )
        if "metric_final_goal_yaw_error_deg" in ep:
            term_rows.append(
                ("Final yaw err", f"{ep['metric_final_goal_yaw_error_deg']:.1f} deg")
            )
        if "r" in ep:
            term_rows.append(("Final return", f"{ep['r']:.2f}"))
        if "l" in ep:
            term_rows.append(("Episode length", f"{ep['l']} steps"))
        rl_sections.append(("Termination", term_rows))

    if actions is not None:
        joint_names = getattr(task, "motor_joint_names", None)
        default_dof = getattr(task, "default_dof_pos", None)
        action_scale = 0.25
        try:
            action_scale = float(
                getattr(task, "control_config", {}).get("action_scale", 0.25)
            )
        except (TypeError, ValueError):
            pass
        act = actions[0] if hasattr(actions, "ndim") and actions.ndim > 1 else actions
        act_np = act.cpu().numpy() if hasattr(act, "cpu") else np.asarray(act)
        act_flat = act_np.flatten()
        n = len(act_flat)
        if n > 0:
            default_flat = None
            if default_dof is not None:
                d = default_dof[0] if default_dof.ndim > 1 else default_dof
                default_flat = (
                    d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
                ).flatten()
            if joint_names is not None and len(joint_names) >= n:
                cmd_rows = []
                for i, (name, v) in enumerate(zip(joint_names[:n], act_flat)):
                    raw_str = f"{float(v):+.3f}"
                    if default_flat is not None and i < len(default_flat):
                        target_rad = float(v) * action_scale + float(default_flat[i])
                        val_str = f"{raw_str} → {target_rad:+.3f} rad"
                    else:
                        val_str = raw_str
                    cmd_rows.append((name.replace("_joint", ""), val_str))
            else:
                cmd_rows = []
                for i, v in enumerate(act_flat):
                    raw_str = f"{float(v):+.3f}"
                    if default_flat is not None and i < len(default_flat):
                        target_rad = float(v) * action_scale + float(default_flat[i])
                        val_str = f"{raw_str} → {target_rad:+.3f} rad"
                    else:
                        val_str = raw_str
                    cmd_rows.append((f"a{i}", val_str))
            robot_sections.append(("Commands (actions)", cmd_rows))

    if task is not None:
        try:
            dist = float(task.goal_distance[0].item())
            planar = float(torch.linalg.norm(task.base_lin_vel_body[0, :2]).item())
            bear_deg = float(
                torch.rad2deg(torch.abs(task.goal_bearing_error[0])).item()
            )
            goal_vec = (task.goal_pos[0, :2] - task.base_pos[0, :2]).detach()
            gnorm = float(torch.linalg.norm(goal_vec).item()) or 1e-6
            goal_dir = goal_vec / gnorm
            proj = float((goal_dir * task.base_lin_vel_body[0, :2]).sum().item())
            init_d = getattr(task, "initial_goal_distance", None)
            init_str = f"{float(init_d[0].item()):.2f} m" if init_d is not None else "?"
            bezier_cfg = getattr(task, "bezier_config", {}) or {}
            radius = bezier_cfg.get("success_radius_m", "?")
            cross_track = float(task.cross_track_error[0].item())
            cmd_vx = float(task.cmd_vel[0, 0].item())
            cmd_yaw = float(task.cmd_vel[0, 2].item())
            goal_rows = [
                ("Goal distance", f"{dist:.2f} m"),
                ("Initial dist", init_str),
                ("Success radius", f"{radius} m"),
                ("Bearing error", f"{bear_deg:.1f} deg"),
                ("Cross track", f"{cross_track:+.2f} m"),
                ("Speed (planar)", f"{planar:.2f} m/s"),
                ("Speed -> goal", f"{proj:+.2f} m/s"),
                ("Cmd vx", f"{cmd_vx:+.2f} m/s"),
                ("Cmd yaw rate", f"{cmd_yaw:+.2f} rad/s"),
            ]
            robot_sections.append(("Goal", goal_rows))
        except Exception:
            pass

        try:
            gyro_norm = float(torch.linalg.norm(task.imu_gyro[0]).item())
            acc_norm = float(torch.linalg.norm(task.imu_acc[0]).item())
            feet = getattr(task, "foot_contact", None)
            if feet is not None:
                c = feet[0].detach().cpu().numpy()
                feet_str = " ".join("1" if c[i] else "0" for i in range(min(4, len(c))))
            else:
                feet_str = "n/a"
            dof_pos = getattr(task, "dof_pos", None)
            dof_vel = getattr(task, "dof_vel", None)
            if dof_pos is not None:
                p = dof_pos[0].detach().cpu().numpy()
                pos_str = f"range [{p.min():.2f}, {p.max():.2f}]"
            else:
                pos_str = "n/a"
            if dof_vel is not None:
                v = dof_vel[0].detach().cpu().numpy()
                vel_str = f"range [{v.min():.2f}, {v.max():.2f}]"
            else:
                vel_str = "n/a"
            sens_rows = [
                ("IMU gyro norm", f"{gyro_norm:.3f}"),
                ("IMU acc norm", f"{acc_norm:.3f}"),
                ("Feet contact", feet_str),
                ("DOF pos", pos_str),
                ("DOF vel", vel_str),
            ]
            robot_sections.append(("Sensors", sens_rows))
        except Exception:
            pass

    return (robot_sections, rl_sections)

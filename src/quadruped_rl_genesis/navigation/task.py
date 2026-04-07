"""Go2 Bezier-navigation task: Genesis scene, observations, rewards, and curriculum."""

from __future__ import annotations

import math
from typing import Any

import genesis as gs
import numpy as np
import torch
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)

from quadruped_rl_genesis.navigation.bezier import BezierPlanner
from quadruped_rl_genesis.navigation.goals import (
    reset_goal_trackers,
    sample_goal_pose,
)
from quadruped_rl_genesis.navigation.rewards import (
    REWARD_FUNCTIONS,
    turn_in_place_mask,
)
from quadruped_rl_genesis.navigation.termination import (
    compute_termination,
    terminal_reward_terms,
)
from quadruped_rl_genesis.services.logger import get_logger
from quadruped_rl_genesis.simulation.robot import add_go2_robot, build_go2_setup
from quadruped_rl_genesis.simulation.sensors import (
    build_stack,
    read_stack,
)
from quadruped_rl_genesis.simulation.terrain import (
    TERRAIN_MODE_ROUGH,
    build_rough_terrain_morph_kwargs,
    estimate_heightmap_slopes_torch,
    generate_random_terrain_heightmap,
    normalize_terrain_mode,
    resolve_terrain_curriculum_spec,
    sample_height_torch,
    validate_rough_terrain_extent,
)
from quadruped_rl_genesis.utils.tensors import (
    as_matrix_tensor,
    wrap_angle,
)

LOGGER = get_logger(__name__)


class Go2NavigationTask:
    """Bezier-commanded Go2 navigation task using a mounted sensor stack.

    Manages the Genesis simulation scene, robot, terrain, goal spawning,
    per-episode Bezier planning, reward computation, and termination logic
    for quadruped trajectory tracking toward a sampled target position.
    """

    def __init__(
        self,
        experiment_config: dict[str, Any],
        num_envs: int,
        show_viewer: bool = False,
        add_camera: bool = False,
        fast_viz: bool = False,
        viewer_help_text: bool = True,
        disable_reward_curriculum: bool = False,
        disable_terrain_curriculum: bool = False,
    ) -> None:
        """Build the full Go2 navigation task and its simulation resources.

        Args:
            experiment_config (dict[str, Any]): Resolved experiment
                configuration.
            num_envs (int): Number of parallel simulated environments.
            show_viewer (bool, optional): Whether to open the Genesis viewer.
            add_camera (bool, optional): Whether to create the optional video
                camera.
            fast_viz (bool, optional): Whether to use lighter simulation and
                rendering settings for visualization.
            viewer_help_text (bool, optional): Whether to show the default
                keyboard instructions in the viewer. Set False to use a
                custom overlay only.
            disable_reward_curriculum (bool, optional): If True, reward
                curriculum stages are ignored and every configured reward term
                uses its full scale (for visualization / eval of trained policies
                where ``global_step`` does not match training).
            disable_terrain_curriculum (bool, optional): If True, terrain resets
                use the configured evaluation/final stage instead of the
                training terrain curriculum.
        """
        self.experiment_config = experiment_config
        self._disable_reward_curriculum = bool(disable_reward_curriculum)
        self._disable_terrain_curriculum = bool(disable_terrain_curriculum)
        self.environment_config = experiment_config["environment"]
        self.training_config = experiment_config["training"]
        self.visualization_config = experiment_config.get("visualization", {})
        self.fast_viz = fast_viz
        self.num_envs = num_envs
        self.device = gs.device

        simulator_config = self.environment_config["simulator"]
        robot_config = self.environment_config["robot"]
        control_config = self.environment_config["control"]
        observation_config = self.environment_config["observations"]
        terrain_config = self.environment_config["terrain"]
        goal_config = self.environment_config["goal"]
        sensors_config = self.environment_config["sensors"]
        reward_config = self.environment_config["rewards"]
        termination_config = self.environment_config["termination"]
        bezier_config = self.environment_config["bezier"]

        self.robot_config = robot_config
        self.control_config = control_config
        self.observation_config = observation_config
        self.terrain_config = terrain_config
        self.goal_config = goal_config
        self.sensors_config = sensors_config
        self.reward_config = reward_config
        self.termination_config = termination_config
        self.bezier_config = bezier_config
        self.success_yaw_rad = math.radians(
            float(bezier_config.get("success_yaw_deg", 10.0))
        )

        self.dt = float(simulator_config["dt"])
        self.simulate_action_latency = bool(control_config["simulate_action_latency"])
        self.max_episode_length = math.ceil(
            float(termination_config["episode_length_s"]) / self.dt
        )
        self.num_actions = int(robot_config["num_actions"])
        self.goal_rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
        self.terrain_curriculum = resolve_terrain_curriculum_spec(
            terrain_config=terrain_config,
            goal_config=goal_config,
        )

        substeps = 1 if fast_viz else int(simulator_config["substeps"])
        max_collision_pairs = int(simulator_config.get("max_collision_pairs", 320))
        viewer_res = (1280, 720) if fast_viz else None

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=substeps),
            viewer_options=gs.options.ViewerOptions(
                res=viewer_res,
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.4, -0.9, 1.8),
                camera_lookat=(0.0, 0.0, 0.35),
                camera_fov=45,
                enable_help_text=viewer_help_text,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
                shadow=not fast_viz,
                plane_reflection=False,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=max_collision_pairs,
            ),
            show_viewer=show_viewer,
        )

        self.camera = None
        self._camera_dynamic_goal_lookat = False
        self._camera_smoothed_pos: torch.Tensor | None = None
        self._camera_smoothed_lookat: torch.Tensor | None = None
        self.goal_ball = None
        self.show_markers = (show_viewer or add_camera) and self.num_envs == 1

        self._build_terrain()

        self.robot = add_go2_robot(self.scene, robot_config)

        self.sensor_stack = build_stack(
            scene=self.scene,
            robot=self.robot,
            config=sensors_config,
            draw_debug=show_viewer,
        )

        if self.show_markers:
            self._build_markers()

        if add_camera:
            camera_config = self._video_camera_config()
            base_pos = torch.tensor(
                robot_config["base_init_pos"],
                device=self.device,
                dtype=gs.tc_float,
            )
            camera_pos = tuple(
                (
                    base_pos
                    + torch.tensor(
                        camera_config["offset"],
                        device=self.device,
                        dtype=gs.tc_float,
                    )
                )
                .cpu()
                .tolist()
            )
            camera_lookat = tuple(
                (
                    base_pos
                    + torch.tensor(
                        camera_config["lookat_offset"],
                        device=self.device,
                        dtype=gs.tc_float,
                    )
                )
                .cpu()
                .tolist()
            )
            self.camera = self.scene.add_camera(
                res=camera_config["res"],
                pos=camera_pos,
                lookat=camera_lookat,
                fov=camera_config["fov"],
                GUI=show_viewer,
            )

        self.scene.build(n_envs=num_envs)

        robot_setup = build_go2_setup(
            robot=self.robot,
            robot_config=robot_config,
            control_config=control_config,
            device=self.device,
        )
        self.motor_dof_indices = robot_setup.motor_dof_indices
        self.motor_joint_names = robot_setup.motor_joint_names
        self.default_dof_pos = robot_setup.default_dof_pos
        self.foot_link_indices = robot_setup.foot_link_indices
        self.foot_link_names = robot_setup.foot_link_names
        self.control_kp = float(robot_setup.kp)
        self.control_kd = float(robot_setup.kd)
        self._init_joint_noise_rad = float(
            robot_config.get("init_joint_noise_rad", 0.0)
        )
        self.base_init_pos = robot_setup.base_init_pos
        self.base_init_quat = robot_setup.base_init_quat
        self.inv_base_init_quat = robot_setup.inv_base_init_quat
        self._log_control_sanity_checks()

        if self.camera is not None:
            camera_config = self._video_camera_config()
            self._camera_dynamic_goal_lookat = bool(
                camera_config["follow_robot"] and camera_config["lookat_goal"]
            )
            if camera_config["follow_robot"] and not self._camera_dynamic_goal_lookat:
                self.camera.follow_entity(
                    self.robot,
                    smoothing=(
                        float(camera_config["follow_smoothing"])
                        if camera_config["follow_smoothing"] is not None
                        else None
                    ),
                    fix_orientation=False,
                )

        self.global_step = 0
        self.planner = BezierPlanner(
            num_envs=self.num_envs,
            config=self.bezier_config,
            device=self.device,
            dtype=gs.tc_float,
        )
        self._init_buffers()
        self._prepare_rewards()

    def _video_camera_config(self) -> dict[str, Any]:
        """Resolve the optional recording camera settings.

        Defaults depend on whether fast visualization is enabled so interactive
        runs can use lighter rendering without extra configuration.

        Returns:
            dict[str, Any]: Camera configuration with resolution, offsets, and
                follow behavior.
        """
        camera_config = self.visualization_config.get("video_camera", {})
        default_res = (1280, 720) if self.fast_viz else (1920, 1080)

        return {
            "res": tuple(camera_config.get("res", default_res)),
            "offset": tuple(camera_config.get("offset", (-1.6, -0.45, 0.45))),
            "lock_horizon": bool(camera_config.get("lock_horizon", True)),
            "orbit_around_goal_direction": bool(
                camera_config.get("orbit_around_goal_direction", True)
            ),
            "orbit_back_distance_m": float(
                camera_config.get("orbit_back_distance_m", 1.6)
            ),
            "orbit_side_offset_m": float(
                camera_config.get("orbit_side_offset_m", -0.45)
            ),
            "orbit_height_m": float(camera_config.get("orbit_height_m", 0.45)),
            "lookat_offset": tuple(camera_config.get("lookat_offset", (0.2, 0.0, 0.0))),
            "lookat_goal": bool(camera_config.get("lookat_goal", True)),
            "lookat_distance_m": float(camera_config.get("lookat_distance_m", 1.2)),
            "lookat_height_offset_m": float(
                camera_config.get("lookat_height_offset_m", 0.2)
            ),
            "fov": float(camera_config.get("fov", 55)),
            "follow_robot": bool(camera_config.get("follow_robot", True)),
            "follow_smoothing": camera_config.get("follow_smoothing", 0.92),
        }

    def _compute_goal_direction_xy(self) -> torch.Tensor:
        """Compute a robust normalized direction from robot to current goal.

        When the goal direction is ill-conditioned (very close to zero), the
        fallback uses the robot yaw so the camera remains stable.
        """
        base = self.base_pos[0]
        goal_delta_xy = self.goal_pos[0, :2] - base[:2]
        delta_norm = torch.linalg.vector_norm(goal_delta_xy)

        if float(delta_norm.item()) > 1e-6:
            direction_xy = goal_delta_xy / delta_norm
        else:
            yaw = self.base_euler_rad[0, 2]
            direction_xy = torch.stack((torch.cos(yaw), torch.sin(yaw)))

        return direction_xy

    def _compute_goal_facing_lookat(
        self, camera_config: dict[str, Any], direction_xy: torch.Tensor
    ) -> torch.Tensor:
        """Compute a look-at point along the robot-to-goal horizontal direction.

        Args:
            camera_config (dict[str, Any]): Video camera block with look-at distances.
            direction_xy (torch.Tensor): Unit direction in the world XY plane.

        Returns:
            torch.Tensor: World-frame look-at position ``(3,)`` on the robot device.
        """
        base = self.base_pos[0]

        lookat = base.clone()
        lookat_distance = float(camera_config["lookat_distance_m"])
        lookat[0] = base[0] + direction_xy[0] * lookat_distance
        lookat[1] = base[1] + direction_xy[1] * lookat_distance
        lookat[2] = base[2] + float(camera_config["lookat_height_offset_m"])

        return lookat

    def _compute_orbital_camera_position(
        self, camera_config: dict[str, Any], direction_xy: torch.Tensor
    ) -> torch.Tensor:
        """Compute an orbital camera position behind and beside the robot.

        Args:
            camera_config (dict[str, Any]): Orbit offsets and height from video config.
            direction_xy (torch.Tensor): Unit vector from robot toward the goal.

        Returns:
            torch.Tensor: World-frame camera position ``(3,)``.
        """
        base = self.base_pos[0]
        perpendicular_xy = torch.stack((-direction_xy[1], direction_xy[0]))
        camera_pos = base.clone()
        camera_pos[0] = (
            base[0]
            - direction_xy[0] * float(camera_config["orbit_back_distance_m"])
            + perpendicular_xy[0] * float(camera_config["orbit_side_offset_m"])
        )
        camera_pos[1] = (
            base[1]
            - direction_xy[1] * float(camera_config["orbit_back_distance_m"])
            + perpendicular_xy[1] * float(camera_config["orbit_side_offset_m"])
        )
        camera_pos[2] = base[2] + float(camera_config["orbit_height_m"])

        return camera_pos

    def _update_video_camera_pose(self) -> None:
        """Update the Genesis video camera pose for single-env visualization.

        No-op when ``camera`` is ``None`` or ``num_envs != 1``. Applies optional
        exponential smoothing when ``follow_smoothing`` is set.
        """
        if self.camera is None or self.num_envs != 1:
            return

        camera_config = self._video_camera_config()
        if not bool(camera_config["follow_robot"]):
            return

        base = self.base_pos[0]
        direction_xy = self._compute_goal_direction_xy()
        use_orbital_pose = bool(
            camera_config["lookat_goal"]
            and camera_config["orbit_around_goal_direction"]
        )
        if use_orbital_pose:
            target_pos = self._compute_orbital_camera_position(
                camera_config, direction_xy
            )
        else:
            offset = torch.tensor(
                camera_config["offset"], device=self.device, dtype=gs.tc_float
            )
            target_pos = base + offset

        if bool(camera_config["lookat_goal"]):
            target_lookat = self._compute_goal_facing_lookat(
                camera_config, direction_xy
            )
        else:
            target_lookat = base + torch.tensor(
                camera_config["lookat_offset"], device=self.device, dtype=gs.tc_float
            )

        smoothing = camera_config["follow_smoothing"]
        if smoothing is not None:
            alpha = float(smoothing)
            if self._camera_smoothed_pos is None:
                self._camera_smoothed_pos = target_pos.clone()
            else:
                self._camera_smoothed_pos = (
                    alpha * self._camera_smoothed_pos + (1.0 - alpha) * target_pos
                )
            camera_pos = self._camera_smoothed_pos
            camera_lookat = target_lookat
        else:
            camera_pos = target_pos
            camera_lookat = target_lookat
            self._camera_smoothed_pos = camera_pos.clone()
        self._camera_smoothed_lookat = camera_lookat.clone()

        set_pose_kwargs = {
            "pos": camera_pos.detach().cpu().tolist(),
            "lookat": camera_lookat.detach().cpu().tolist(),
        }
        if bool(camera_config["lock_horizon"]):
            set_pose_kwargs["up"] = [0.0, 0.0, 1.0]
        self.camera.set_pose(**set_pose_kwargs)

    def _log_control_sanity_checks(self) -> None:
        """Warn when configured control targets disagree with robot joint limits.

        The collected summary is stored on ``self.control_sanity`` for callers
        and diagnostics.
        """
        joint_names = list(self.motor_joint_names)
        lower_limits, upper_limits = self.robot.get_dofs_limit(self.motor_dof_indices)
        qpos0 = self.robot.get_dofs_position(self.motor_dof_indices)

        def _as_dof_vector(values: torch.Tensor) -> torch.Tensor:
            """Normalize joint tensors into a one-dimensional DOF vector.

            Args:
                values (torch.Tensor): Tensor returned by Genesis.

            Returns:
                torch.Tensor: One-dimensional tensor aligned with the motor DOF
                    order.
            """
            if values.ndim == 0:
                return values.reshape(1)
            if values.ndim == 1:
                return values

            return values[0]

        lower_limits = _as_dof_vector(lower_limits)
        upper_limits = _as_dof_vector(upper_limits)
        qpos0 = _as_dof_vector(qpos0)

        action_span = float(self.control_config["clip_actions"]) * float(
            self.control_config["action_scale"]
        )
        commanded_lower = self.default_dof_pos - action_span
        commanded_upper = self.default_dof_pos + action_span

        def _collect(
            mask: torch.Tensor, value_a: torch.Tensor, value_b: torch.Tensor
        ) -> list[str]:
            """Format joint-limit violations for warning logs.

            Args:
                mask (torch.Tensor): Boolean mask selecting problematic joints.
                value_a (torch.Tensor): Observed or commanded values.
                value_b (torch.Tensor): Reference limits associated with each
                    joint.

            Returns:
                list[str]: Readable per-joint diagnostic strings.
            """
            entries: list[str] = []
            for index, violated in enumerate(mask.tolist()):
                if not violated:
                    continue
                entries.append(
                    f"{joint_names[index]}="
                    f"{float(value_a[index].item()):.3f}"
                    f" limit={float(value_b[index].item()):.3f}"
                )
            return entries

        qpos0_outside = (qpos0 < lower_limits) | (qpos0 > upper_limits)
        default_outside = (self.default_dof_pos < lower_limits) | (
            self.default_dof_pos > upper_limits
        )
        command_range_outside = (commanded_lower < lower_limits) | (
            commanded_upper > upper_limits
        )
        qpos0_delta = torch.abs(qpos0 - self.default_dof_pos)
        qpos0_mismatch = qpos0_delta > 0.25

        if torch.any(qpos0_outside):
            LOGGER.debug(
                "URDF qpos0 exceeds joint limits | %s",
                ", ".join(
                    _collect(
                        qpos0_outside,
                        qpos0,
                        torch.where(qpos0 > upper_limits, upper_limits, lower_limits),
                    )
                ),
            )

        if torch.any(default_outside):
            LOGGER.debug(
                "Configured default joint pose exceeds limits | %s",
                ", ".join(
                    _collect(
                        default_outside,
                        self.default_dof_pos,
                        torch.where(
                            self.default_dof_pos > upper_limits,
                            upper_limits,
                            lower_limits,
                        ),
                    )
                ),
            )

        if torch.any(command_range_outside):
            entries = []
            for index, violated in enumerate(command_range_outside.tolist()):
                if not violated:
                    continue
                entries.append(
                    f"{joint_names[index]}="
                    f"[{float(commanded_lower[index].item()):.3f}, "
                    f"{float(commanded_upper[index].item()):.3f}] "
                    f"limits=[{float(lower_limits[index].item()):.3f}, "
                    f"{float(upper_limits[index].item()):.3f}]"
                )
            LOGGER.debug(
                "Applied action range exceeds joint limits | action_scale=%.3f clip=%.3f | %s",
                float(self.control_config["action_scale"]),
                float(self.control_config["clip_actions"]),
                ", ".join(entries),
            )

        if torch.any(qpos0_mismatch):
            entries = []
            for index, mismatched in enumerate(qpos0_mismatch.tolist()):
                if not mismatched:
                    continue
                entries.append(
                    f"{joint_names[index]}="
                    f"qpos0={float(qpos0[index].item()):.3f} "
                    f"default={float(self.default_dof_pos[index].item()):.3f}"
                )
            LOGGER.debug(
                "URDF qpos0 differs materially from configured default pose | %s",
                ", ".join(entries),
            )

        self.control_sanity = {
            "qpos0_outside_limits": bool(torch.any(qpos0_outside).item()),
            "default_pose_outside_limits": bool(torch.any(default_outside).item()),
            "action_range_outside_limits": bool(
                torch.any(command_range_outside).item()
            ),
            "qpos0_default_mismatch": bool(torch.any(qpos0_mismatch).item()),
            "action_scale": float(self.control_config["action_scale"]),
            "clip_actions": float(self.control_config["clip_actions"]),
        }

    def _build_terrain(self) -> None:
        """Create the terrain entity and cache heightmap sampling metadata.

        With ``terrain.enabled`` and ``terrain.mode: irregular``, builds a procedural
        heightmap (``generator``, optional terraces and terrain curriculum). With
        ``terrain.mode: rough``, builds Genesis ``Terrain`` from ``terrain.rough``
        (subterrain grid). With ``terrain.enabled`` false, uses a plane. Optional
        ``textures/checker.png`` supplies a tiled diffuse; ``uv_scale`` / plane
        ``tile_size`` control UV repetition.
        """
        terrain_config = self.terrain_config
        use_terrain = bool(terrain_config["enabled"])
        terrain_mode = normalize_terrain_mode(terrain_config.get("mode"))
        terrain_size = tuple(terrain_config["size"])
        terrain_uv_scale = float(terrain_config["uv_scale"])

        checker_texture = None
        try:
            checker_texture = gs.textures.ImageTexture(
                image_path="textures/checker.png"
            )
        except Exception:
            checker_texture = None
        ground_surface = (
            gs.surfaces.Rough(diffuse_texture=checker_texture)
            if checker_texture is not None
            else None
        )

        if use_terrain:
            width, length = terrain_size
            terrain_pos = (-width / 2, -length / 2, 0.0)

            if terrain_mode == TERRAIN_MODE_ROUGH:
                validate_rough_terrain_extent(terrain_config)
                morph_kw = build_rough_terrain_morph_kwargs(
                    terrain_config,
                    terrain_pos=terrain_pos,
                    default_uv_scale=terrain_uv_scale,
                )
                terrain_morph = gs.morphs.Terrain(**morph_kw)
                if ground_surface is None:
                    terrain_entity = self.scene.add_entity(morph=terrain_morph)
                else:
                    terrain_entity = self.scene.add_entity(
                        morph=terrain_morph, surface=ground_surface
                    )
                heightmap = np.asarray(terrain_entity.terrain_hf, dtype=np.float32)
                wr, lr = heightmap.shape
                center_height = float(heightmap[wr // 2, lr // 2])
                heightmap = (heightmap - center_height).astype(np.float32)
                hr = tuple(
                    float(x) for x in terrain_config.get("height_range", (-2.0, 2.0))
                )
                if len(hr) == 2:
                    heightmap = np.clip(heightmap, hr[0], hr[1]).astype(np.float32)
                horizontal_scale = float(terrain_entity.terrain_scale[0])
                self.terrain_heightmap = heightmap
                self.terrain_heightmap_tensor = torch.from_numpy(heightmap).to(
                    device=self.device,
                    dtype=gs.tc_float,
                )
                self.terrain_size = terrain_size
                self.terrain_resolution = (heightmap.shape[0], heightmap.shape[1])
                self.terrain_pos = terrain_pos
                self.terrain_horizontal_scale = horizontal_scale
            else:
                base_resolution = tuple(terrain_config["resolution"])
                terrain_resolution = (
                    (max(25, base_resolution[0] // 2), max(25, base_resolution[1] // 2))
                    if self.fast_viz
                    else base_resolution
                )
                terrain_height_range = tuple(terrain_config["height_range"])
                terrain_num_functions = int(terrain_config["num_functions"])
                flat_radius = float(terrain_config.get("flat_radius", 5.0))
                terrain_generator_config = terrain_config.get("generator")

                terrain_curriculum = self.terrain_curriculum
                if getattr(
                    self, "_disable_terrain_curriculum", False
                ) and terrain_curriculum.get("enabled", False):
                    evaluation_stage = self._terrain_stage_spec(
                        int(
                            terrain_curriculum.get(
                                "evaluation_stage_index",
                                len(terrain_curriculum.get("stages", [])) - 1,
                            )
                        )
                    )
                    terrain_curriculum = {
                        "enabled": True,
                        "progression": "steps",
                        "stage_steps": [0],
                        "evaluation_stage_index": 0,
                        "stages": [evaluation_stage],
                    }
                heightmap = generate_random_terrain_heightmap(
                    size=terrain_size,
                    resolution=terrain_resolution,
                    height_range=terrain_height_range,
                    flat_radius=flat_radius,
                    num_functions=terrain_num_functions,
                    seed=terrain_config.get("seed"),
                    generator_config=terrain_generator_config,
                    terrain_curriculum=terrain_curriculum,
                )

                width_res, _ = terrain_resolution
                horizontal_scale = width / (width_res - 1) if width_res > 1 else width

                terrain_morph = gs.morphs.Terrain(
                    height_field=heightmap,
                    horizontal_scale=horizontal_scale,
                    vertical_scale=1.0,
                    pos=terrain_pos,
                    uv_scale=terrain_uv_scale,
                )
                if ground_surface is None:
                    self.scene.add_entity(morph=terrain_morph)
                else:
                    self.scene.add_entity(morph=terrain_morph, surface=ground_surface)

                self.terrain_heightmap = heightmap
                self.terrain_heightmap_tensor = torch.from_numpy(heightmap).to(
                    device=self.device,
                    dtype=gs.tc_float,
                )
                self.terrain_size = terrain_size
                self.terrain_resolution = terrain_resolution
                self.terrain_pos = terrain_pos
                self.terrain_horizontal_scale = horizontal_scale
        else:
            width, length = terrain_size
            tile_size = max(1, int(terrain_size[0] / terrain_uv_scale))
            plane_morph = gs.morphs.Plane(
                pos=(0.0, 0.0, 0.0),
                plane_size=(width, length),
                tile_size=(tile_size, tile_size),
                fixed=True,
            )
            if ground_surface is None:
                self.scene.add_entity(morph=plane_morph)
            else:
                self.scene.add_entity(morph=plane_morph, surface=ground_surface)

            self.terrain_heightmap = None
            self.terrain_heightmap_tensor = None
            self.terrain_size = terrain_size
            self.terrain_resolution = None
            self.terrain_pos = (-width / 2, -length / 2, 0.0)
            self.terrain_horizontal_scale = None

    def _build_markers(self) -> None:
        """Create the goal sphere marker used during visualization.

        Markers are only meaningful when one environment is being visualized.
        """
        marker_radius = float(self.goal_config.get("marker_radius_m", 0.12))

        try:
            ball_surface = gs.surfaces.Smooth(color=(0.95, 0.15, 0.15, 1.0))
        except Exception:
            ball_surface = None

        ball_morph = gs.morphs.Sphere(
            radius=marker_radius,
            pos=(0.0, 0.0, 0.0),
            collision=False,
            fixed=True,
        )
        if ball_surface is None:
            self.goal_ball = self.scene.add_entity(ball_morph)
        else:
            self.goal_ball = self.scene.add_entity(ball_morph, surface=ball_surface)

    def _init_buffers(self) -> None:
        """Allocate the persistent tensors used throughout the task lifecycle.

        Buffers are organized by robot state, goal state, sensor state, reward
        bookkeeping, and observation assembly.
        """
        scales = self.observation_config["scales"]

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dof_acc = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.spawn_pos = torch.zeros_like(self.base_pos)
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.base_euler_deg = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_euler_rad = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_lin_vel_world = torch.zeros_like(self.base_lin_vel)
        self.base_lin_vel_body = torch.zeros_like(self.base_lin_vel)
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.gravity_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.gravity_vec[:, 2] = -1.0

        self.terrain_height_at_robot = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.goal_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.goal_distance = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.previous_goal_distance = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.initial_goal_distance = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.goal_bearing_error = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.previous_goal_bearing_error = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.initial_goal_bearing_error_abs = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.cmd_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.arc_s = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.arc_progress = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.remaining_distance = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.cross_track_error = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.lookahead_tangent_xy = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=gs.tc_float
        )
        self.heading_error_to_tangent = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.goal_yaw_target = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.goal_yaw_error = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.curve_length = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.lookahead_curvature = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.deviation_mask = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.within_approach_zone = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.within_success_zone = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.nominal_base_height = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.local_terrain_slope_rad = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.local_terrain_slope_deg = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.local_forward_slope_rad = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.local_forward_slope_deg = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.local_terrain_gradient = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=gs.tc_float
        )
        self.local_terrain_normal = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.local_terrain_normal[:, 2] = 1.0
        self.climbable_slope_factor = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.curve_progress_gate = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.curve_progress_heading_gate = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.curve_progress_corridor_gate = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.terrain_stage_index = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.terrain_step_height_m = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.terrain_terrace_width_m = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.terrain_edge_smoothing = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.terrain_global_height_scale = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.terrain_local_irregularity_m = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.terrain_roughness_residual_m = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )

        self.imu_gyro = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.imu_acc = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.imu_mag = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.foot_force = torch.zeros(
            (self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float
        )
        self.foot_positions = torch.zeros(
            (self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float
        )
        self.foot_positions_body = torch.zeros(
            (self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float
        )
        self.foot_velocities = torch.zeros(
            (self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float
        )
        self.foot_kinematics_available = False
        self._prev_foot_positions_world = torch.zeros(
            (self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float
        )
        self.foot_force_norm = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.foot_contact = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.bool
        )
        self.last_foot_contact = torch.zeros_like(self.foot_contact)
        self.foot_airtime = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.foot_airtime_term = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.impact_term = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.joint_torques = torch.zeros_like(self.actions)
        self.target_dof_pos = torch.zeros_like(self.actions)

        imu_config = self.sensors_config["imu"]
        feet_config = self.sensors_config["feet"]
        self._imu_enabled = bool(imu_config.get("enabled", True))
        self._feet_enabled = bool(feet_config.get("enabled", True))

        self.success_hold_counter = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.stagnation_counter = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.bad_posture_counter = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.critical_posture_counter = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )
        self.episode_reward_sum = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )

        self.episode_metric_sums: dict[str, torch.Tensor] = {
            "planar_speed": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "vertical_speed_abs": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "mean_cross_track_error": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "mean_arc_progress_speed": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "mean_curve_progress_gate": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "mean_local_slope_deg": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "mean_forward_slope_deg": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "reverse_motion_ratio": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "lateral_motion_ratio": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "airborne_ratio": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "trot_contact_score": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "turn_in_place_steps": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "foot_airtime": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
            "impact": torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            ),
        }

        self.obs_parts = {
            "projected_gravity": 3,
            "imu_gyro": 3,
            "cmd_vel": 3,
            "cross_track_error": 1,
            "heading_error": 2,
            "remaining_distance": 1,
            "goal_heading": 2,
            "dof_pos": self.num_actions,
            "dof_vel": self.num_actions,
            "foot_contact": 4,
            "actions": self.num_actions,
        }
        self.num_observations = sum(self.obs_parts.values())
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observations),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras: dict[str, Any] = {"observations": {}}

        self.obs_scales = {
            "projected_gravity": float(scales.get("projected_gravity", 1.0)),
            "imu_gyro": float(scales["imu_gyro"]),
            "cmd_vel": float(scales.get("cmd_vel", 1.0)),
            "cross_track_error": float(scales.get("cross_track_error", 1.0)),
            "heading_error": float(scales.get("heading_error", 1.0)),
            "remaining_distance": float(scales.get("remaining_distance", 0.25)),
            "goal_heading": float(scales.get("goal_heading", 1.0)),
            "dof_pos": float(scales["dof_pos"]),
            "dof_vel": float(scales["dof_vel"]),
        }

    def _prepare_rewards(self) -> None:
        """Wire YAML reward names to callables, apply dt scaling, curriculum, caps.

        Dense terms are multiplied by ``dt`` so config scales match reward per
        second, while terminal bonuses and penalties are applied without
        ``dt``. Unknown names in config are skipped and the reward registry is
        treated as the source of truth.
        """
        dense_scale_config = {
            name: float(value)
            for name, value in self.reward_config["dense_scales"].items()
        }
        terminal_scale_config = {
            name: float(value)
            for name, value in self.reward_config["terminal_scales"].items()
        }

        self.reward_scales = {
            name: float(scale) * self.dt
            for name, scale in dense_scale_config.items()
            if name in REWARD_FUNCTIONS
        }
        self.terminal_reward_scales = dict(terminal_scale_config)
        self.reward_functions = {
            name: REWARD_FUNCTIONS[name] for name in self.reward_scales
        }
        self.episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            for name in (
                *self.reward_scales.keys(),
                *self.terminal_reward_scales.keys(),
            )
        }

        curriculum = self.reward_config.get("curriculum", {})
        self._curriculum_enabled = bool(curriculum.get("enabled", False))
        self._curriculum_stage_steps = list(
            curriculum.get("stage_steps", [0, 3_000_000, 8_000_000])
        )
        self._term_to_min_stage: dict[str, int] = {}
        if self._curriculum_enabled:
            stage_keys = sorted(
                key
                for key in curriculum
                if key.startswith("stage_") and key != "stage_steps"
            )
            for stage_index, stage_key in enumerate(stage_keys):
                for term in curriculum.get(stage_key, {}):
                    if term not in self._term_to_min_stage:
                        self._term_to_min_stage[term] = stage_index

        self._reward_caps: dict[str, float] = {
            key: float(value)
            for key, value in self.reward_config.get("caps", {}).items()
        }

    def _refresh_markers(self) -> None:
        """Move visualization markers to the current goal position.

        This method exits immediately when markers are disabled.
        """
        if not self.show_markers or self.goal_ball is None:
            return

        goal_pos = self.goal_pos[[0]].clone()
        self.goal_ball.set_pos(goal_pos, envs_idx=torch.tensor([0], device=self.device))

    def set_goal_position_xy(
        self,
        x: float,
        y: float,
        *,
        env_idx: int = 0,
        reset_trackers: bool = True,
    ) -> bool:
        """Set goal XY position for one environment during runtime.

        Args:
            x (float): Target X coordinate in world frame.
            y (float): Target Y coordinate in world frame.
            env_idx (int, optional): Environment index to update.
            reset_trackers (bool, optional): Whether to reset distance/bearing
                trackers so rewards reflect the new target as a fresh objective.

        Returns:
            bool: ``True`` when the goal was updated, ``False`` for invalid index.
        """
        if env_idx < 0 or env_idx >= self.num_envs:
            return False

        width, length = tuple(self.terrain_config["size"])
        margin = float(self.goal_config.get("spawn_margin_m", 0.0))
        min_x = -0.5 * float(width) + margin
        max_x = 0.5 * float(width) - margin
        min_y = -0.5 * float(length) + margin
        max_y = 0.5 * float(length) - margin

        clamped_x = min(max(float(x), min_x), max_x)
        clamped_y = min(max(float(y), min_y), max_y)

        goal_xy = torch.tensor(
            [[clamped_x, clamped_y]],
            device=self.device,
            dtype=gs.tc_float,
        )
        goal_z = sample_height_torch(
            self.terrain_heightmap_tensor,
            goal_xy,
            size=tuple(self.terrain_config["size"]),
        )[0]
        self.goal_pos[env_idx, 0] = goal_xy[0, 0]
        self.goal_pos[env_idx, 1] = goal_xy[0, 1]
        self.goal_pos[env_idx, 2] = goal_z

        if reset_trackers:
            goal_delta = self.goal_pos[env_idx, :2] - self.base_pos[env_idx, :2]
            current_distance = torch.linalg.vector_norm(goal_delta, dim=0)
            self.initial_goal_distance[env_idx] = current_distance
            self.previous_goal_distance[env_idx] = current_distance

            current_yaw = quat_to_xyz(
                self.base_quat[[env_idx]], rpy=True, degrees=False
            )[0, 2]
            goal_bearing = torch.atan2(goal_delta[1], goal_delta[0])
            bearing_error = wrap_angle(goal_bearing - current_yaw)
            self.goal_bearing_error[env_idx] = bearing_error
            self.previous_goal_bearing_error[env_idx] = bearing_error
            self.initial_goal_bearing_error_abs[env_idx] = torch.abs(bearing_error)

        env_ids = torch.tensor([env_idx], device=self.device, dtype=torch.long)
        self._reset_navigation(env_ids)
        self._update_navigation_state()
        self._update_local_terrain_features()
        self._refresh_markers()
        return True

    def nudge_goal_position_xy(
        self,
        dx: float,
        dy: float,
        *,
        env_idx: int = 0,
        reset_trackers: bool = True,
    ) -> bool:
        """Move goal XY position by a delta for one environment.

        Args:
            dx (float): Delta on X in meters.
            dy (float): Delta on Y in meters.
            env_idx (int, optional): Environment index to update.
            reset_trackers (bool, optional): Whether to reset distance/bearing
                trackers after movement.

        Returns:
            bool: ``True`` when movement was applied.
        """
        if env_idx < 0 or env_idx >= self.num_envs:
            return False

        current_x = float(self.goal_pos[env_idx, 0].item())
        current_y = float(self.goal_pos[env_idx, 1].item())
        return self.set_goal_position_xy(
            x=current_x + float(dx),
            y=current_y + float(dy),
            env_idx=env_idx,
            reset_trackers=reset_trackers,
        )

    def _as_matrix(self, values: torch.Tensor, *, width: int) -> torch.Tensor:
        """Normalize a Tensor into ``[num_envs, width]``.

        Args:
            values (torch.Tensor): Raw tensor returned by Genesis.
            width (int): Expected second dimension width.

        Returns:
            torch.Tensor: Tensor shaped as ``[num_envs, width]``.
        """
        return as_matrix_tensor(values, num_envs=self.num_envs, width=width)

    def _safe_get_joint_torques(self) -> torch.Tensor:
        """Read measured joint torques/forces, falling back to PD torques.

        Returns:
            torch.Tensor: Shape ``[num_envs, num_actions]`` motor torques.
        """
        for getter_name in ("get_dofs_force", "get_dofs_torque"):
            getter = getattr(self.robot, getter_name, None)
            if getter is None:
                continue
            try:
                measured = getter(self.motor_dof_indices)
                measured = self._as_matrix(measured, width=self.num_actions)
                return measured.to(device=self.device, dtype=gs.tc_float)
            except Exception:
                continue

        return (
            self.control_kp * (self.target_dof_pos - self.dof_pos)
            - self.control_kd * self.dof_vel
        )

    def _safe_get_foot_kinematics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read foot link positions and velocities with graceful degradation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: World-frame positions and velocities,
                each ``[num_envs, 4, 3]``. Sets ``foot_kinematics_available`` when
                link APIs succeed.
        """
        foot_pos = None
        foot_vel = None
        self.foot_kinematics_available = False

        get_links_pos = getattr(self.robot, "get_links_pos", None)
        if get_links_pos is not None:
            try:
                raw_pos = get_links_pos(self.foot_link_indices)
                raw_pos = raw_pos.to(device=self.device, dtype=gs.tc_float)
                if raw_pos.ndim == 3:
                    if raw_pos.shape[0] == self.num_envs:
                        foot_pos = raw_pos
                    elif raw_pos.shape[1] == self.num_envs:
                        foot_pos = raw_pos.transpose(0, 1)
                elif raw_pos.ndim == 2 and raw_pos.shape[0] == len(
                    self.foot_link_indices
                ):
                    foot_pos = raw_pos.unsqueeze(0).repeat(self.num_envs, 1, 1)
                self.foot_kinematics_available = foot_pos is not None
            except Exception:
                foot_pos = None

        get_links_vel = getattr(self.robot, "get_links_vel", None)
        if get_links_vel is not None:
            try:
                raw_vel = get_links_vel(self.foot_link_indices)
                raw_vel = raw_vel.to(device=self.device, dtype=gs.tc_float)
                if raw_vel.ndim == 3:
                    if raw_vel.shape[0] == self.num_envs:
                        foot_vel = raw_vel
                    elif raw_vel.shape[1] == self.num_envs:
                        foot_vel = raw_vel.transpose(0, 1)
                elif raw_vel.ndim == 2 and raw_vel.shape[0] == len(
                    self.foot_link_indices
                ):
                    foot_vel = raw_vel.unsqueeze(0).repeat(self.num_envs, 1, 1)
            except Exception:
                foot_vel = None

        if foot_pos is None:
            foot_pos = torch.zeros(
                (self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float
            )
        if foot_vel is None:
            foot_vel = (foot_pos - self._prev_foot_positions_world) / max(self.dt, 1e-6)

        return foot_pos, foot_vel

    def _current_world_yaw(self) -> torch.Tensor:
        """Return robot yaw in the world frame for every environment.

        Returns:
            torch.Tensor: Yaw radians, shape ``[num_envs]``.
        """
        return quat_to_xyz(self.base_quat, rpy=True, degrees=False)[:, 2]

    def _robot_forward_xy(self) -> torch.Tensor:
        """Return the body x-axis projected and normalized in the world XY plane.

        Returns:
            torch.Tensor: Unit forward vectors, shape ``[num_envs, 2]``.
        """
        basis_x = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        basis_x[:, 0] = 1.0
        forward_world = transform_by_quat(basis_x, self.base_quat)
        forward_xy = forward_world[:, :2]
        forward_norm = torch.linalg.vector_norm(forward_xy, dim=1, keepdim=True).clamp(
            min=1.0e-6
        )

        return forward_xy / forward_norm

    def _current_terrain_curriculum_stage_index(self) -> int:
        """Return the active terrain curriculum stage from ``global_step``.

        Returns:
            int: Clamped stage index into ``terrain_curriculum.stages``.
        """
        stages = list(self.terrain_curriculum.get("stages", []))
        if not stages:
            return 0
        if getattr(self, "_disable_terrain_curriculum", False):
            return int(
                self.terrain_curriculum.get(
                    "evaluation_stage_index",
                    len(stages) - 1,
                )
            )

        current_stage_index = 0
        for stage_index, threshold in enumerate(
            self.terrain_curriculum.get("stage_steps", [0] * len(stages))
        ):
            if self.global_step >= int(threshold):
                current_stage_index = stage_index

        return int(np.clip(current_stage_index, 0, len(stages) - 1))

    def _terrain_stage_spec(self, stage_index: int | None = None) -> dict[str, Any]:
        """Return the resolved terrain-stage dictionary for one stage.

        Args:
            stage_index (int | None): Explicit stage index, or ``None`` to use the
                active curriculum stage.

        Returns:
            dict[str, Any]: Stage parameters including ``spawn_xy`` and height scales.
        """
        stages = list(self.terrain_curriculum.get("stages", []))
        if not stages:
            return {
                "index": 0,
                "name": "stage_1",
                "step_height_m": 0.0,
                "terrace_width_m": 0.0,
                "edge_smoothing": 0.0,
                "global_height_scale": 1.0,
                "local_irregularity_m": 0.0,
                "roughness_residual_m": 0.0,
                "spawn_xy": (0.0, 0.0),
            }

        resolved_index = (
            self._current_terrain_curriculum_stage_index()
            if stage_index is None
            else int(stage_index)
        )
        resolved_index = int(np.clip(resolved_index, 0, len(stages) - 1))
        return dict(stages[resolved_index])

    def _apply_terrain_stage(self, env_ids: torch.Tensor) -> None:
        """Write the active terrain curriculum parameters into per-env tensors.

        Args:
            env_ids (torch.Tensor): Environment indices to update.
        """
        if env_ids.numel() == 0:
            return

        stage = self._terrain_stage_spec()
        self.terrain_stage_index[env_ids] = int(stage["index"]) + 1
        self.terrain_step_height_m[env_ids] = float(stage["step_height_m"])
        self.terrain_terrace_width_m[env_ids] = float(stage["terrace_width_m"])
        self.terrain_edge_smoothing[env_ids] = float(stage["edge_smoothing"])
        self.terrain_global_height_scale[env_ids] = float(
            stage.get("global_height_scale", 1.0)
        )
        self.terrain_local_irregularity_m[env_ids] = float(
            stage.get("local_irregularity_m", 0.0)
        )
        self.terrain_roughness_residual_m[env_ids] = float(
            stage.get("roughness_residual_m", 0.0)
        )

    def _episode_spawn_positions(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return spawn world positions on the heightmap for the current stage.

        Args:
            env_ids (torch.Tensor): Environments to sample spawns for.

        Returns:
            torch.Tensor: Positions ``[len(env_ids), 3]`` with Z from the heightmap.
        """
        stage = self._terrain_stage_spec()
        spawn_xy = torch.tensor(
            [float(stage["spawn_xy"][0]), float(stage["spawn_xy"][1])],
            device=self.device,
            dtype=gs.tc_float,
        )
        xy = spawn_xy.unsqueeze(0).repeat(env_ids.numel(), 1)
        terrain_z = sample_height_torch(
            self.terrain_heightmap_tensor,
            xy,
            size=tuple(self.terrain_config["size"]),
        )
        spawn_pos = torch.zeros(
            (env_ids.numel(), 3), device=self.device, dtype=gs.tc_float
        )
        spawn_pos[:, :2] = xy
        spawn_pos[:, 2] = terrain_z + float(self.base_init_pos[2].item())
        return spawn_pos

    def _update_local_terrain_features(self) -> None:
        """Update local heightmap slopes, normals, and climbability factors.

        Writes ``local_terrain_*`` and ``climbable_slope_factor`` from samples around
        each robot base position.
        """
        sample_distance_m = float(
            self.reward_config.get("slope_sample_distance_m", 0.22)
        )
        slope_rad, normal, gradient = estimate_heightmap_slopes_torch(
            self.terrain_heightmap_tensor,
            self.base_pos[:, :2],
            size=tuple(self.terrain_config["size"]),
            sample_distance_m=sample_distance_m,
        )
        self.local_terrain_slope_rad[:] = slope_rad
        self.local_terrain_slope_deg[:] = torch.rad2deg(slope_rad)
        self.local_terrain_normal[:] = normal
        self.local_terrain_gradient[:] = gradient

        forward_slope = torch.sum(gradient * self.lookahead_tangent_xy, dim=1)
        self.local_forward_slope_rad[:] = torch.atan(forward_slope)
        self.local_forward_slope_deg[:] = torch.rad2deg(self.local_forward_slope_rad)

        climbable_min_deg = float(
            self.reward_config.get("climbable_slope_deg_min", 3.0)
        )
        climbable_max_deg = float(
            self.reward_config.get("climbable_slope_deg_max", 16.0)
        )
        slope_span = max(climbable_max_deg - climbable_min_deg, 1.0e-6)
        climbable = (
            (self.local_terrain_slope_deg - climbable_min_deg) / slope_span
        ).clamp(0.0, 1.0)
        self.climbable_slope_factor[:] = climbable * climbable * (3.0 - 2.0 * climbable)

    def _reset_navigation(self, env_ids: torch.Tensor) -> None:
        """Reset Bezier planner and navigation buffers for ``env_ids``.

        Args:
            env_ids (torch.Tensor): Environments entering a new episode.
        """
        if env_ids.numel() == 0:
            return

        self.planner.reset_envs(
            env_ids=env_ids,
            robot_xy=self.base_pos[:, :2],
            robot_forward_xy=self._robot_forward_xy(),
            goal_xy=self.goal_pos[:, :2],
        )
        height_above_terrain = self.base_pos[:, 2] - self.terrain_height_at_robot
        self.nominal_base_height[env_ids] = height_above_terrain[env_ids]
        self.cmd_vel[env_ids] = 0.0
        self.arc_s[env_ids] = 0.0
        self.arc_progress[env_ids] = 0.0
        self.remaining_distance[env_ids] = 0.0
        self.cross_track_error[env_ids] = 0.0
        self.heading_error_to_tangent[env_ids] = 0.0
        self.goal_yaw_target[env_ids] = self.planner.goal_yaw_target[env_ids]
        self.goal_yaw_error[env_ids] = 0.0
        self.curve_length[env_ids] = self.planner.curve_length[env_ids]
        self.lookahead_curvature[env_ids] = 0.0
        self.lookahead_tangent_xy[env_ids] = 0.0
        self.deviation_mask[env_ids] = False
        self.within_approach_zone[env_ids] = False
        self.within_success_zone[env_ids] = False
        self.stagnation_counter[env_ids] = 0
        self.success_hold_counter[env_ids] = 0

        goal_delta = self.goal_pos[env_ids, :2] - self.base_pos[env_ids, :2]
        initial_distance = torch.linalg.vector_norm(goal_delta, dim=1)
        self.initial_goal_distance[env_ids] = initial_distance
        self.previous_goal_distance[env_ids] = initial_distance
        world_yaw = self._current_world_yaw()[env_ids]
        goal_bearing = torch.atan2(goal_delta[:, 1], goal_delta[:, 0])
        bearing_error = wrap_angle(goal_bearing - world_yaw)
        self.goal_bearing_error[env_ids] = bearing_error
        self.previous_goal_bearing_error[env_ids] = bearing_error
        self.initial_goal_bearing_error_abs[env_ids] = torch.abs(bearing_error)

    def _update_navigation_state(self) -> None:
        """Run one Bezier planner step and copy outputs into task tensors.

        Updates command velocity, arc length, cross-track error, and related fields.
        """
        planner_state = self.planner.step(
            robot_xy=self.base_pos[:, :2],
            robot_yaw=self._current_world_yaw(),
        )
        self.cmd_vel[:] = planner_state["cmd_vel"]
        self.arc_s[:] = planner_state["arc_s"]
        self.arc_progress[:] = planner_state["arc_progress"]
        self.remaining_distance[:] = planner_state["remaining_distance"]
        self.goal_distance[:] = planner_state["goal_distance"]
        self.cross_track_error[:] = planner_state["cross_track_error"]
        self.heading_error_to_tangent[:] = planner_state["heading_error_to_tangent"]
        self.goal_yaw_target[:] = planner_state["goal_yaw_target"]
        self.goal_yaw_error[:] = planner_state["goal_yaw_error"]
        self.curve_length[:] = planner_state["curve_length"]
        self.deviation_mask[:] = planner_state["deviation_mask"]
        self.within_approach_zone[:] = planner_state["within_approach_zone"]
        self.within_success_zone[:] = planner_state["within_success_zone"]
        self.lookahead_curvature[:] = planner_state["lookahead_curvature"]
        lt = planner_state["lookahead_tangent_xy"]
        lt_norm = torch.linalg.vector_norm(lt[:, :2], dim=1, keepdim=True).clamp(
            min=1.0e-6
        )
        self.lookahead_tangent_xy[:] = lt[:, :2] / lt_norm

    def _refresh_state(self) -> None:
        """Refresh robot, terrain, goal, and sensor state buffers from Genesis.

        This is the main synchronization point between physics state and the
        tensors consumed by rewards, terminations, and observations.
        """
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()

        relative_quat = transform_quat_by_quat(
            torch.ones_like(self.base_quat) * self.inv_base_init_quat,
            self.base_quat,
        )
        self.base_euler_rad[:] = quat_to_xyz(relative_quat, rpy=True, degrees=False)
        self.base_euler_deg[:] = torch.rad2deg(self.base_euler_rad)

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel_world[:] = self.robot.get_vel()
        self.base_lin_vel_body[:] = transform_by_quat(
            self.base_lin_vel_world, inv_base_quat
        )
        self.base_lin_vel[:] = self.base_lin_vel_body
        self.projected_gravity[:] = transform_by_quat(self.gravity_vec, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dof_indices)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dof_indices)
        self.dof_acc[:] = (self.dof_vel - self.last_dof_vel) / max(self.dt, 1e-6)
        self.joint_torques[:] = self._safe_get_joint_torques()
        self.terrain_height_at_robot[:] = sample_height_torch(
            self.terrain_heightmap_tensor,
            self.base_pos[:, :2],
            size=tuple(self.terrain_config["size"]),
        )
        self.foot_positions[:], self.foot_velocities[:] = (
            self._safe_get_foot_kinematics()
        )
        foot_positions_centered = self.foot_positions - self.base_pos.unsqueeze(1)
        self.foot_positions_body[:] = transform_by_quat(
            foot_positions_centered.reshape(-1, 3),
            inv_base_quat.repeat_interleave(4, dim=0),
        ).reshape(self.num_envs, 4, 3)
        self._prev_foot_positions_world[:] = self.foot_positions

        sensor_data = read_stack(
            self.sensor_stack,
            lidar_noise_std=(
                float(self.sensors_config["lidar"].get("noise_std", 0.0))
                if bool(self.sensors_config["lidar"].get("enable_noise", False))
                else 0.0
            ),
            num_envs=self.num_envs,
        )
        self.imu_gyro[:] = sensor_data["imu_gyro"]
        self.imu_acc[:] = sensor_data["imu_acc"]
        self.imu_mag[:] = sensor_data["imu_mag"]
        self.foot_force[:] = sensor_data["foot_force"]
        self.foot_force_norm[:] = sensor_data["foot_force_norm"]
        self.foot_contact[:] = self.foot_force_norm >= float(
            self.sensors_config["feet"]["contact_force_threshold"]
        )

        goal_vector_world = self.goal_pos - self.base_pos
        direct_goal_distance = torch.linalg.vector_norm(goal_vector_world[:, :2], dim=1)
        if not torch.any(self.curve_length > 0.0):
            self.goal_distance[:] = direct_goal_distance

        world_yaw = self._current_world_yaw()
        goal_bearing = torch.atan2(goal_vector_world[:, 1], goal_vector_world[:, 0])
        self.goal_bearing_error[:] = wrap_angle(goal_bearing - world_yaw)

    def _update_foot_events(self) -> None:
        """Update airtime and impact terms derived from foot contact events.

        These derived quantities are reused by both reward computation and
        episode-level metric summaries.
        """
        planar_speed = torch.linalg.vector_norm(self.base_lin_vel_body[:, :2], dim=1)
        move_mask = planar_speed > float(self.reward_config["airtime_speed_threshold"])
        just_contact = (~self.last_foot_contact) & self.foot_contact

        airtime_cap = float(self.reward_config["airtime_cap_s"])
        airtime_value = torch.clamp(self.foot_airtime, min=0.0, max=airtime_cap)
        self.foot_airtime_term[:] = (
            airtime_value * just_contact.float() * move_mask[:, None].float()
        ).sum(dim=1)

        impact_threshold = float(self.sensors_config["feet"]["impact_force_threshold"])
        self.impact_term[:] = torch.clamp(
            self.foot_force_norm - impact_threshold,
            min=0.0,
        ).mean(dim=1)

        self.foot_airtime = torch.where(
            self.foot_contact,
            torch.zeros_like(self.foot_airtime),
            self.foot_airtime + self.dt,
        )
        self.last_foot_contact[:] = self.foot_contact

    def _compute_observations(self) -> torch.Tensor:
        """Assemble the observation tensor consumed by policies.

        Observation blocks are concatenated in the same order described by
        ``self.obs_parts`` so downstream consumers stay aligned.

        Returns:
            torch.Tensor: Batched observation tensor for all environments.
        """
        heading_error = torch.stack(
            [
                torch.sin(self.heading_error_to_tangent),
                torch.cos(self.heading_error_to_tangent),
            ],
            dim=1,
        )
        goal_heading = torch.stack(
            [
                torch.sin(self.goal_yaw_error),
                torch.cos(self.goal_yaw_error),
            ],
            dim=1,
        )

        parts = [
            self.projected_gravity * self.obs_scales["projected_gravity"],
            self.imu_gyro * self.obs_scales["imu_gyro"],
            self.cmd_vel * self.obs_scales["cmd_vel"],
            self.cross_track_error.unsqueeze(1) * self.obs_scales["cross_track_error"],
            heading_error * self.obs_scales["heading_error"],
            self.remaining_distance.unsqueeze(1)
            * self.obs_scales["remaining_distance"],
            goal_heading * self.obs_scales["goal_heading"],
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.foot_contact.to(gs.tc_float),
            self.last_actions,
        ]
        self.obs_buf = torch.cat(parts, dim=1)
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf

    def _compute_dense_rewards(self) -> dict[str, torch.Tensor]:
        """Evaluate scaled dense rewards for the current step.

        Applies optional reward curriculum gating and per-term caps, accumulates
        ``episode_sums`` and ``episode_reward_sum``.

        Returns:
            dict[str, torch.Tensor]: Per-term reward tensors before terminal bonuses.
        """
        reward_terms: dict[str, torch.Tensor] = {}
        step_reward_sum = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )

        global_step = getattr(self, "global_step", 0)
        curriculum_enabled = getattr(self, "_curriculum_enabled", False)
        stage_step_thresholds = getattr(
            self, "_curriculum_stage_steps", [0, 2_000_000, 5_000_000]
        )
        term_to_min_stage = getattr(self, "_term_to_min_stage", {})
        reward_caps = getattr(self, "_reward_caps", {})

        current_stage_index = 0
        for stage_index, threshold in enumerate(stage_step_thresholds):
            if global_step >= threshold:
                current_stage_index = stage_index

        curriculum_disabled = getattr(self, "_disable_reward_curriculum", False)

        for name, reward_func in self.reward_functions.items():
            base_scale = float(self.reward_scales[name])
            if curriculum_enabled and not curriculum_disabled:
                minimum_stage = term_to_min_stage.get(name, 999)
                effective_scale = (
                    base_scale if current_stage_index >= minimum_stage else 0.0
                )
            else:
                effective_scale = base_scale

            term_value = reward_func(self) * effective_scale
            if name in reward_caps:
                cap_magnitude = reward_caps[name]
                term_value = torch.clamp(term_value, -cap_magnitude, cap_magnitude)

            reward_terms[name] = term_value
            self.episode_sums[name] += term_value
            step_reward_sum += term_value

        self.episode_reward_sum += step_reward_sum

        return reward_terms

    def _current_curriculum_stage_index(self) -> int:
        """Return the reward curriculum stage index from ``global_step``.

        Returns:
            int: Index into ``_curriculum_stage_steps`` thresholds.
        """
        global_step = getattr(self, "global_step", 0)
        current_stage_index = 0
        for stage_index, threshold in enumerate(
            getattr(self, "_curriculum_stage_steps", [0, 3_000_000, 8_000_000])
        ):
            if global_step >= threshold:
                current_stage_index = stage_index

        return current_stage_index

    def _effective_reward_scale(self, name: str, base_scale: float) -> float:
        """Return the scale for one reward term after curriculum gating.

        Args:
            name (str): Reward term key in ``reward_functions``.
            base_scale (float): YAML-configured scale before gating.

        Returns:
            float: ``0.0`` when the term is disabled for the current stage, else
                ``base_scale``.
        """
        if not getattr(self, "_curriculum_enabled", False):
            return base_scale
        if getattr(self, "_disable_reward_curriculum", False):
            return base_scale

        minimum_stage = getattr(self, "_term_to_min_stage", {}).get(name, 999)
        return (
            base_scale
            if self._current_curriculum_stage_index() >= minimum_stage
            else 0.0
        )

    def _apply_terminal_rewards(
        self,
        reward_terms: dict[str, torch.Tensor],
        termination,
    ) -> None:
        """Add terminal reward terms to the current reward dictionary.

        Args:
            reward_terms (dict[str, torch.Tensor]): Dense reward terms already
                computed for the step.
            termination: Structured termination flags for the current step.
        """
        base_terminal_scales = dict(self.terminal_reward_scales)
        self.terminal_reward_scales = {
            name: self._effective_reward_scale(name, float(scale))
            for name, scale in base_terminal_scales.items()
        }
        terminal_terms = terminal_reward_terms(self, termination)
        self.terminal_reward_scales = base_terminal_scales
        reward_terms.update(terminal_terms)

        for name, value in terminal_terms.items():
            if name not in self.episode_sums:
                self.episode_sums[name] = torch.zeros(
                    (self.num_envs,), device=self.device, dtype=gs.tc_float
                )
            self.episode_sums[name] += value
            self.episode_reward_sum += value

    def _update_episode_metrics(self) -> None:
        """Accumulate episode-level metrics used for evaluation summaries.

        The stored sums are normalized only when an episode terminates.
        """
        planar_speed = torch.linalg.vector_norm(self.base_lin_vel_body[:, :2], dim=1)
        vertical_speed_abs = torch.abs(self.base_lin_vel_world[:, 2])
        arc_progress_speed = torch.clamp(
            self.arc_progress / max(self.dt, 1e-6), -0.5, 1.5
        )
        moving_mask = planar_speed > float(
            self.reward_config.get("moving_threshold_mps", 0.15)
        )
        reverse_motion = (self.base_lin_vel_body[:, 0] < -0.05).to(gs.tc_float)
        lateral_motion_ratio = (
            torch.abs(self.base_lin_vel_body[:, 1])
            / torch.clamp(planar_speed, min=1.0e-6)
        ) * moving_mask.to(gs.tc_float)
        airborne_ratio = (self.foot_contact.float().sum(dim=1) == 0).to(gs.tc_float)
        trot_contact_score = REWARD_FUNCTIONS["trot_contact"](self)

        self.episode_metric_sums["planar_speed"] += planar_speed
        self.episode_metric_sums["vertical_speed_abs"] += vertical_speed_abs
        self.episode_metric_sums["mean_cross_track_error"] += torch.abs(
            self.cross_track_error
        )
        self.episode_metric_sums["mean_arc_progress_speed"] += arc_progress_speed
        self.episode_metric_sums["reverse_motion_ratio"] += reverse_motion
        self.episode_metric_sums["lateral_motion_ratio"] += lateral_motion_ratio
        self.episode_metric_sums["airborne_ratio"] += airborne_ratio
        self.episode_metric_sums["trot_contact_score"] += trot_contact_score
        self.episode_metric_sums["turn_in_place_steps"] += turn_in_place_mask(self).to(
            gs.tc_float
        )
        self.episode_metric_sums["foot_airtime"] += self.foot_airtime_term
        self.episode_metric_sums["impact"] += self.impact_term
        self.episode_metric_sums["mean_curve_progress_gate"] += self.curve_progress_gate
        self.episode_metric_sums["mean_local_slope_deg"] += self.local_terrain_slope_deg
        self.episode_metric_sums["mean_forward_slope_deg"] += (
            self.local_forward_slope_deg
        )

    def _build_done_infos(
        self,
        done_ids: torch.Tensor,
        terminal_obs: torch.Tensor,
        termination,
    ) -> list[dict[str, Any]]:
        """Build Gym/SB3-compatible info dictionaries for completed episodes.

        Args:
            done_ids (torch.Tensor): Environment indices that terminated.
            terminal_obs (torch.Tensor): Terminal observations captured before
                reset.
            termination: Structured termination flags for the current step.

        Returns:
            list[dict[str, Any]]: One info dictionary per environment slot.
        """
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        for index in done_ids.tolist():
            reason = "timeout"
            if bool(termination.success[index].item()):
                reason = "success"
            elif bool(termination.fall[index].item()):
                reason = "fall"
            elif bool(termination.posture[index].item()):
                reason = "posture"
            elif bool(termination.curve_deviation[index].item()):
                reason = "curve_deviation"
            elif bool(termination.stagnation[index].item()):
                reason = "stagnation"

            episode_length = max(int(self.episode_length_buf[index].item()), 1)
            final_planar_speed = torch.linalg.vector_norm(
                self.base_lin_vel_body[index, :2], dim=0
            )
            final_yaw_rate_deg_s = torch.rad2deg(torch.abs(self.imu_gyro[index, 2]))
            mean_arc_progress_speed = (
                self.episode_metric_sums["mean_arc_progress_speed"][index]
                / episode_length
            )
            mean_cross_track_error = (
                self.episode_metric_sums["mean_cross_track_error"][index]
                / episode_length
            )
            curve_progress_fraction = (
                self.arc_s[index] / torch.clamp(self.curve_length[index], min=1.0e-6)
            ).clamp(0.0, 1.0)
            metric_payload = {
                "success": float(termination.success[index].item()),
                "time_to_goal_s": float(episode_length * self.dt),
                "final_goal_distance": float(self.goal_distance[index].item()),
                "final_goal_yaw_error_deg": float(
                    torch.rad2deg(torch.abs(self.goal_yaw_error[index])).item()
                ),
                "planar_speed": float(
                    (
                        self.episode_metric_sums["planar_speed"][index] / episode_length
                    ).item()
                ),
                "vertical_speed_abs": float(
                    (
                        self.episode_metric_sums["vertical_speed_abs"][index]
                        / episode_length
                    ).item()
                ),
                "mean_cross_track_error": float(mean_cross_track_error.item()),
                "mean_arc_progress_speed": float(mean_arc_progress_speed.item()),
                "reverse_motion_ratio": float(
                    (
                        self.episode_metric_sums["reverse_motion_ratio"][index]
                        / episode_length
                    ).item()
                ),
                "lateral_motion_ratio": float(
                    (
                        self.episode_metric_sums["lateral_motion_ratio"][index]
                        / episode_length
                    ).item()
                ),
                "airborne_ratio": float(
                    (
                        self.episode_metric_sums["airborne_ratio"][index]
                        / episode_length
                    ).item()
                ),
                "trot_contact_score": float(
                    (
                        self.episode_metric_sums["trot_contact_score"][index]
                        / episode_length
                    ).item()
                ),
                "turn_in_place_fraction": float(
                    (
                        self.episode_metric_sums["turn_in_place_steps"][index]
                        / episode_length
                    ).item()
                ),
                "mean_stop_speed_at_goal": float(final_planar_speed.item()),
                "final_planar_speed": float(final_planar_speed.item()),
                "final_yaw_rate_deg_s": float(final_yaw_rate_deg_s.item()),
                "impact": float(
                    (self.episode_metric_sums["impact"][index] / episode_length).item()
                ),
                "foot_airtime": float(
                    (
                        self.episode_metric_sums["foot_airtime"][index] / episode_length
                    ).item()
                ),
                "curve_deviation": float(termination.curve_deviation[index].item()),
                "stagnation": float(termination.stagnation[index].item()),
                "timeout": float(termination.timeout[index].item()),
                "arc_progress_fraction": float(curve_progress_fraction.item()),
                "mean_curve_progress_gate": float(
                    (
                        self.episode_metric_sums["mean_curve_progress_gate"][index]
                        / episode_length
                    ).item()
                ),
                "mean_local_slope_deg": float(
                    (
                        self.episode_metric_sums["mean_local_slope_deg"][index]
                        / episode_length
                    ).item()
                ),
                "mean_forward_slope_deg": float(
                    (
                        self.episode_metric_sums["mean_forward_slope_deg"][index]
                        / episode_length
                    ).item()
                ),
                "final_local_slope_deg": float(
                    self.local_terrain_slope_deg[index].item()
                ),
                "final_forward_slope_deg": float(
                    self.local_forward_slope_deg[index].item()
                ),
                "terrain_stage": float(self.terrain_stage_index[index].item()),
                "terrain_step_height_m": float(
                    self.terrain_step_height_m[index].item()
                ),
                "terrain_terrace_width_m": float(
                    self.terrain_terrace_width_m[index].item()
                ),
                "terrain_edge_smoothing": float(
                    self.terrain_edge_smoothing[index].item()
                ),
                "terrain_global_height_scale": float(
                    self.terrain_global_height_scale[index].item()
                ),
                "terrain_local_irregularity_m": float(
                    self.terrain_local_irregularity_m[index].item()
                ),
                "terrain_roughness_residual_m": float(
                    self.terrain_roughness_residual_m[index].item()
                ),
            }

            episode_info: dict[str, Any] = {
                "r": float(self.episode_reward_sum[index].item()),
                "l": episode_length,
                "termination_reason": reason,
            }
            for reward_name, reward_sum in self.episode_sums.items():
                episode_info[f"rew_{reward_name}"] = float(reward_sum[index].item())
            for metric_name, metric_value in metric_payload.items():
                episode_info[f"metric_{metric_name}"] = float(metric_value)

            info_payload = {
                "episode": episode_info,
                "terminal_observation": terminal_obs[index].detach().cpu().numpy(),
                "termination_reason": reason,
                "TimeLimit.truncated": reason == "timeout",
            }
            for reward_name, reward_sum in self.episode_sums.items():
                info_payload[f"rew_{reward_name}"] = float(reward_sum[index].item())
            for metric_name, metric_value in metric_payload.items():
                info_payload[f"metric_{metric_name}"] = float(metric_value)

            infos[index] = info_payload

        return infos

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset a subset of environments to the initial robot and goal state.

        Args:
            env_ids (torch.Tensor): Environment indices to reset.
        """
        if env_ids.numel() == 0:
            return

        n_reset = env_ids.numel()
        default_expanded = self.default_dof_pos.unsqueeze(0).expand(n_reset, -1)
        if self._init_joint_noise_rad > 0:
            noise = (
                torch.rand(
                    n_reset, self.num_actions, device=self.device, dtype=gs.tc_float
                )
                * 2
                - 1
            ) * self._init_joint_noise_rad
            init_pos = default_expanded + noise
            lower, upper = self.robot.get_dofs_limit(self.motor_dof_indices)
            if lower.ndim > 1:
                lower = lower[0]
            if upper.ndim > 1:
                upper = upper[0]
            init_pos = torch.clamp(init_pos, lower.unsqueeze(0), upper.unsqueeze(0))
        else:
            init_pos = default_expanded

        self.robot.set_dofs_position(
            position=init_pos,
            dofs_idx_local=self.motor_dof_indices,
            zero_velocity=True,
            envs_idx=env_ids,
        )

        self._apply_terrain_stage(env_ids)
        base_pos = self._episode_spawn_positions(env_ids)
        self.spawn_pos[env_ids] = base_pos
        base_quat = self.base_init_quat.repeat(env_ids.numel(), 1)
        self.robot.set_pos(base_pos, zero_velocity=False, envs_idx=env_ids)
        self.robot.set_quat(base_quat, zero_velocity=False, envs_idx=env_ids)
        self.robot.zero_all_dofs_velocity(envs_idx=env_ids)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.target_dof_pos[env_ids] = init_pos
        self.dof_vel[env_ids] = 0.0
        self.dof_acc[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.joint_torques[env_ids] = 0.0
        self.base_lin_vel_world[env_ids] = 0.0
        self.base_lin_vel_body[env_ids] = 0.0
        self.base_lin_vel[env_ids] = 0.0
        self.foot_airtime[env_ids] = 0.0
        self.foot_airtime_term[env_ids] = 0.0
        self.impact_term[env_ids] = 0.0
        self.last_foot_contact[env_ids] = False
        self._prev_foot_positions_world[env_ids] = 0.0
        self.success_hold_counter[env_ids] = 0
        self.stagnation_counter[env_ids] = 0
        self.bad_posture_counter[env_ids] = 0
        self.critical_posture_counter[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.episode_reward_sum[env_ids] = 0.0
        self.previous_goal_bearing_error[env_ids] = 0.0
        self.initial_goal_bearing_error_abs[env_ids] = 0.0
        self.cmd_vel[env_ids] = 0.0
        self.arc_s[env_ids] = 0.0
        self.arc_progress[env_ids] = 0.0
        self.remaining_distance[env_ids] = 0.0
        self.cross_track_error[env_ids] = 0.0
        self.heading_error_to_tangent[env_ids] = 0.0
        self.goal_yaw_target[env_ids] = 0.0
        self.goal_yaw_error[env_ids] = 0.0
        self.curve_length[env_ids] = 0.0
        self.lookahead_curvature[env_ids] = 0.0
        self.lookahead_tangent_xy[env_ids] = 0.0
        self.deviation_mask[env_ids] = False
        self.within_approach_zone[env_ids] = False
        self.within_success_zone[env_ids] = False
        self.nominal_base_height[env_ids] = 0.0
        self.local_terrain_slope_rad[env_ids] = 0.0
        self.local_terrain_slope_deg[env_ids] = 0.0
        self.local_forward_slope_rad[env_ids] = 0.0
        self.local_forward_slope_deg[env_ids] = 0.0
        self.local_terrain_gradient[env_ids] = 0.0
        self.local_terrain_normal[env_ids] = 0.0
        self.local_terrain_normal[env_ids, 2] = 1.0
        self.climbable_slope_factor[env_ids] = 0.0
        self.curve_progress_gate[env_ids] = 0.0
        self.curve_progress_heading_gate[env_ids] = 0.0
        self.curve_progress_corridor_gate[env_ids] = 0.0

        for reward_sum in self.episode_sums.values():
            reward_sum[env_ids] = 0.0
        for metric_sum in self.episode_metric_sums.values():
            metric_sum[env_ids] = 0.0

        sample_goal_pose(self, env_ids)
        reset_goal_trackers(self, env_ids)

    def reset(self) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Reset all environments and return fresh observations.

        One physics step is executed after resetting so sensors and kinematic
        buffers reflect a valid post-reset state.

        Returns:
            tuple[torch.Tensor, list[dict[str, Any]]]: Observation tensor and an
                empty info dictionary for each environment.
        """
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_idx(env_ids)
        self.scene.step()
        self._refresh_state()
        self._reset_navigation(env_ids)
        self._update_navigation_state()
        self._update_local_terrain_features()
        self._update_video_camera_pose()
        self.previous_goal_distance[:] = self.goal_distance
        self.initial_goal_bearing_error_abs[:] = torch.abs(self.goal_bearing_error)
        self.previous_goal_bearing_error[:] = self.goal_bearing_error
        self._refresh_markers()
        observations = self._compute_observations()

        return observations, [{} for _ in range(self.num_envs)]

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Advance one control step and return SB3-compatible outputs.

        Args:
            actions (torch.Tensor): Batched action tensor from the policy.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
                Next observations, rewards, done mask, and info dictionaries.
        """
        self.actions = torch.clamp(
            actions,
            -float(self.control_config["clip_actions"]),
            float(self.control_config["clip_actions"]),
        )
        executed_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = (
            executed_actions * float(self.control_config["action_scale"])
            + self.default_dof_pos
        )
        self.target_dof_pos[:] = target_dof_pos

        self.robot.control_dofs_position(target_dof_pos, self.motor_dof_indices)
        self.scene.step()

        self.episode_length_buf += 1
        self._refresh_state()
        self._update_navigation_state()
        self._update_local_terrain_features()
        self._update_video_camera_pose()
        self._update_foot_events()
        terminal_obs = self._compute_observations().clone()

        reward_terms = self._compute_dense_rewards()
        termination_mask, termination = compute_termination(self)
        self._apply_terminal_rewards(reward_terms, termination)
        self._update_episode_metrics()

        total_reward = torch.zeros_like(self.rew_buf)
        for reward_term in reward_terms.values():
            total_reward += reward_term
        self.rew_buf[:] = total_reward
        if bool(self.reward_config.get("only_positive_rewards", False)):
            self.rew_buf.clamp_(min=0.0)

        done_ids = termination_mask.nonzero(as_tuple=False).flatten()
        infos = self._build_done_infos(done_ids, terminal_obs, termination)

        self.previous_goal_distance[:] = self.goal_distance
        self.previous_goal_bearing_error[:] = self.goal_bearing_error
        if done_ids.numel() > 0:
            self._reset_idx(done_ids)
            self.scene.step()
            self._refresh_state()
            self._reset_navigation(done_ids)
            self._update_navigation_state()
            self._update_local_terrain_features()
            self._update_video_camera_pose()
            self.previous_goal_distance[done_ids] = self.goal_distance[done_ids]
            self.initial_goal_bearing_error_abs[done_ids] = torch.abs(
                self.goal_bearing_error[done_ids]
            )
            self.previous_goal_bearing_error[done_ids] = self.goal_bearing_error[
                done_ids
            ]
        self._refresh_markers()

        next_obs = self._compute_observations()
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.reset_buf[:] = termination_mask
        self.global_step = getattr(self, "global_step", 0) + self.num_envs

        return next_obs, self.rew_buf.clone(), termination_mask.clone(), infos

    def capture_rgb_frame(self) -> np.ndarray | None:
        """Capture the current RGB frame when the optional camera is enabled.

        The method accepts both direct array returns and tuple-based render
        results from Genesis.

        Returns:
            np.ndarray | None: Rendered RGB frame or ``None`` when capture is
                unavailable.
        """
        if self.camera is None:
            return None

        try:
            rendered = self.camera.render()
        except Exception:
            return None

        if isinstance(rendered, np.ndarray):
            return rendered
        if isinstance(rendered, tuple) and rendered:
            first = rendered[0]
            if isinstance(first, np.ndarray):
                return first

        return None

    def close(self) -> None:
        """Destroy the Genesis scene and swallow cleanup errors.

        Cleanup is intentionally best-effort so shutdown paths stay simple.
        """
        try:
            self.scene.destroy()
        except Exception:
            pass

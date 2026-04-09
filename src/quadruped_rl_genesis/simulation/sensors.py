"""IMU, radar, and foot contact sensors for the Go2 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    import genesis as gs
except ImportError:
    gs = None


GO2_IMU_OFFSET = (-0.02557, 0.0, 0.04232)
GO2_RADAR_OFFSET = (0.28945, 0.0, -0.046825)
GO2_RADAR_EULER = (0.0, 164.908551, 0.0)
GO2_FEET = ("FL", "FR", "RL", "RR")
GO2_FOOT_CANDIDATES = {
    "FL": ("FL_foot", "FL_calf"),
    "FR": ("FR_foot", "FR_calf"),
    "RL": ("RL_foot", "RL_calf"),
    "RR": ("RR_foot", "RR_calf"),
}
GO2_LINKS_TO_KEEP = (
    "imu",
    "radar",
    *tuple(GO2_FOOT_CANDIDATES[leg][0] for leg in GO2_FEET),
)


@dataclass(frozen=True)
class Mount:
    """Resolved link mount and rigid offset for a robot sensor.

    Attributes:
        link_name (str): Name of the robot link the sensor is mounted on.
        link_idx_local (int): Local link index within the robot entity.
        pos_offset (tuple[float, float, float]): Position offset from link origin.
        euler_offset (tuple[float, float, float]): Euler angle offset in radians.
    """

    link_name: str
    link_idx_local: int
    pos_offset: tuple[float, float, float]
    euler_offset: tuple[float, float, float]


@dataclass
class Stack:
    """Runtime handles for the mounted Genesis sensor stack.

    Attributes:
        imu (Any | None): IMU sensor handle or None if not mounted.
        lidar (Any | None): LiDAR sensor handle or None if not mounted.
        feet (tuple[Any, ...]): Contact-force sensor handles per leg.
        lidar_max_range (float): Maximum LiDAR range in meters.
        lidar_min_range (float): Minimum LiDAR range in meters.
        lidar_no_hit_value (float): Value returned when no hit is detected.
        lidar_bins_azimuth (int): Number of azimuth bins for sector aggregation.
        lidar_bins_elevation (int): Number of elevation bins.
    """

    imu: Any | None
    lidar: Any | None
    feet: tuple[Any, ...]
    lidar_max_range: float
    lidar_min_range: float
    lidar_no_hit_value: float
    lidar_bins_azimuth: int
    lidar_bins_elevation: int


def _require_genesis() -> None:
    """Ensure Genesis is available before building or querying sensors.

    Raises:
        ImportError: If Genesis could not be imported in the current runtime.
    """
    if gs is None:
        raise ImportError(
            "Genesis is required to build or read the robot sensor stack."
        )


def _lookup_link(robot: Any, name: str) -> Any | None:
    """Try to resolve a robot link by name without surfacing lookup errors.

    Args:
        robot (Any): Genesis robot entity.
        name (str): Link name to resolve.

    Returns:
        Any | None: Link handle when found, otherwise ``None``.
    """
    try:
        return robot.get_link(name)
    except Exception:
        return None


def resolve_base(robot: Any) -> Mount:
    """Resolve the robot base mount using ``base`` or ``base_link``.

    Args:
        robot (Any): Genesis robot entity.

    Returns:
        Mount: Base-frame mount with zero offsets.

    Raises:
        KeyError: If neither ``base`` nor ``base_link`` can be resolved.
    """
    for name in ("base", "base_link"):
        link = _lookup_link(robot, name)

        if link is not None:
            return Mount(
                link_name=name,
                link_idx_local=int(link.idx_local),
                pos_offset=(0.0, 0.0, 0.0),
                euler_offset=(0.0, 0.0, 0.0),
            )
    raise KeyError("Neither 'base' nor 'base_link' exists in the robot URDF.")


def resolve_mount(
    robot: Any,
    *,
    preferred_link: str,
    fallback: Mount,
    fallback_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fallback_euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Mount:
    """Resolve a mount by preferred link or fall back to a base-frame offset.

    Args:
        robot (Any): Genesis robot entity.
        preferred_link (str): Link name preferred for the sensor mount.
        fallback (Mount): Base mount used when the preferred link is missing.
        fallback_offset (tuple[float, float, float], optional): Position offset
            applied in the fallback frame.
        fallback_euler (tuple[float, float, float], optional): Euler rotation
            offset applied in the fallback frame.

    Returns:
        Mount: Resolved mount with either the preferred link or the fallback
            base-frame offset.
    """
    link = _lookup_link(robot, preferred_link)

    if link is not None:
        return Mount(
            link_name=preferred_link,
            link_idx_local=int(link.idx_local),
            pos_offset=(0.0, 0.0, 0.0),
            euler_offset=(0.0, 0.0, 0.0),
        )

    return Mount(
        link_name=fallback.link_name,
        link_idx_local=fallback.link_idx_local,
        pos_offset=fallback_offset,
        euler_offset=fallback_euler,
    )


def resolve_feet(robot: Any) -> tuple[Mount, ...]:
    """Resolve one contact-force mount per leg.

    Args:
        robot (Any): Genesis robot entity.

    Returns:
        tuple[Mount, ...]: Mounts for the four Go2 feet.

    Raises:
        KeyError: If any leg cannot be mapped to a known contact link.
    """
    mounts: list[Mount] = []

    for leg in GO2_FEET:
        for candidate in GO2_FOOT_CANDIDATES[leg]:
            link = _lookup_link(robot, candidate)

            if link is not None:
                mounts.append(
                    Mount(
                        link_name=candidate,
                        link_idx_local=int(link.idx_local),
                        pos_offset=(0.0, 0.0, 0.0),
                        euler_offset=(0.0, 0.0, 0.0),
                    )
                )
                break
        else:
            tried = ", ".join(GO2_FOOT_CANDIDATES[leg])
            raise KeyError(
                f"Could not resolve a contact link for leg '{leg}'. Tried: {tried}."
            )

    return tuple(mounts)


def lidar_feature_size(config: dict[str, Any]) -> int:
    """Return the flattened LiDAR feature size after sector aggregation.

    Args:
        config (dict[str, Any]): LiDAR configuration payload with azimuth and
            elevation bin counts.

    Returns:
        int: Flattened feature size containing sector-wise min and mean values.
    """
    bins_azimuth = int(config["bins_azimuth"])
    bins_elevation = int(config["bins_elevation"])

    return bins_azimuth * bins_elevation * 2


def aggregate_lidar(
    distances: torch.Tensor,
    *,
    max_range: float,
    bins_azimuth: int,
    bins_elevation: int,
) -> torch.Tensor:
    """Aggregate raw LiDAR distances into sector-wise min and mean features.

    Args:
        distances (torch.Tensor): Raw LiDAR tensor with shape
            ``[env, azimuth, elevation]``.
        max_range (float): Maximum LiDAR range used for normalization.
        bins_azimuth (int): Number of azimuth sectors.
        bins_elevation (int): Number of elevation sectors.

    Returns:
        torch.Tensor: Flattened feature tensor concatenating normalized sector
            minima and means.

    Raises:
        ValueError: If the input shape is invalid or incompatible with the
            requested sector counts.
    """
    if distances.ndim != 3:
        raise ValueError(
            f"Expected LiDAR distances with shape [env, azimuth, elevation], got {tuple(distances.shape)}."
        )

    envs, azimuth_points, elevation_points = distances.shape

    if azimuth_points % bins_azimuth != 0 or elevation_points % bins_elevation != 0:
        raise ValueError(
            "LiDAR ray pattern must divide evenly into the requested bins. "
            f"Got raw=({azimuth_points}, {elevation_points}) and bins=({bins_azimuth}, {bins_elevation})."
        )

    azimuth_stride = azimuth_points // bins_azimuth
    elevation_stride = elevation_points // bins_elevation
    shaped = distances.view(
        envs,
        bins_azimuth,
        azimuth_stride,
        bins_elevation,
        elevation_stride,
    )
    mins = shaped.amin(dim=(2, 4))
    means = shaped.mean(dim=(2, 4))
    scale = max(float(max_range), 1e-6)

    mins = torch.clamp(mins / scale, 0.0, 1.0)
    means = torch.clamp(means / scale, 0.0, 1.0)

    return torch.cat([mins.flatten(start_dim=1), means.flatten(start_dim=1)], dim=1)


def build_stack(
    *,
    scene: Any,
    robot: Any,
    config: dict[str, Any],
    draw_debug: bool = False,
) -> Stack:
    """Mount the Go2 sensor stack on the robot and return runtime handles.

    Args:
        scene (Any): Genesis scene that owns the robot and sensors.
        robot (Any): Genesis robot entity.
        config (dict[str, Any]): Sensor configuration payload.
        draw_debug (bool, optional): Whether eligible sensors should render
            debug geometry.

    Returns:
        Stack: Runtime handles and normalization metadata for the mounted sensor
            stack.
    """
    _require_genesis()

    base = resolve_base(robot)
    imu_mount = resolve_mount(
        robot,
        preferred_link="imu",
        fallback=base,
        fallback_offset=GO2_IMU_OFFSET,
    )
    lidar_mount = resolve_mount(
        robot,
        preferred_link="radar",
        fallback=base,
        fallback_offset=GO2_RADAR_OFFSET,
        fallback_euler=GO2_RADAR_EULER,
    )
    foot_mounts = resolve_feet(robot)

    imu_config = config["imu"]
    lidar_config = config["lidar"]
    feet_config = config["feet"]

    imu = None
    if bool(imu_config.get("enabled", True)):
        imu = scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=robot.idx,
                link_idx_local=imu_mount.link_idx_local,
                pos_offset=imu_mount.pos_offset,
                euler_offset=imu_mount.euler_offset,
                magnetic_field=tuple(imu_config.get("magnetic_field", (0.3, 0.0, 0.5))),
                acc_noise=tuple(imu_config.get("acc_noise", (0.0, 0.0, 0.0))),
                gyro_noise=tuple(imu_config.get("gyro_noise", (0.0, 0.0, 0.0))),
                mag_noise=tuple(imu_config.get("mag_noise", (0.0, 0.0, 0.0))),
                acc_bias=tuple(imu_config.get("acc_bias", (0.0, 0.0, 0.0))),
                gyro_bias=tuple(imu_config.get("gyro_bias", (0.0, 0.0, 0.0))),
                mag_bias=tuple(imu_config.get("mag_bias", (0.0, 0.0, 0.0))),
                acc_random_walk=tuple(
                    imu_config.get("acc_random_walk", (0.0, 0.0, 0.0))
                ),
                gyro_random_walk=tuple(
                    imu_config.get("gyro_random_walk", (0.0, 0.0, 0.0))
                ),
                mag_random_walk=tuple(
                    imu_config.get("mag_random_walk", (0.0, 0.0, 0.0))
                ),
                delay=float(imu_config.get("delay", 0.0)),
                jitter=float(imu_config.get("jitter", 0.0)),
                interpolate=bool(imu_config.get("interpolate", False)),
                draw_debug=draw_debug and bool(imu_config.get("draw_debug", False)),
            )
        )

    feet: tuple[Any, ...] = ()
    if bool(feet_config.get("enabled", True)):
        feet = tuple(
            scene.add_sensor(
                gs.sensors.ContactForce(
                    entity_idx=robot.idx,
                    link_idx_local=mount.link_idx_local,
                    min_force=float(feet_config.get("min_force", 0.0)),
                    max_force=tuple(
                        feet_config.get("max_force", (200.0, 200.0, 200.0))
                    ),
                    noise=tuple(feet_config.get("noise", (0.0, 0.0, 0.0))),
                    bias=tuple(feet_config.get("bias", (0.0, 0.0, 0.0))),
                    random_walk=tuple(feet_config.get("random_walk", (0.0, 0.0, 0.0))),
                    delay=float(feet_config.get("delay", 0.0)),
                    jitter=float(feet_config.get("jitter", 0.0)),
                    interpolate=bool(feet_config.get("interpolate", False)),
                    draw_debug=draw_debug
                    and bool(feet_config.get("draw_debug", False)),
                )
            )
            for mount in foot_mounts
        )

    lidar = None
    lidar_max_range = 0.0
    lidar_min_range = 0.0
    lidar_no_hit_value = 0.0
    lidar_bins_azimuth = 0
    lidar_bins_elevation = 0

    if bool(lidar_config.get("enabled", True)):
        lidar_pattern = gs.sensors.SphericalPattern(
            fov=(
                tuple(lidar_config["horizontal_fov_deg"]),
                tuple(lidar_config["vertical_fov_deg"]),
            ),
            n_points=(
                int(lidar_config["n_points"][0]),
                int(lidar_config["n_points"][1]),
            ),
        )
        ray_start_offset_m = float(lidar_config.get("ray_start_offset_m", 0.0))
        if ray_start_offset_m > 0:
            lidar_pattern._ray_starts = lidar_pattern._ray_dirs * ray_start_offset_m

        lidar = scene.add_sensor(
            gs.sensors.Lidar(
                entity_idx=robot.idx,
                link_idx_local=lidar_mount.link_idx_local,
                pos_offset=lidar_mount.pos_offset,
                euler_offset=lidar_mount.euler_offset,
                pattern=lidar_pattern,
                min_range=float(lidar_config["min_range"]),
                max_range=float(lidar_config["max_range"]),
                no_hit_value=float(
                    lidar_config.get("no_hit_value", lidar_config["max_range"])
                ),
                return_world_frame=False,
                draw_debug=draw_debug and bool(lidar_config.get("draw_debug", False)),
            )
        )
        lidar_max_range = float(lidar_config["max_range"])
        lidar_min_range = float(lidar_config["min_range"])
        lidar_no_hit_value = float(
            lidar_config.get("no_hit_value", lidar_config["max_range"])
        )
        lidar_bins_azimuth = int(lidar_config["bins_azimuth"])
        lidar_bins_elevation = int(lidar_config["bins_elevation"])

    return Stack(
        imu=imu,
        lidar=lidar,
        feet=feet,
        lidar_max_range=lidar_max_range,
        lidar_min_range=lidar_min_range,
        lidar_no_hit_value=lidar_no_hit_value,
        lidar_bins_azimuth=lidar_bins_azimuth,
        lidar_bins_elevation=lidar_bins_elevation,
    )


def read_stack(
    stack: Stack,
    *,
    lidar_noise_std: float = 0.0,
    num_envs: int | None = None,
) -> dict[str, torch.Tensor]:
    """Read and normalize the mounted sensor stack into tensor blocks.

    IMU readings (ang_vel, lin_acc, mag) are in the sensor/body frame as provided
    by Genesis. Use them for body-frame angular rate and linear acceleration.

    Args:
        stack (Stack): Mounted sensor stack returned by ``build_stack``.
        lidar_noise_std (float, optional): Standard deviation of Gaussian noise
            added to LiDAR distances before normalization.
        num_envs (int, optional): Number of environments. Required when all
            proprioceptive sensors (IMU, feet, LiDAR) are disabled.

    Returns:
        dict[str, torch.Tensor]: Sensor readings grouped by modality and ready
            to be assembled into observations.
    """
    num_envs_resolved: int
    device: torch.device
    dtype: torch.dtype

    if stack.imu is not None:
        imu_data = stack.imu.read()
        num_envs_resolved = imu_data.ang_vel.shape[0]
        device = imu_data.ang_vel.device
        dtype = imu_data.ang_vel.dtype
    elif len(stack.feet) > 0:
        foot_read = stack.feet[0].read()
        num_envs_resolved = foot_read.shape[0]
        device = foot_read.device
        dtype = foot_read.dtype
    elif stack.lidar is not None:
        lidar_data = stack.lidar.read()
        num_envs_resolved = lidar_data.distances.shape[0]
        device = lidar_data.distances.device
        dtype = lidar_data.distances.dtype
    elif num_envs is not None:
        num_envs_resolved = num_envs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
    else:
        raise ValueError(
            "Cannot infer num_envs: all sensors (IMU, feet, LiDAR) are disabled. "
            "Pass num_envs to read_stack."
        )

    if stack.imu is not None:
        imu_data = stack.imu.read()
        imu_gyro = imu_data.ang_vel
        imu_acc = imu_data.lin_acc
        imu_mag = getattr(imu_data, "mag", torch.zeros_like(imu_data.lin_acc))
    else:
        imu_gyro = torch.zeros((num_envs_resolved, 3), device=device, dtype=dtype)
        imu_acc = torch.zeros((num_envs_resolved, 3), device=device, dtype=dtype)
        imu_mag = torch.zeros((num_envs_resolved, 3), device=device, dtype=dtype)

    if len(stack.feet) > 0:
        foot_force = torch.stack([sensor.read() for sensor in stack.feet], dim=1)
        foot_force_norm = torch.linalg.vector_norm(foot_force, dim=-1)
    else:
        foot_force = torch.zeros((num_envs_resolved, 4, 3), device=device, dtype=dtype)
        foot_force_norm = torch.zeros(
            (num_envs_resolved, 4), device=device, dtype=dtype
        )

    if stack.lidar is not None:
        lidar_data = stack.lidar.read()
        lidar_distances = lidar_data.distances.clone()

        out_of_range = (lidar_distances < stack.lidar_min_range) | (
            lidar_distances > stack.lidar_max_range
        )
        lidar_distances = torch.where(
            out_of_range,
            torch.full_like(lidar_distances, stack.lidar_no_hit_value),
            lidar_distances,
        )

        if lidar_noise_std > 0.0:
            lidar_distances = lidar_distances + torch.randn_like(
                lidar_distances
            ) * float(lidar_noise_std)
            lidar_distances = torch.clamp(
                lidar_distances,
                min=stack.lidar_min_range,
                max=stack.lidar_max_range,
            )
        lidar_features = aggregate_lidar(
            lidar_distances,
            max_range=stack.lidar_max_range,
            bins_azimuth=stack.lidar_bins_azimuth,
            bins_elevation=stack.lidar_bins_elevation,
        )
    else:
        lidar_distances = torch.empty(
            (num_envs_resolved, 0), device=device, dtype=dtype
        )
        lidar_features = torch.empty((num_envs_resolved, 0), device=device, dtype=dtype)

    return {
        "imu_gyro": imu_gyro,
        "imu_acc": imu_acc,
        "imu_mag": imu_mag,
        "foot_force": foot_force,
        "foot_force_norm": foot_force_norm,
        "lidar_raw": lidar_distances,
        "lidar": lidar_features,
    }

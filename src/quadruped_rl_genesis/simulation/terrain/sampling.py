"""Terrain bounds, goal sampling, and height queries (NumPy and Torch)."""

from __future__ import annotations

import numpy as np
import torch

from quadruped_rl_genesis.simulation.terrain.fields import _sample_bilinear


def terrain_limits(
    size: tuple[float, float],
    margin: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return interior terrain bounds for a terrain centered at the origin.

    Args:
        size (tuple[float, float]): Terrain width and length in meters.
        margin (float, optional): Margin removed from each border.

    Returns:
        tuple[float, float, float, float]: ``(x_min, x_max, y_min, y_max)``
            bounds for valid interior positions.
    """
    width, length = size
    inset = max(float(margin), 0.0)
    half_width = max(width / 2 - inset, 0.0)
    half_length = max(length / 2 - inset, 0.0)

    return (-half_width, half_width, -half_length, half_length)


def sample_goal_xy(
    *,
    count: int,
    size: tuple[float, float],
    margin: float,
    min_distance: float,
    max_distance: float,
    spawn_xy: tuple[float, float] = (0.0, 0.0),
    rng: np.random.Generator | None = None,
    max_attempts: int = 256,
) -> np.ndarray:
    """Sample valid XY goal positions inside the terrain and away from spawn.

    Args:
        count (int): Number of goal positions to sample.
        size (tuple[float, float]): Terrain width and length in meters.
        margin (float): Border margin where goals cannot be placed.
        min_distance (float): Minimum distance from the spawn point.
        max_distance (float): Maximum distance from the spawn point.
        spawn_xy (tuple[float, float], optional): Spawn position in world XY.
        rng (np.random.Generator | None, optional): Random generator used for
            sampling.
        max_attempts (int, optional): Maximum rejection-sampling iterations
            before falling back to clipped polar samples.

    Returns:
        np.ndarray: ``[count, 2]`` array of sampled XY positions.
    """
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    generator = rng or np.random.default_rng()
    x_min, x_max, y_min, y_max = terrain_limits(size=size, margin=margin)
    spawn_x, spawn_y = spawn_xy
    min_distance = max(float(min_distance), 0.0)
    max_distance = max(float(max_distance), min_distance + 1e-6)

    samples = np.zeros((count, 2), dtype=np.float32)
    filled = 0
    attempts = 0

    while filled < count and attempts < max_attempts:
        attempts += 1
        remaining = count - filled
        candidate_x = generator.uniform(x_min, x_max, size=remaining).astype(np.float32)
        candidate_y = generator.uniform(y_min, y_max, size=remaining).astype(np.float32)
        candidate_xy = np.stack([candidate_x, candidate_y], axis=1)
        delta = candidate_xy - np.array([spawn_x, spawn_y], dtype=np.float32)
        distance = np.linalg.norm(delta, axis=1)
        keep = (distance >= min_distance) & (distance <= max_distance)
        kept = candidate_xy[keep]
        take = min(len(kept), remaining)

        if take > 0:
            samples[filled : filled + take] = kept[:take]
            filled += take

    if filled < count:
        remaining = count - filled
        angles = generator.uniform(0.0, 2.0 * np.pi, size=remaining).astype(np.float32)
        radii = generator.uniform(min_distance, max_distance, size=remaining).astype(
            np.float32
        )
        samples[filled:] = np.stack(
            [
                np.clip(spawn_x + radii * np.cos(angles), x_min, x_max),
                np.clip(spawn_y + radii * np.sin(angles), y_min, y_max),
            ],
            axis=1,
        ).astype(np.float32)

    return samples


def sample_height_numpy(
    heightmap: np.ndarray | None,
    positions_xy: np.ndarray,
    *,
    size: tuple[float, float],
) -> np.ndarray:
    """Sample terrain heights for XY positions using NumPy bilinear interpolation.

    Args:
        heightmap (np.ndarray | None): Terrain heightmap or ``None`` for flat
            ground.
        positions_xy (np.ndarray): XY positions in world coordinates.
        size (tuple[float, float]): Terrain width and length in meters.

    Returns:
        np.ndarray: Terrain height for each input position.
    """
    if heightmap is None:
        return np.zeros((positions_xy.shape[0],), dtype=np.float32)

    width, length = size
    width_res, length_res = heightmap.shape
    horizontal_scale = width / max(width_res - 1, 1)
    terrain_origin_x = -width / 2
    terrain_origin_y = -length / 2

    terrain_x = np.clip(positions_xy[:, 0] - terrain_origin_x, 0.0, width)
    terrain_y = np.clip(positions_xy[:, 1] - terrain_origin_y, 0.0, length)
    sample_row = np.clip(terrain_x / horizontal_scale, 0.0, width_res - 1.001)
    sample_col = np.clip(terrain_y / horizontal_scale, 0.0, length_res - 1.001)

    return _sample_bilinear(heightmap, sample_row, sample_col)


def sample_height_torch(
    heightmap: torch.Tensor | None,
    positions_xy: torch.Tensor,
    *,
    size: tuple[float, float],
) -> torch.Tensor:
    """Sample terrain heights for XY positions using Torch bilinear interpolation.

    Args:
        heightmap (torch.Tensor | None): Terrain heightmap or ``None`` for flat
            ground.
        positions_xy (torch.Tensor): XY positions in world coordinates.
        size (tuple[float, float]): Terrain width and length in meters.

    Returns:
        torch.Tensor: Terrain height for each input position.
    """
    if heightmap is None:
        return torch.zeros(
            (positions_xy.shape[0],),
            device=positions_xy.device,
            dtype=positions_xy.dtype,
        )

    width, length = size
    width_res, length_res = heightmap.shape
    horizontal_scale = width / max(width_res - 1, 1)
    terrain_origin_x = -width / 2
    terrain_origin_y = -length / 2

    terrain_x = torch.clamp(positions_xy[:, 0] - terrain_origin_x, 0.0, width)
    terrain_y = torch.clamp(positions_xy[:, 1] - terrain_origin_y, 0.0, length)
    sample_row = torch.clamp(terrain_x / horizontal_scale, 0.0, width_res - 1.001)
    sample_col = torch.clamp(terrain_y / horizontal_scale, 0.0, length_res - 1.001)

    r0 = sample_row.floor().long()
    c0 = sample_col.floor().long()
    r1 = torch.clamp(r0 + 1, max=width_res - 1)
    c1 = torch.clamp(c0 + 1, max=length_res - 1)

    wr = sample_row - r0.to(sample_row.dtype)
    wc = sample_col - c0.to(sample_col.dtype)

    h00 = heightmap[r0, c0]
    h10 = heightmap[r1, c0]
    h01 = heightmap[r0, c1]
    h11 = heightmap[r1, c1]

    near = h00 * (1.0 - wc) + h01 * wc
    far = h10 * (1.0 - wc) + h11 * wc

    return near * (1.0 - wr) + far * wr


def estimate_heightmap_slopes_torch(
    heightmap: torch.Tensor | None,
    positions_xy: torch.Tensor,
    *,
    size: tuple[float, float],
    sample_distance_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate terrain slope, normal, and horizontal gradient at batch positions.

    Args:
        heightmap (torch.Tensor | None): Optional height grid; ``None`` yields flat
            terrain (zero slope).
        positions_xy (torch.Tensor): World XY samples, shape ``[N, 2]``.
        size (tuple[float, float]): Terrain extent in meters (width, length).
        sample_distance_m (float): Finite-difference offset for central differences.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(slope_rad, normal, gradient)``
            with shapes ``[N]``, ``[N, 3]``, ``[N, 2]``.
    """
    batch_size = positions_xy.shape[0]
    if heightmap is None:
        slope_rad = torch.zeros(
            (batch_size,),
            device=positions_xy.device,
            dtype=positions_xy.dtype,
        )
        normal = torch.zeros(
            (batch_size, 3),
            device=positions_xy.device,
            dtype=positions_xy.dtype,
        )
        normal[:, 2] = 1.0
        gradient = torch.zeros(
            (batch_size, 2),
            device=positions_xy.device,
            dtype=positions_xy.dtype,
        )
        return slope_rad, normal, gradient

    spacing = max(float(sample_distance_m), 1.0e-3)
    offsets = torch.tensor(
        [
            [spacing, 0.0],
            [-spacing, 0.0],
            [0.0, spacing],
            [0.0, -spacing],
        ],
        device=positions_xy.device,
        dtype=positions_xy.dtype,
    )
    h_x_pos, h_x_neg, h_y_pos, h_y_neg = [
        sample_height_torch(heightmap, positions_xy + offset.unsqueeze(0), size=size)
        for offset in offsets
    ]
    dz_dx = (h_x_pos - h_x_neg) / (2.0 * spacing)
    dz_dy = (h_y_pos - h_y_neg) / (2.0 * spacing)
    gradient = torch.stack((dz_dx, dz_dy), dim=1)
    slope_rad = torch.atan(torch.linalg.vector_norm(gradient, dim=1))

    normal = torch.stack((-dz_dx, -dz_dy, torch.ones_like(dz_dx)), dim=1)
    normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True).clamp(
        min=1.0e-6
    )

    return slope_rad, normal, gradient

"""Small tensor helpers shared by the Go2 navigation task."""

from __future__ import annotations

import torch


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles into the ``[-pi, pi]`` interval.

    Args:
        angle (torch.Tensor): Input angle tensor in radians.

    Returns:
        torch.Tensor: Wrapped angle tensor in radians.
    """
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def as_matrix_tensor(
    values: torch.Tensor,
    *,
    num_envs: int,
    width: int,
) -> torch.Tensor:
    """Normalize a tensor into ``[num_envs, width]``.

    Args:
        values (torch.Tensor): Raw tensor returned by Genesis.
        num_envs (int): Number of parallel environments.
        width (int): Expected second dimension width.

    Returns:
        torch.Tensor: Tensor shaped as ``[num_envs, width]``.
    """
    if values.ndim == 1:
        return values.unsqueeze(0)
    if values.ndim == 2:
        if values.shape[0] == num_envs and values.shape[1] == width:
            return values
        if values.shape[0] == width and values.shape[1] == num_envs:
            return values.transpose(0, 1)
        if values.shape[1] == width:
            return values[:num_envs]
    return values.reshape(num_envs, width)

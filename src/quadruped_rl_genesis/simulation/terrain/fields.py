"""Low-level NumPy heightfield helpers (blur, normalization, bilinear sampling)."""

from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


def _box_blur(field: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Blur a 2D field along one axis using a reflected box filter.

    Args:
        field (np.ndarray): Input 2D field.
        radius (int): Blur radius in pixels.
        axis (int): Axis along which the blur is applied.

    Returns:
        np.ndarray: Blurred field as ``float32``.
    """
    if radius <= 0:
        return field.astype(np.float32, copy=False)

    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(field, pad_width, mode="reflect")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float64)
    cumulative_pad = [(0, 0), (0, 0)]
    cumulative_pad[axis] = (1, 0)
    cumulative = np.pad(cumulative, cumulative_pad, mode="constant")

    slicer_start = [slice(None)] * field.ndim
    slicer_end = [slice(None)] * field.ndim
    slicer_start[axis] = slice(2 * radius + 1, None)
    slicer_end[axis] = slice(None, -(2 * radius + 1))
    blurred = cumulative[tuple(slicer_start)] - cumulative[tuple(slicer_end)]

    return (blurred / (2 * radius + 1)).astype(np.float32)


def _smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a field with a Gaussian filter or a box-blur fallback.

    Args:
        field (np.ndarray): Input 2D field.
        sigma (float): Smoothing strength.

    Returns:
        np.ndarray: Smoothed field as ``float32``.
    """
    sigma = float(max(0.0, sigma))

    if sigma <= 1e-6:
        return field.astype(np.float32, copy=False)

    if ndimage is not None:
        return ndimage.gaussian_filter(field, sigma=sigma, mode="reflect").astype(
            np.float32
        )

    radius = max(1, int(np.ceil(1.5 * sigma)))
    blurred = _box_blur(field, radius, axis=1)

    return _box_blur(blurred, radius, axis=0)


def _normalize_centered(field: np.ndarray) -> np.ndarray:
    """Center and scale a field to approximately ``[-1, 1]``.

    Args:
        field (np.ndarray): Input field.

    Returns:
        np.ndarray: Centered and normalized field as ``float32``.
    """
    centered = field - np.mean(field)
    max_abs = np.max(np.abs(centered))

    if max_abs < 1e-8:
        return np.zeros_like(field, dtype=np.float32)

    return (centered / max_abs).astype(np.float32)


def _normalize_unit(field: np.ndarray) -> np.ndarray:
    """Scale a field into the ``[0, 1]`` range.

    Args:
        field (np.ndarray): Input field.

    Returns:
        np.ndarray: Unit-normalized field as ``float32``.
    """
    field_min = float(np.min(field))
    field_max = float(np.max(field))

    if field_max - field_min < 1e-8:
        return np.zeros_like(field, dtype=np.float32)

    return ((field - field_min) / (field_max - field_min)).astype(np.float32)


def _smoothstep(values: np.ndarray) -> np.ndarray:
    """Apply a cubic smoothstep after clipping inputs to ``[0, 1]``.

    Args:
        values (np.ndarray): Raw interpolation parameters.

    Returns:
        np.ndarray: ``3t^2 - 2t^3`` element-wise as ``float32``.
    """
    clipped = np.clip(values, 0.0, 1.0).astype(np.float32)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _sample_bilinear(
    field: np.ndarray,
    sample_row: np.ndarray,
    sample_col: np.ndarray,
) -> np.ndarray:
    """Sample a 2D field with bilinear interpolation in NumPy.

    Args:
        field (np.ndarray): Input 2D field.
        sample_row (np.ndarray): Row coordinates to sample.
        sample_col (np.ndarray): Column coordinates to sample.

    Returns:
        np.ndarray: Interpolated values as ``float32``.
    """
    rows, cols = field.shape
    r0 = np.floor(sample_row).astype(np.int32)
    c0 = np.floor(sample_col).astype(np.int32)
    r0 = np.clip(r0, 0, rows - 1)
    c0 = np.clip(c0, 0, cols - 1)
    r1 = np.clip(r0 + 1, 0, rows - 1)
    c1 = np.clip(c0 + 1, 0, cols - 1)

    wr = sample_row - r0
    wc = sample_col - c0

    near = field[r0, c0] * (1.0 - wc) + field[r0, c1] * wc
    far = field[r1, c0] * (1.0 - wc) + field[r1, c1] * wc

    return (near * (1.0 - wr) + far * wr).astype(np.float32)

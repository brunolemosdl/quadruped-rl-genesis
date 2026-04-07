"""Genesis backend initialization and one-shot scene setup."""

from __future__ import annotations

import os
from pathlib import Path

from quadruped_rl_genesis.services.logger import get_logger

LOGGER = get_logger(__name__)

_GENESIS_INITIALIZED = False


def resolve_runtime_device(requested_device: str) -> str:
    """Resolve the requested runtime device into an explicit backend string.

    Args:
        requested_device (str): Requested device, typically ``"auto"``,
            ``"cpu"``, or ``"cuda"``.

    Returns:
        str: Explicit backend string. ``"auto"`` becomes ``"cuda"`` when Torch
            reports GPU availability, otherwise ``"cpu"``.
    """
    normalized = requested_device.lower()

    if normalized != "auto":
        return requested_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ModuleNotFoundError:
        pass

    return "cpu"


def initialize_genesis(seed: int, device: str) -> str:
    """Initialize Genesis once per process and return the resolved device.

    Args:
        seed (int): Seed forwarded to Genesis initialization.
        device (str): Requested runtime device.

    Returns:
        str: Explicit backend string used by the current process.
    """
    global _GENESIS_INITIALIZED

    runtime_device = resolve_runtime_device(device)

    if _GENESIS_INITIALIZED:
        return runtime_device

    import genesis as gs

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    _ensure_writable_cache_dirs()
    backend = (
        gs.constants.backend.gpu
        if runtime_device.startswith("cuda")
        else gs.constants.backend.cpu
    )

    try:
        gs.init(seed=seed, backend=backend, logging_level="Error")
    except Exception as exc:
        auto_requested = device.lower() == "auto"
        if not auto_requested or not runtime_device.startswith("cuda"):
            raise

        LOGGER.warning(
            "Genesis CUDA init failed under device=auto; falling back to CPU. "
            "Original error: %s",
            exc,
        )
        runtime_device = "cpu"
        gs.init(
            seed=seed,
            backend=gs.constants.backend.cpu,
            logging_level="Error",
        )

    _GENESIS_INITIALIZED = True
    LOGGER.info("Genesis initialized with backend=%s seed=%s", runtime_device, seed)

    return runtime_device


def _ensure_writable_cache_dirs() -> None:
    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.append(cwd / ".cache")
    project_root = Path(__file__).resolve().parents[3]
    candidates.append(project_root / ".cache")

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write_test"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
        except Exception:
            continue

        os.environ.setdefault("XDG_CACHE_HOME", str(base))
        os.environ.setdefault("QUADRANTS_CACHE_DIR", str(base / "quadrants"))
        os.environ.setdefault(
            "QUADRANTS_QDCACHE_DIR", str(base / "quadrants" / "qdcache")
        )
        return


def shutdown_genesis() -> None:
    """Shut down the active Genesis runtime if it has been initialized.

    The internal initialization flag is reset even if ``gs.destroy()`` raises.
    """
    global _GENESIS_INITIALIZED

    if not _GENESIS_INITIALIZED:
        return

    import genesis as gs

    try:
        gs.destroy()
    finally:
        _GENESIS_INITIALIZED = False

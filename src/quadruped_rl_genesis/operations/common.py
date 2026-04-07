"""Shared helpers for headless runtime configuration and output paths."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

from quadruped_rl_genesis.settings import AppSettings


def configure_headless_runtime_env(project_root: Path) -> None:
    """Populate environment variables required for headless Genesis execution.

    Args:
        project_root (Path): Repository root used to inspect likely virtual
            environment locations.
    """
    if sys.platform.startswith("linux"):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_DEVICE_ID", "0")
        os.environ.setdefault("MUJOCO_GL", "egl")

        vendor_file = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
        if vendor_file.exists():
            os.environ.setdefault(
                "__EGL_VENDOR_LIBRARY_FILENAMES",
                str(vendor_file),
            )

    lib_dirs: list[str] = []
    candidate_roots: list[Path] = []

    if os.environ.get("VIRTUAL_ENV"):
        candidate_roots.append(Path(os.environ["VIRTUAL_ENV"]))

    candidate_roots.append(project_root / ".venv")

    for root in candidate_roots:
        if not root.exists():
            continue

        lib_dirs.extend(
            str(path)
            for path in root.glob("lib/python*/site-packages/nvidia/*/lib")
            if path.is_dir()
        )

    if lib_dirs:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        joined = ":".join(dict.fromkeys(lib_dirs))

        os.environ["LD_LIBRARY_PATH"] = f"{joined}:{current}" if current else joined


def derive_output_settings(settings: AppSettings, output_root: Path) -> AppSettings:
    """Create a settings object whose artifacts live under a benchmark folder.

    Args:
        settings (AppSettings): Base application settings.
        output_root (Path): Benchmark output root selected by the caller.

    Returns:
        AppSettings: Copy of ``settings`` with artifact and Optuna paths scoped
            to the benchmark output directory.
    """
    output_root = output_root.resolve()
    artifacts_root = output_root / "artifacts"
    optuna_storage = f"sqlite:///{(artifacts_root / 'optuna' / 'studies.db').resolve()}"

    return replace(
        settings,
        artifacts_root=artifacts_root,
        optuna_storage=optuna_storage,
    )

"""Project setup: assets, dependencies, and environment preparation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from quadruped_rl_genesis.operations.common import configure_headless_runtime_env
from quadruped_rl_genesis.services.logger import get_logger
from quadruped_rl_genesis.settings import AppSettings

LOGGER = get_logger(__name__)


def run_setup(
    settings: AppSettings,
    *,
    python_bin: str | None = None,
    venv_dir: str | None = None,
) -> dict[str, str]:
    """Create or update the local virtual environment and install dependencies.

    Args:
        settings (AppSettings): Global application settings.
        python_bin (str | None, optional): Python executable used to create the
            virtual environment. Falls back to ``python3`` or ``python`` from
            ``PATH``.
        venv_dir (str | None, optional): Virtual environment directory. If
            omitted, ``<project_root>/.venv`` is used.

    Returns:
        dict[str, str]: Summary containing the chosen Python executable,
            virtual-environment directory, and project root.

    Raises:
        RuntimeError: If no Python interpreter is found in ``PATH``.
    """
    project_root = settings.project_root
    resolved_python = python_bin or shutil.which("python3") or shutil.which("python")

    if resolved_python is None:
        raise RuntimeError("Python was not found in PATH.")

    resolved_venv = Path(venv_dir) if venv_dir else project_root / ".venv"
    resolved_venv = resolved_venv.resolve()
    venv_python = resolved_venv / "bin" / "python"
    venv_poetry = resolved_venv / "bin" / "poetry"

    if not resolved_venv.exists():
        LOGGER.info("Creating virtual environment at %s", resolved_venv)
        subprocess.run(
            [resolved_python, "-m", "venv", str(resolved_venv)],
            check=True,
            cwd=project_root,
        )

    subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ],
        check=True,
        cwd=project_root,
    )

    poetry_command = None
    if venv_poetry.exists():
        poetry_command = [str(venv_python), "-m", "poetry"]
    else:
        global_poetry = shutil.which("poetry")
        if global_poetry:
            poetry_command = [global_poetry]

    if poetry_command:
        LOGGER.info("Installing project dependencies with Poetry.")
        env = dict(os.environ)
        env["POETRY_VIRTUALENVS_CREATE"] = "false"
        env["VIRTUAL_ENV"] = str(resolved_venv)
        env["PATH"] = f"{resolved_venv / 'bin'}:{os.environ.get('PATH', '')}"

        subprocess.run(
            [*poetry_command, "install", "--extras", "dev"],
            check=True,
            cwd=project_root,
            env=env,
        )
    else:
        LOGGER.info(
            "Poetry was not found; installing project in editable mode with pip extras."
        )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", f"{project_root}[dev]"],
            check=True,
            cwd=project_root,
        )

    os.environ["VIRTUAL_ENV"] = str(resolved_venv)
    os.environ["PATH"] = f"{resolved_venv / 'bin'}:{os.environ.get('PATH', '')}"
    configure_headless_runtime_env(project_root)

    summary = {
        "python": str(venv_python),
        "venv_dir": str(resolved_venv),
        "project_root": str(project_root),
    }

    LOGGER.info(
        "Environment configured | python=%s venv=%s", venv_python, resolved_venv
    )

    return summary

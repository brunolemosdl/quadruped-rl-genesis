"""Environment and dependency checks for the CLI ``check`` command."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from typing import Any

from quadruped_rl_genesis.cli import build_argument_parser
from quadruped_rl_genesis.operations.common import configure_headless_runtime_env
from quadruped_rl_genesis.services.logger import get_logger
from quadruped_rl_genesis.services.platform import collect_runtime_context
from quadruped_rl_genesis.settings import AppSettings

LOGGER = get_logger(__name__)


def _module_version(module_name: str) -> str | None:
    """Import a module and return its ``__version__`` attribute when present.

    Args:
        module_name (str): Import path of the module to inspect.

    Returns:
        str | None: Version string if import succeeds and the attribute exists,
            otherwise ``None``.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    return getattr(module, "__version__", None)


def run_check(settings: AppSettings) -> dict[str, Any]:
    """Validate the local runtime for Quadruped RL Genesis workflows.

    Args:
        settings (AppSettings): Global application settings.

    Returns:
        dict[str, Any]: Summary of the detected runtime, installed libraries,
            and basic CLI health checks.
    """
    configure_headless_runtime_env(settings.project_root)

    build_argument_parser()

    gpu_summary = None

    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            cwd=settings.project_root,
        )

        if result.returncode == 0:
            gpu_summary = result.stdout.strip()

    summary = {
        "project_root": str(settings.project_root),
        "python_version": sys.version.split()[0],
        "artifacts_root": str(settings.artifacts_root),
        "optuna_storage": settings.optuna_storage,
        "runtime_context": collect_runtime_context(),
        "torch_version": _module_version("torch"),
        "genesis_version": _module_version("genesis"),
        "cli_status": "ok",
        "nvidia_smi": gpu_summary,
    }

    LOGGER.info("Runtime check passed | artifacts_root=%s", settings.artifacts_root)

    return summary

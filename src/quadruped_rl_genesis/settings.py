"""Application settings loaded from environment and optional ``.env`` files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*_args, **_kwargs) -> bool:
        """Fallback no-op ``load_dotenv`` used when python-dotenv is unavailable.

        Returns:
            bool: Always ``False`` to indicate that no environment file was
                loaded.
        """
        return False


@dataclass(frozen=True)
class AppSettings:
    """Resolved application settings shared by all workflows.

    Attributes:
        project_root (Path): Repository root directory.
        configs_root (Path): Root for experiment and algorithm configs.
        artifacts_root (Path): Root for runs, models, and outputs.
        default_experiment (str): Default experiment profile name.
        default_hyperparams_source (str): Default hyperparameter source.
        seed (int): Global random seed.
        device (str): Default runtime device for both Genesis and algorithm.
        genesis_device (str): Device for Genesis simulation (defaults to device).
        algorithm_device (str): Device for SB3/RL algorithm (defaults to device).
        log_level (str): Logging verbosity level.
        optuna_storage (str): Optuna storage URL for hyperparameter tuning.
    """

    project_root: Path
    configs_root: Path
    artifacts_root: Path
    default_experiment: str
    default_hyperparams_source: str
    seed: int
    device: str
    genesis_device: str
    algorithm_device: str
    log_level: str
    optuna_storage: str


def load_app_settings(project_root: Path | None = None) -> AppSettings:
    """Load application settings from the repository root and environment.

    Relative configuration and artifact paths are resolved against the detected
    project root so the rest of the codebase can work with absolute paths.

    Args:
        project_root (Path | None, optional): Explicit repository root. If
            ``None``, it is inferred from this module location.

    Returns:
        AppSettings: Fully resolved settings object used by CLI commands and
            workflows.
    """
    root = project_root or Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)

    configs_root = Path(os.getenv("QUADRUPED_RL_GENESIS_CONFIGS_ROOT", "configs"))
    artifacts_root = Path(os.getenv("QUADRUPED_RL_GENESIS_ARTIFACTS_ROOT", "artifacts"))

    if not configs_root.is_absolute():
        configs_root = root / configs_root

    if not artifacts_root.is_absolute():
        artifacts_root = root / artifacts_root

    optuna_storage = os.getenv(
        "QUADRUPED_RL_GENESIS_OPTUNA_STORAGE", "sqlite:///artifacts/optuna/studies.db"
    )
    if optuna_storage.startswith("sqlite:///"):
        storage_path = Path(optuna_storage.replace("sqlite:///", "", 1))

        if not storage_path.is_absolute():
            storage_path = root / storage_path

        optuna_storage = f"sqlite:///{storage_path}"

    return AppSettings(
        project_root=root,
        configs_root=configs_root,
        artifacts_root=artifacts_root,
        default_experiment=os.getenv(
            "QUADRUPED_RL_GENESIS_DEFAULT_EXPERIMENT", "default"
        ),
        default_hyperparams_source=os.getenv(
            "QUADRUPED_RL_GENESIS_DEFAULT_HPARAMS_SOURCE", "default"
        ),
        seed=int(os.getenv("QUADRUPED_RL_GENESIS_SEED", "42")),
        device=os.getenv("QUADRUPED_RL_GENESIS_DEVICE", "auto"),
        genesis_device=os.getenv(
            "QUADRUPED_RL_GENESIS_GENESIS_DEVICE",
            os.getenv("QUADRUPED_RL_GENESIS_DEVICE", "auto"),
        ),
        algorithm_device=os.getenv(
            "QUADRUPED_RL_GENESIS_ALGORITHM_DEVICE",
            os.getenv("QUADRUPED_RL_GENESIS_DEVICE", "auto"),
        ),
        log_level=os.getenv("QUADRUPED_RL_GENESIS_LOG_LEVEL", "INFO"),
        optuna_storage=optuna_storage,
    )

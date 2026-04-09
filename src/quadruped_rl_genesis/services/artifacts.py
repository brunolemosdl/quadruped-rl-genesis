"""Run artifact paths and artifact-tree resolution helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import write_json, write_yaml

_RUN_ID_PATTERN = re.compile(
    r"^(?P<timestamp>\d{8}-\d{6})_seed(?P<seed>\d+)_(?P<hyperparams_source>.+)$"
)


@dataclass(frozen=True)
class RunArtifacts:
    """Filesystem paths allocated for a single experiment run.

    Attributes:
        run_id (str): Unique run identifier (timestamp, seed, hyperparams).
        run_root (Path): Root directory for the run.
        config_dir (Path): Directory for resolved config files.
        logs_dir (Path): Directory for training logs.
        checkpoints_dir (Path): Directory for training checkpoints.
        models_dir (Path): Directory for saved models.
        eval_dir (Path): Directory for evaluation outputs.
        videos_dir (Path): Directory for recorded videos.
    """

    run_id: str
    run_root: Path
    config_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    models_dir: Path
    eval_dir: Path
    videos_dir: Path


class ArtifactStore:
    """Persist and resolve experiment artifacts in a predictable directory tree.

    Manages run creation, model lookup, config persistence, and artifact
    resolution under the configured artifacts root.
    """

    def __init__(self, settings: AppSettings):
        """Create an artifact helper bound to a resolved settings object.

        Args:
            settings (AppSettings): Global application settings with the
                artifact root already resolved.
        """
        self.settings = settings

    @staticmethod
    def parse_run_id(run_id: str) -> dict[str, str | int] | None:
        """Parse a canonical run identifier.

        Args:
            run_id (str): Run identifier such as
                ``20260331-142000_seed42_utd_gs2``.

        Returns:
            dict[str, str | int] | None: Parsed fields when the identifier
                matches the canonical format, otherwise ``None``.
        """
        match = _RUN_ID_PATTERN.match(run_id)
        if match is None:
            return None

        return {
            "timestamp": match.group("timestamp"),
            "seed": int(match.group("seed")),
            "hyperparams_source": match.group("hyperparams_source"),
        }

    @staticmethod
    def sanitize_label(label: str | None) -> str:
        """Return a filesystem-safe label fragment.

        Args:
            label (str | None): Arbitrary label text.

        Returns:
            str: Sanitized label containing alphanumerics, ``-``, and ``_``.
        """
        if label is None:
            return ""

        sanitized = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in str(label)
        ).strip("_")

        return sanitized

    @staticmethod
    def create_execution_id(seed: int, label: str | None = None) -> str:
        """Create a unique identifier for an ad-hoc evaluation/visualization.

        Args:
            seed (int): Evaluation or visualization seed.
            label (str | None, optional): Optional descriptive suffix.

        Returns:
            str: Timestamped execution identifier.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        safe_label = ArtifactStore.sanitize_label(label)
        if safe_label:
            return f"{timestamp}_seed{seed}_{safe_label}"

        return f"{timestamp}_seed{seed}"

    def create_run(
        self,
        experiment_name: str,
        algorithm: str,
        seed: int,
        hyperparams_source: str,
    ) -> RunArtifacts:
        """Create and return the directory structure for a new training run.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Training algorithm name.
            seed (int): Training seed used in the run identifier.
            hyperparams_source (str): Hyperparameter source label used in the
                run identifier.

        Returns:
            RunArtifacts: All canonical directories allocated for the run.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}_seed{seed}_{hyperparams_source}"
        run_root = (
            self.settings.artifacts_root
            / "experiments"
            / experiment_name
            / algorithm
            / run_id
        )

        config_dir = run_root / "config"
        logs_dir = run_root / "logs"
        checkpoints_dir = run_root / "checkpoints"
        models_dir = run_root / "models"
        eval_dir = run_root / "eval"
        videos_dir = run_root / "videos"

        for directory in (
            config_dir,
            logs_dir,
            checkpoints_dir,
            models_dir,
            eval_dir,
            videos_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        return RunArtifacts(
            run_id=run_id,
            run_root=run_root,
            config_dir=config_dir,
            logs_dir=logs_dir,
            checkpoints_dir=checkpoints_dir,
            models_dir=models_dir,
            eval_dir=eval_dir,
            videos_dir=videos_dir,
        )

    def get_runs_root(self, experiment_name: str, algorithm: str) -> Path:
        """Return the root directory containing training runs.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.

        Returns:
            Path: Canonical runs directory.
        """
        return (
            self.settings.artifacts_root / "experiments" / experiment_name / algorithm
        )

    def iter_run_roots(
        self,
        experiment_name: str,
        algorithm: str,
    ) -> list[Path]:
        """List existing run directories sorted from newest to oldest.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.

        Returns:
            list[Path]: Existing run roots ordered by descending run id.
        """
        runs_root = self.get_runs_root(experiment_name, algorithm)
        if not runs_root.exists():
            return []

        return sorted(
            (
                path
                for path in runs_root.iterdir()
                if path.is_dir() and self.parse_run_id(path.name) is not None
            ),
            key=lambda path: path.name,
            reverse=True,
        )

    def get_run_root(
        self,
        experiment_name: str,
        algorithm: str,
        run_id: str,
    ) -> Path | None:
        """Resolve a run directory by identifier.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            run_id (str): Exact run identifier.

        Returns:
            Path | None: Matching run root when it exists.
        """
        run_root = self.get_runs_root(experiment_name, algorithm) / run_id
        if run_root.is_dir() and self.parse_run_id(run_root.name) is not None:
            return run_root

        return None

    def is_completed_training_run(self, run_root: Path) -> bool:
        """Check whether a run contains final training outputs.

        Args:
            run_root (Path): Candidate run directory.

        Returns:
            bool: ``True`` when the run has a training summary or final model.
        """
        return (run_root / "eval" / "training_summary.json").exists() or (
            run_root / "models" / "final_model.zip"
        ).exists()

    def find_latest_run(
        self,
        experiment_name: str,
        algorithm: str,
        *,
        hyperparams_source: str | None = None,
        train_seed: int | None = None,
        completed_only: bool = False,
    ) -> Path | None:
        """Find the newest run matching the requested selectors.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            hyperparams_source (str | None, optional): Hyperparameter source to
                match exactly.
            train_seed (int | None, optional): Training seed to match exactly.
            completed_only (bool, optional): Require a finished training run.

        Returns:
            Path | None: Matching run root when found.
        """
        for run_root in self.iter_run_roots(experiment_name, algorithm):
            parsed = self.parse_run_id(run_root.name)
            if parsed is None:
                continue

            if hyperparams_source is not None and (
                parsed["hyperparams_source"] != hyperparams_source
            ):
                continue
            if train_seed is not None and parsed["seed"] != int(train_seed):
                continue
            if completed_only and not self.is_completed_training_run(run_root):
                continue

            return run_root

        return None

    def get_model_path_for_run(
        self, run_root: Path, model_kind: str = "best"
    ) -> Path | None:
        """Resolve the preferred model artifact inside one run.

        Args:
            run_root (Path): Training run root.
            model_kind (str, optional): Requested model kind, usually
                ``"best"`` or ``"final"``.

        Returns:
            Path | None: Existing model artifact when found.
        """
        best_model_path = run_root / "models" / "best_model" / "best_model.zip"
        final_model_path = run_root / "models" / "final_model.zip"

        if model_kind == "best" and best_model_path.exists():
            return best_model_path
        if final_model_path.exists():
            return final_model_path
        if best_model_path.exists():
            return best_model_path

        return None

    def save_resolved_config(
        self, run_artifacts: RunArtifacts, resolved_config: dict
    ) -> Path:
        """Persist the fully resolved configuration for a run.

        Args:
            run_artifacts (RunArtifacts): Target run directories.
            resolved_config (dict): Combined experiment/algorithm/runtime
                configuration payload.

        Returns:
            Path: Path of the written YAML file.
        """
        output_path = run_artifacts.config_dir / "resolved_config.yaml"
        write_yaml(output_path, resolved_config)

        return output_path

    def save_runtime_metadata(self, run_artifacts: RunArtifacts, payload: dict) -> Path:
        """Persist runtime metadata captured when a workflow starts.

        Args:
            run_artifacts (RunArtifacts): Target run directories.
            payload (dict): Runtime metadata to serialize as JSON.

        Returns:
            Path: Path of the written metadata file.
        """
        output_path = run_artifacts.config_dir / "runtime_metadata.json"
        write_json(output_path, payload)

        return output_path

    def create_ad_hoc_execution_dir(
        self,
        kind: str,
        experiment_name: str,
        algorithm: str,
        seed: int,
        label: str | None = None,
    ) -> Path:
        """Create a directory for ad-hoc evaluation or visualization outputs.

        Args:
            kind (str): Ad-hoc artifact family, such as ``"evaluate"`` or
                ``"visualize"``.
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            seed (int): Execution seed.
            label (str | None, optional): Optional label suffix.

        Returns:
            Path: Created directory root.
        """
        execution_id = self.create_execution_id(seed, label)
        directory = (
            self.settings.artifacts_root
            / "ad_hoc"
            / kind
            / experiment_name
            / algorithm
            / execution_id
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def get_optuna_dir(self, experiment_name: str, algorithm: str) -> Path:
        """Return the root directory for Optuna artifacts of an algorithm.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.

        Returns:
            Path: Existing directory reserved for Optuna outputs.
        """
        directory = (
            self.settings.artifacts_root / "optuna" / experiment_name / algorithm
        )
        directory.mkdir(parents=True, exist_ok=True)

        return directory

    def get_optuna_best_overrides_path(
        self, experiment_name: str, algorithm: str
    ) -> Path:
        """Return the canonical YAML path for the best Optuna overrides.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.

        Returns:
            Path: ``best_overrides.yaml`` under the Optuna experiment directory.
        """
        return self.get_optuna_dir(experiment_name, algorithm) / "best_overrides.yaml"

    def get_optuna_best_resolved_config_path(
        self, experiment_name: str, algorithm: str
    ) -> Path:
        """Return the canonical YAML path for the best resolved Optuna config.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.

        Returns:
            Path: ``best_resolved_config.yaml`` under the Optuna experiment directory.
        """
        return (
            self.get_optuna_dir(experiment_name, algorithm)
            / "best_resolved_config.yaml"
        )

    def get_optuna_best_trial_summary_path(
        self, experiment_name: str, algorithm: str
    ) -> Path:
        """Return the canonical JSON path for the best-trial summary.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.

        Returns:
            Path: ``best_trial_summary.json`` under the Optuna experiment directory.
        """
        return (
            self.get_optuna_dir(experiment_name, algorithm) / "best_trial_summary.json"
        )

    def get_optuna_study_summary_path(
        self, experiment_name: str, algorithm: str
    ) -> Path:
        """Return the canonical JSON path for the top-level Optuna summary.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.

        Returns:
            Path: ``study_summary.json`` under the Optuna experiment directory.
        """
        return self.get_optuna_dir(experiment_name, algorithm) / "study_summary.json"

    def get_optuna_phases_root(self, experiment_name: str, algorithm: str) -> Path:
        """Return the root directory used to store Optuna phases.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.

        Returns:
            Path: ``phases/`` directory (created if missing).
        """
        directory = self.get_optuna_dir(experiment_name, algorithm) / "phases"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def get_optuna_phase_dir(
        self,
        experiment_name: str,
        algorithm: str,
        phase_name: str,
    ) -> Path:
        """Return the directory used to store one Optuna phase.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            phase_name (str): Phase label (for example ``locomotion``).

        Returns:
            Path: Phase directory (created if missing).
        """
        directory = self.get_optuna_phases_root(experiment_name, algorithm) / phase_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def get_optuna_phase_current_trial_path(
        self,
        experiment_name: str,
        algorithm: str,
        phase_name: str,
    ) -> Path:
        """Return the JSON path for the current-trial status of one phase.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            phase_name (str): Phase label.

        Returns:
            Path: ``current_trial.json`` inside the phase directory.
        """
        return (
            self.get_optuna_phase_dir(experiment_name, algorithm, phase_name)
            / "current_trial.json"
        )

    def get_optuna_phase_study_summary_path(
        self,
        experiment_name: str,
        algorithm: str,
        phase_name: str,
    ) -> Path:
        """Return the JSON path for one phase study summary.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            phase_name (str): Phase label.

        Returns:
            Path: ``study_summary.json`` inside the phase directory.
        """
        return (
            self.get_optuna_phase_dir(experiment_name, algorithm, phase_name)
            / "study_summary.json"
        )

    def get_optuna_phase_trials_dir(
        self,
        experiment_name: str,
        algorithm: str,
        phase_name: str,
    ) -> Path:
        """Return the directory used to store individual Optuna trials.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            phase_name (str): Phase label.

        Returns:
            Path: ``trials/`` subdirectory (created if missing).
        """
        directory = (
            self.get_optuna_phase_dir(experiment_name, algorithm, phase_name) / "trials"
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def get_optuna_trial_dir(
        self,
        experiment_name: str,
        algorithm: str,
        phase_name: str,
        trial_number: int,
    ) -> Path:
        """Return the directory reserved for a specific Optuna trial.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            phase_name (str): Optuna phase name.
            trial_number (int): Optuna trial number.

        Returns:
            Path: Existing directory for the requested trial.
        """
        directory = self.get_optuna_phase_trials_dir(
            experiment_name, algorithm, phase_name
        ) / (f"trial_{trial_number:04d}")
        directory.mkdir(parents=True, exist_ok=True)

        return directory

    def get_optuna_trial_overrides_path(
        self,
        experiment_name: str,
        algorithm: str,
        phase_name: str,
        trial_number: int,
    ) -> Path:
        """Return the YAML path for one trial overrides snapshot.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            phase_name (str): Phase label.
            trial_number (int): Optuna trial index.

        Returns:
            Path: ``trial_overrides.yaml`` inside the trial directory.
        """
        return (
            self.get_optuna_trial_dir(
                experiment_name, algorithm, phase_name, trial_number
            )
            / "trial_overrides.yaml"
        )

    def get_optuna_trial_resolved_config_path(
        self,
        experiment_name: str,
        algorithm: str,
        phase_name: str,
        trial_number: int,
    ) -> Path:
        """Return the YAML path for one resolved trial config snapshot.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            phase_name (str): Phase label.
            trial_number (int): Optuna trial index.

        Returns:
            Path: ``resolved_config.yaml`` inside the trial directory.
        """
        return (
            self.get_optuna_trial_dir(
                experiment_name, algorithm, phase_name, trial_number
            )
            / "resolved_config.yaml"
        )

    def get_optuna_trial_summary_path(
        self,
        experiment_name: str,
        algorithm: str,
        phase_name: str,
        trial_number: int,
    ) -> Path:
        """Return the JSON path for one trial summary.

        Args:
            experiment_name (str): Experiment profile name.
            algorithm (str): Algorithm name.
            phase_name (str): Phase label.
            trial_number (int): Optuna trial index.

        Returns:
            Path: ``trial_summary.json`` inside the trial directory.
        """
        return (
            self.get_optuna_trial_dir(
                experiment_name, algorithm, phase_name, trial_number
            )
            / "trial_summary.json"
        )

    def get_vecnormalize_path(self, run_artifacts: RunArtifacts) -> Path:
        """Return the canonical VecNormalize stats path for a run.

        Args:
            run_artifacts (RunArtifacts): Run artifact layout.

        Returns:
            Path: Path used to persist normalization statistics for the final
                model of the run.
        """
        return run_artifacts.models_dir / "vecnormalize.pkl"

    def get_best_vecnormalize_path(self, run_artifacts: RunArtifacts) -> Path:
        """Return the canonical VecNormalize stats path for the best model.

        Args:
            run_artifacts (RunArtifacts): Run artifact layout.

        Returns:
            Path: Path used to persist normalization statistics for the best
                checkpoint selected during training.
        """
        return run_artifacts.models_dir / "best_model" / "vecnormalize.pkl"


def find_run_root_from_model_path(model_path: Path) -> Path | None:
    """Find the run root that owns a saved model artifact.

    The search walks upward from the model path until it finds the canonical
    ``config/resolved_config.yaml`` marker used by training runs.

    Args:
        model_path (Path): Path to a saved model file.

    Returns:
        Path | None: Run root if a matching artifact layout is found, otherwise
            ``None``.
    """
    for candidate in [model_path.parent, *model_path.parents]:
        if (candidate / "config" / "resolved_config.yaml").exists():
            return candidate

    return None


def find_vecnormalize_path_from_model_path(model_path: Path) -> Path | None:
    """Find the normalization stats file that matches a saved model path.

    Args:
        model_path (Path): Model file used for evaluation or visualization.

    Returns:
        Path | None: Matching VecNormalize stats path when one exists.
    """
    direct_candidate = model_path.parent / "vecnormalize.pkl"
    if direct_candidate.exists():
        return direct_candidate

    run_root = find_run_root_from_model_path(model_path)
    if run_root is None:
        return None

    run_candidate = run_root / "models" / "vecnormalize.pkl"
    if run_candidate.exists():
        return run_candidate

    return None

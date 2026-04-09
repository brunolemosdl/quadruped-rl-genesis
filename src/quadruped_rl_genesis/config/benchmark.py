"""Load and merge benchmark profile YAML with CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quadruped_rl_genesis.settings import AppSettings
from quadruped_rl_genesis.utils.io import read_yaml


@dataclass(frozen=True)
class ResolvedBenchmarkArgs:
    """Fully resolved benchmark invocation after profile YAML + CLI merge.

    Attributes:
        output_root (Path): Benchmark output directory root.
        experiments (list[str]): Experiment profile names to run.
        algorithms (list[str]): Algorithm names included in the sweep.
        seeds (list[int]): Training seeds per run.
        device (str): Default Torch/Genesis device string.
        genesis_device (str): Device string for Genesis simulation.
        algorithm_device (str): Device string for SB3.
        hyperparams_source (str): Hyperparameter profile for train/eval.
        tune_base_source (str): Base hyperparameters for Optuna studies.
        tune_seed (int): Optuna RNG seed.
        eval_seed (int): Final evaluation RNG seed.
        final_eval_episodes (int): Episodes in deterministic eval.
        skip_tune (bool): Whether Optuna is skipped.
        skip_train (bool): Whether training is skipped.
        skip_eval (bool): Whether final eval is skipped.
        record_videos (bool): Whether to record rollout videos.
        video_episodes (int): Episodes per recorded video.
        video_max_steps (int): Step cap per video episode.
        video_seed (int | None): Optional video RNG seed.
        video_fast_viz (bool): Lighter visualization while recording.
        resume (bool): Skip completed training triples when True.
    """

    output_root: Path
    experiments: list[str]
    algorithms: list[str]
    seeds: list[int]
    device: str
    genesis_device: str
    algorithm_device: str
    hyperparams_source: str
    tune_base_source: str
    tune_seed: int
    eval_seed: int
    final_eval_episodes: int
    skip_tune: bool
    skip_train: bool
    skip_eval: bool
    record_videos: bool
    video_episodes: int
    video_max_steps: int
    video_seed: int | None
    video_fast_viz: bool
    resume: bool


_BENCHMARK_FALLBACKS: dict[str, Any] = {
    "output_root": "paper_outputs",
    "experiments": ["flat", "irregular", "rough"],
    "algorithms": ["ppo", "sac", "td3"],
    "seeds": [42, 43, 44, 45, 46],
    "hyperparams_source": None,
    "tune_base_source": "default",
    "final_eval_episodes": 100,
}


def resolve_profile_path(settings: AppSettings, name: str) -> Path:
    """Resolve a benchmark profile path from a short name or explicit path.

    Args:
        settings (AppSettings): Application settings (``configs_root``).
        name (str): Absolute path, relative path, or basename resolved under
            ``configs/benchmarks/<name>.yaml``.

    Returns:
        Path: Existing benchmark profile YAML file.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    raw = Path(name)
    if raw.is_absolute() and raw.exists():
        return raw
    if raw.exists():
        return raw.resolve()
    stem = name.removesuffix(".yaml")
    candidate = settings.configs_root / "benchmarks" / f"{stem}.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Benchmark profile not found: {name!r} "
        f"(searched {settings.configs_root / 'benchmarks'})"
    )


def load_profile(path: Path) -> dict[str, Any]:
    """Load benchmark profile YAML from disk.

    Args:
        path (Path): Profile file path.

    Returns:
        dict[str, Any]: Parsed mapping (may be empty).
    """
    return read_yaml(path)


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Expand nested ``video`` block into flat keys used by the CLI merge.

    Args:
        profile (dict[str, Any]): Raw profile payload.

    Returns:
        dict[str, Any]: Normalized mapping with ``video_*`` keys set.
    """
    out = dict(profile)
    nested = out.pop("video", None)
    if isinstance(nested, dict):
        if "video_episodes" not in out and "episodes" in nested:
            out["video_episodes"] = nested.get("episodes")
        if "video_max_steps" not in out and "max_steps" in nested:
            out["video_max_steps"] = nested.get("max_steps")
        if "video_seed" not in out and "seed" in nested:
            out["video_seed"] = nested.get("seed")
        if "video_fast_viz" not in out and "fast_viz" in nested:
            out["video_fast_viz"] = nested.get("fast_viz")
    return out


def _pick(
    args: Any,
    profile: dict[str, Any],
    attr: str,
    key: str,
    fallback: Any,
    settings: AppSettings | None = None,
) -> Any:
    if attr in vars(args):
        return getattr(args, attr)
    if key in profile:
        val = profile[key]
        if val is not None:
            return val
    if fallback is None and attr == "hyperparams_source" and settings is not None:
        return settings.default_hyperparams_source
    return fallback


def _pick_seed_override(
    args: Any,
    profile: dict[str, Any],
    attr: str,
    key: str,
    master: int,
) -> int:
    if attr in vars(args):
        v = getattr(args, attr)
        return master if v is None else int(v)
    if key in profile and profile[key] is not None:
        return int(profile[key])
    return master


def _pick_bool(
    args: Any,
    profile: dict[str, Any],
    attr: str,
    key: str,
    default: bool,
) -> bool:
    v = vars(args).get(attr)
    if v is not None:
        return bool(v)
    if key in profile and profile[key] is not None:
        return bool(profile[key])
    return default


def merge_benchmark_invocation(
    args: Any,
    settings: AppSettings,
    profile: dict[str, Any] | None,
) -> ResolvedBenchmarkArgs:
    """Merge ``--profile`` YAML with explicit CLI flags (CLI wins when set).

    Args:
        args (argparse.Namespace): Parsed benchmark subcommand namespace.
        settings (AppSettings): Application defaults (seed, device, hyperparams).
        profile (dict[str, Any] | None): Normalized profile payload, or
            ``None`` when ``--profile`` was not used.

    Returns:
        ResolvedBenchmarkArgs: Values to pass to ``run_benchmark``.
    """
    flat = normalize_profile(profile or {})

    output_root = Path(
        str(
            _pick(
                args,
                flat,
                "output_root",
                "output_root",
                _BENCHMARK_FALLBACKS["output_root"],
            )
        )
    )
    experiments = list(
        _pick(
            args,
            flat,
            "experiments",
            "experiments",
            _BENCHMARK_FALLBACKS["experiments"],
        )
    )
    algorithms = list(
        _pick(
            args,
            flat,
            "algorithms",
            "algorithms",
            _BENCHMARK_FALLBACKS["algorithms"],
        )
    )
    seeds = [
        int(s)
        for s in _pick(
            args,
            flat,
            "seeds",
            "seeds",
            _BENCHMARK_FALLBACKS["seeds"],
        )
    ]

    device = str(
        _pick(args, flat, "device", "device", settings.device, settings=settings)
    )
    genesis_raw = _pick(args, flat, "genesis_device", "genesis_device", None)
    algorithm_raw = _pick(args, flat, "algorithm_device", "algorithm_device", None)
    genesis_device = str(genesis_raw) if genesis_raw else device
    algorithm_device = str(algorithm_raw) if algorithm_raw else device

    hyperparams_source = str(
        _pick(
            args,
            flat,
            "hyperparams_source",
            "hyperparams_source",
            _BENCHMARK_FALLBACKS["hyperparams_source"],
            settings=settings,
        )
    )
    tune_base_source = str(
        _pick(
            args,
            flat,
            "tune_base_source",
            "tune_base_source",
            _BENCHMARK_FALLBACKS["tune_base_source"],
        )
    )

    master_seed = int(
        _pick(args, flat, "seed", "seed", settings.seed, settings=settings)
    )
    tune_seed = _pick_seed_override(args, flat, "tune_seed", "tune_seed", master_seed)
    eval_seed = _pick_seed_override(args, flat, "eval_seed", "eval_seed", master_seed)

    final_eval_episodes = int(
        _pick(
            args,
            flat,
            "final_eval_episodes",
            "final_eval_episodes",
            _BENCHMARK_FALLBACKS["final_eval_episodes"],
        )
    )

    skip_tune = _pick_bool(args, flat, "skip_tune", "skip_tune", False)
    skip_train = _pick_bool(args, flat, "skip_train", "skip_train", False)
    skip_eval = _pick_bool(args, flat, "skip_eval", "skip_eval", False)
    record_videos = _pick_bool(args, flat, "record_videos", "record_videos", False)
    resume = _pick_bool(args, flat, "resume", "resume", False)

    video_episodes = int(
        _pick(args, flat, "video_episodes", "video_episodes", 1),
    )
    video_max_steps = int(
        _pick(args, flat, "video_max_steps", "video_max_steps", 1200),
    )
    video_seed_raw = _pick(
        args,
        flat,
        "video_seed",
        "video_seed",
        None,
    )
    video_seed = int(video_seed_raw) if video_seed_raw is not None else None
    video_fast_viz = _pick_bool(
        args,
        flat,
        "video_fast_viz",
        "video_fast_viz",
        False,
    )

    return ResolvedBenchmarkArgs(
        output_root=output_root,
        experiments=experiments,
        algorithms=algorithms,
        seeds=seeds,
        device=device,
        genesis_device=genesis_device,
        algorithm_device=algorithm_device,
        hyperparams_source=hyperparams_source,
        tune_base_source=tune_base_source,
        tune_seed=tune_seed,
        eval_seed=eval_seed,
        final_eval_episodes=final_eval_episodes,
        skip_tune=skip_tune,
        skip_train=skip_train,
        skip_eval=skip_eval,
        record_videos=record_videos,
        video_episodes=video_episodes,
        video_max_steps=video_max_steps,
        video_seed=video_seed,
        video_fast_viz=video_fast_viz,
        resume=resume,
    )

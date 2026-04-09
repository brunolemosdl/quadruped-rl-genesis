"""High-level training, tuning, evaluation, and visualization pipelines."""

from __future__ import annotations

from typing import Any

from quadruped_rl_genesis.pipeline.evaluate import run_evaluation
from quadruped_rl_genesis.pipeline.train import run_training
from quadruped_rl_genesis.pipeline.tune import run_tuning

__all__ = [
    "run_evaluation",
    "run_training",
    "run_tuning",
    "run_visualization",
]


def __getattr__(name: str) -> Any:
    if name == "run_visualization":
        from quadruped_rl_genesis.pipeline.visualize import run_visualization

        return run_visualization
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Lazy entrypoints for benchmark, check, setup, and monitor CLI workflows."""

from __future__ import annotations

from typing import Any


def run_benchmark(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the benchmark workflow (lazy import).

    Args:
        *args: Forwarded to ``quadruped_rl_genesis.operations.benchmark.run_benchmark``.
        **kwargs: Forwarded to ``quadruped_rl_genesis.operations.benchmark.run_benchmark``.

    Returns:
        dict[str, Any]: Benchmark summary payload.
    """
    from quadruped_rl_genesis.operations.benchmark import (
        run_benchmark as _run_benchmark,
    )

    return _run_benchmark(*args, **kwargs)


def run_check(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run environment and dependency checks (lazy import).

    Args:
        *args: Forwarded to ``operations.check.run_check``.
        **kwargs: Forwarded to ``operations.check.run_check``.

    Returns:
        dict[str, Any]: Check report payload.
    """
    from quadruped_rl_genesis.operations.check import run_check as _run_check

    return _run_check(*args, **kwargs)


def run_setup(*args: Any, **kwargs: Any) -> dict[str, str]:
    """Run project setup steps (lazy import).

    Args:
        *args: Forwarded to ``operations.setup.run_setup``.
        **kwargs: Forwarded to ``operations.setup.run_setup``.

    Returns:
        dict[str, str]: Setup status messages keyed by step name.
    """
    from quadruped_rl_genesis.operations.setup import run_setup as _run_setup

    return _run_setup(*args, **kwargs)


def run_monitor(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run the live metrics monitor (lazy import).

    Args:
        *args: Forwarded to ``operations.monitor.run_monitor``.
        **kwargs: Forwarded to ``operations.monitor.run_monitor``.

    Returns:
        dict[str, Any]: Monitor snapshot or empty dict depending on mode.
    """
    from quadruped_rl_genesis.operations.monitor import run_monitor as _run_monitor

    return _run_monitor(*args, **kwargs)


__all__ = ["run_benchmark", "run_check", "run_monitor", "run_setup"]

"""Base optimizer interface and result container.

All hyperparameter optimizers (Ray Tune, Optuna, grid search, etc.) must
implement :class:`BaseOptimizer` so the runner can invoke HPO uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OptimizationResult:
    """Container for the outcome of a hyperparameter search.

    Attributes:
        best_config: Hyperparameters of the best trial.
        best_value: Metric value achieved by the best trial.
        n_trials: Total number of evaluated configurations.
        trials_summary: Per-trial records with config and metrics.
        best_checkpoint_path: Filesystem path to the best checkpoint
            (model state dict), if checkpoint saving was enabled.
        storage_path: Root directory where Ray Tune stored trial results.
    """

    best_config: Dict[str, Any]
    best_value: float
    n_trials: int
    trials_summary: List[Dict[str, Any]] = field(default_factory=list)
    best_checkpoint_path: Optional[str] = None
    storage_path: Optional[str] = None


class BaseOptimizer(ABC):
    """Abstract base class for hyperparameter optimizers.

    Subclasses must implement :meth:`run`.
    """

    @abstractmethod
    def run(self, spec: Any, search_space: Dict[str, Any]) -> OptimizationResult:
        """Execute a hyperparameter search.

        Args:
            spec: An :class:`ExperimentSpec` describing the experiment.
            search_space: Mapping of parameter name → search domain.

        Returns:
            An :class:`OptimizationResult` with the best configuration.
        """

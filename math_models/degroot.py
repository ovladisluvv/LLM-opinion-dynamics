import numpy as np

from math_models.common import MathModel, TrajectoryResult, has_consensus, validate_opinions, validate_weights

__all__ = ["DegrootModel", "degroot_step", "simulate_degroot", "has_consensus", "validate_weights", "validate_opinions"]


class DegrootModel(MathModel):
    """
    DeGroot model. The update rule is:
        x(t + 1) = W @ x(t)
    The run stops when the opinions reach consensus (spread at most eps)
    """
    name = "degroot"

    def step(self, opinions: np.ndarray, initial_opinions: np.ndarray) -> np.ndarray:
        return self.weights @ opinions

    def check_stop(self, previous: np.ndarray | None, current: np.ndarray, eps: float) -> bool:
        return has_consensus(current, eps)


def degroot_step(weights: np.ndarray, opinions: np.ndarray) -> np.ndarray:
    """Perform one DeGroot update step x(t + 1) = W @ x(t)"""
    return weights @ opinions


def simulate_degroot(weights: np.ndarray, opinions: np.ndarray, max_steps: int, eps: float = 1e-6) -> TrajectoryResult:
    """Simulate the DeGroot opinion dynamics model"""
    return DegrootModel(weights).simulate(opinions, max_steps, eps)


if __name__ == "__main__":
    W = np.array([
        [0.5, 0.4, 0.0, 0.1],
        [0.2, 0.5, 0.3, 0.0],
        [0.1, 0.3, 0.6, 0.0],
        [0.1, 0.0, 0.2, 0.7]
    ], dtype=float)

    x0 = np.array([1.0, 0.5, 0.0, 0.2], dtype=float)
    T = 100

    result = simulate_degroot(W, x0, T, eps=1e-6)

    result.print_trajectory()
    result.print_results()

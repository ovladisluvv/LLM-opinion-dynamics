import numpy as np

from math_models.common import MathModel, has_consensus


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


if __name__ == "__main__":
    W = np.array([
        [0.5, 0.4, 0.0, 0.1],
        [0.2, 0.5, 0.3, 0.0],
        [0.1, 0.3, 0.6, 0.0],
        [0.1, 0.0, 0.2, 0.7]
    ], dtype=float)

    x0 = np.array([1.0, 0.5, 0.0, 0.2], dtype=float)
    T = 100

    result = DegrootModel(W).simulate(x0, T, eps=1e-6)

    result.print_trajectory()
    result.print_results()

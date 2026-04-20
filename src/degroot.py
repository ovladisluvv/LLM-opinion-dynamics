import numpy as np


class DegrootResult:
    def __init__(self):
        self.trajectory = []
        self.final_opinions = []
        self.consensus_reached = False
        self.total_steps = 0
        self.consensus_steps_count = 0

    def print_trajectory(self, precision: int = 6):
        """
        Print the opinion trajectory
        """
        print("DeGroot opinion trajectory:")

        for t, opinions in enumerate(self.trajectory):
            formatted_opinions = ", ".join(f"{opinion:.{precision}f}" for opinion in opinions)
            print(f"Step {t:>{len(str(self.total_steps))}}: {formatted_opinions}")

        print()

    def print_results(self):
        """
        Print the simulation results
        """
        print("DeGroot simulation results:")
        print(f"- Total steps: {self.total_steps}")

        if self.trajectory:
            print(f"- Final opinions:")

            for i, opinion in enumerate(self.final_opinions):
                print(f"    Agent {i}: {opinion:.6f}")

        print(f"- Consensus {"is" if self.consensus_reached else "isn't"} reached")

        if self.consensus_reached:
            print(f"- Steps to consensus: {self.consensus_steps_count}")

        print()


def validate_weights(weights: np.ndarray) -> None:
    """
    Validate the DeGroot weight matrix

    Parameters:
        weights: np.ndarray
            Row-stochastic influence matrix of shape (n, n)

    Raises:
        ValueError
            If the matrix is empty, not square, not row-stochastic or has negative values
    """
    if weights.ndim != 2:
        raise ValueError("Weight matrix has to be two-dimensional")

    rows, cols = weights.shape
    if rows == 0 or cols == 0:
        raise ValueError("Weight matrix has to be not empty")

    if rows != cols:
        raise ValueError("Weight matrix has to be square")

    if np.any(weights < 0):
        raise ValueError("Weight matrix has to be non-negative")

    row_sums = weights.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"Each row has to sum to 1. Current row sums: {row_sums}")


def validate_opinions(weights: np.ndarray, opinions: np.ndarray) -> None:
    """
    Validate the initial opinion vector

    Parameters:
        weights: np.ndarray
            Weight matrix of shape (n, n)
        opinions: np.ndarray
            Opinion vector of shape (n,)

    Raises
        ValueError
            If the opinion vector is not one-dimensional or has incompatible size with the weight matrix
    """
    if opinions.ndim != 1:
        raise ValueError("Opinion vector has to be one-dimensional")

    if weights.shape[0] != opinions.shape[0]:
        raise ValueError("The size of the opinion vector has to match the weight matrix size")


def degroot_step(weights: np.ndarray, opinions: np.ndarray) -> np.ndarray:
    """
    Perform one DeGroot update step

    The update rule is:
        x(t + 1) = W @ x(t)

    Parameters
        weights: np.ndarray
            Weight matrix of shape (n, n).
        opinions: np.ndarray
            Current opinion vector of shape (n,)

    Returns
        np.ndarray
            Updated opinion vector of shape (n,)
    """
    return weights @ opinions


def has_consensus(opinions: np.ndarray, eps: float = 1e-6) -> bool:
    """
    Check if the system has reached consensus. A state is treated as consensus if
    the difference between the maximum and minimum opinion is at most eps

    Parameters
        opinions: np.ndarray
            Current opinion vector
        eps: float, default=1e-6
            Consensus tolerance

    Returns
        bool
            True if consensus is reached, otherwise False
    """
    return np.ptp(opinions) <= eps


def simulate_degroot(weights: np.ndarray, opinions: np.ndarray, max_steps: int, eps: float = 1e-6) -> DegrootResult:
    """
    Simulate the DeGroot opinion dynamics model

    Parameters
        weights: np.ndarray
            Row-stochastic weight matrix of shape (n, n)
        opinions: np.ndarray
            Initial opinion vector of shape (n,)
        max_steps: int
            Maximum number of update steps
        eps: float, default=1e-6
            Consensus tolerance

    Returns
        DegrootResult
            Full opinion trajectory of shape (steps + 1, n),
            where the first row is the initial state.
    """
    if max_steps < 0:
        raise ValueError("Maximum number of steps has to be non-negative")

    if eps < 0:
        raise ValueError("Epsilon has to be non-negative")

    weights = np.asarray(weights, dtype=float)
    validate_weights(weights)

    opinions = np.asarray(opinions, dtype=float)
    validate_opinions(weights, opinions)

    res = DegrootResult()
    cur_opinion = opinions.copy()
    res.trajectory.append(cur_opinion.copy())

    if has_consensus(cur_opinion, eps):
        res.final_opinions = cur_opinion.copy()
        res.consensus_reached = True

        return res

    for step in range(1, max_steps + 1):
        cur_opinion = degroot_step(weights, cur_opinion)
        res.trajectory.append(cur_opinion.copy())
        res.total_steps += 1

        if has_consensus(cur_opinion, eps):
            res.final_opinions = cur_opinion.copy()
            res.consensus_reached = True
            res.consensus_steps_count = step

            return res

    res.final_opinions = cur_opinion.copy()
    res.consensus_reached = False

    return res


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

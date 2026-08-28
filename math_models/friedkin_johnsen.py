import numpy as np


class FriedkinJohnsenResult:
    """Class to store the results of the Friedkin-Johnsen simulation"""
    def __init__(self):
        self.trajectory = []
        self.final_opinions = np.array([])
        self.converged = False
        self.total_steps = 0
        self.convergence_steps_count = 0

    def print_trajectory(self, precision: int = 6):
        """Print the opinion trajectory"""
        print("Friedkin-Johnsen opinion trajectory:")

        for t, opinions in enumerate(self.trajectory):
            formatted_opinions = ", ".join(f"{opinion:.{precision}f}" for opinion in opinions)
            print(f"Step {t:>{len(str(self.total_steps))}}: {formatted_opinions}")

        print()

    def print_results(self):
        """Print the simulation results"""
        print("Friedkin-Johnsen simulation results:")
        print(f"- Total steps: {self.total_steps}")

        if self.trajectory:
            print("- Final opinions:")

            for i, opinion in enumerate(self.final_opinions):
                print(f"    Agent {i}: {opinion:.6f}")

        convergence_verb = "has" if self.converged else "hasn't"
        print(f"- Model {convergence_verb} converged")

        if self.converged:
            print(f"- Steps to convergence: {self.convergence_steps_count}")

        print()


def validate_weights(weights: np.ndarray) -> None:
    """Validate the Friedkin-Johnsen weight matrix"""
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
    """Validate the initial opinion vector"""
    if opinions.ndim != 1:
        raise ValueError("Opinion vector has to be one-dimensional")

    if weights.shape[0] != opinions.shape[0]:
        raise ValueError("The size of the opinion vector has to match the weight matrix size")


def validate_susceptibility(weights: np.ndarray, susceptibility: np.ndarray) -> None:
    """Validate the vector of agents' susceptibility to social influence"""
    if susceptibility.ndim != 1:
        raise ValueError("Susceptibility vector has to be one-dimensional")

    if weights.shape[0] != susceptibility.shape[0]:
        raise ValueError("The size of the susceptibility vector has to match the weight matrix size")

    if np.any((susceptibility < 0) | (susceptibility > 1)):
        raise ValueError("Each susceptibility value has to be in the interval [0, 1]")


def friedkin_johnsen_step(
    weights: np.ndarray,
    opinions: np.ndarray,
    initial_opinions: np.ndarray,
    susceptibility: np.ndarray,
) -> np.ndarray:
    """
    Perform one Friedkin-Johnsen update step
    The update rule is:
        x(t + 1) = Λ @ W @ x(t) + (I - Λ) @ x(0)
    where Λ is a diagonal matrix of agents' susceptibility to social influence
    """
    social_influence = weights @ opinions
    return susceptibility * social_influence + (1.0 - susceptibility) * initial_opinions


def has_converged(previous_opinions: np.ndarray, opinions: np.ndarray, eps: float = 1e-6) -> bool:
    """
    Check if the system has converged to a stationary state. 
    A state is treated as converged if the maximum absolute change in an
    agent's opinion between two consecutive steps is at most eps
    """
    return np.max(np.abs(opinions - previous_opinions)) <= eps


def simulate_friedkin_johnsen(
    weights: np.ndarray,
    opinions: np.ndarray,
    susceptibility: np.ndarray,
    max_steps: int,
    eps: float = 1e-6,
) -> FriedkinJohnsenResult:
    """Simulate the Friedkin-Johnsen opinion dynamics model"""
    if max_steps < 0:
        raise ValueError("Maximum number of steps has to be non-negative")

    if eps < 0:
        raise ValueError("Epsilon has to be non-negative")

    weights = np.asarray(weights, dtype=float)
    validate_weights(weights)

    opinions = np.asarray(opinions, dtype=float)
    validate_opinions(weights, opinions)

    susceptibility = np.asarray(susceptibility, dtype=float)
    validate_susceptibility(weights, susceptibility)

    result = FriedkinJohnsenResult()
    initial_opinions = opinions.copy()
    cur_opinion = opinions.copy()
    result.trajectory.append(cur_opinion.copy())

    for step in range(1, max_steps + 1):
        previous_opinion = cur_opinion.copy()
        cur_opinion = friedkin_johnsen_step(
            weights,
            previous_opinion,
            initial_opinions,
            susceptibility,
        )
        result.trajectory.append(cur_opinion.copy())
        result.total_steps += 1

        if has_converged(previous_opinion, cur_opinion, eps):
            result.final_opinions = cur_opinion.copy()
            result.converged = True
            result.convergence_steps_count = step

            return result

    result.final_opinions = cur_opinion.copy()
    result.converged = False

    return result


if __name__ == "__main__":
    W = np.array([
        [0.5, 0.4, 0.0, 0.1],
        [0.2, 0.5, 0.3, 0.0],
        [0.1, 0.3, 0.6, 0.0],
        [0.1, 0.0, 0.2, 0.7]
    ], dtype=float)

    x0 = np.array([1.0, 0.5, 0.0, 0.2], dtype=float)
    susceptibility = np.array([0.8, 0.6, 0.9, 0.7], dtype=float)
    T = 100

    result = simulate_friedkin_johnsen(W, x0, susceptibility, T, eps=1e-6)

    result.print_trajectory()
    result.print_results()

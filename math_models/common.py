from abc import ABC, abstractmethod
import numpy as np

from agents.agent_state import AgentState


def validate_weights(weights: np.ndarray) -> None:
    """Validate a row-stochastic influence matrix"""
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
    """Validate an opinion vector against the weight matrix"""
    if opinions.ndim != 1:
        raise ValueError("Opinion vector has to be one-dimensional")

    if weights.shape[0] != opinions.shape[0]:
        raise ValueError("The size of the opinion vector has to match the weight matrix size")


def has_consensus(opinions: np.ndarray, eps: float = 1e-6) -> bool:
    """Consensus: the spread between the maximum and minimum opinion is at most eps"""
    return np.ptp(opinions) <= eps


def has_converged(previous_opinions: np.ndarray, opinions: np.ndarray, eps: float = 1e-6) -> bool:
    """Stationarity: the maximum absolute change of any opinion between two consecutive steps is at most eps"""
    return np.max(np.abs(opinions - previous_opinions)) <= eps


class TrajectoryResult:
    """Numeric opinion trajectory of a mathematical model together with its stopping information"""
    def __init__(self):
        self.trajectory: list[np.ndarray] = []
        self.total_steps = 0
        self.converged = False
        self.convergence_step: int | None = None

    @property
    def final_opinions(self) -> np.ndarray:
        if not self.trajectory:
            return np.array([])

        return self.trajectory[-1].copy()

    def print_trajectory(self, precision: int = 6) -> None:
        print("Opinion trajectory:")

        for t, opinions in enumerate(self.trajectory):
            formatted_opinions = ", ".join(f"{opinion:.{precision}f}" for opinion in opinions)
            print(f"Step {t:>{len(str(self.total_steps))}}: {formatted_opinions}")

        print()

    def print_results(self) -> None:
        print("Simulation results:")
        print(f"- Total steps: {self.total_steps}")

        if self.trajectory:
            print("- Final opinions:")

            for i, opinion in enumerate(self.final_opinions):
                print(f"    Agent {i}: {opinion:.6f}")

        convergence_verb = "has" if self.converged else "hasn't"
        print(f"- Model {convergence_verb} converged")

        if self.converged:
            print(f"- Steps to convergence: {self.convergence_step}")

        print()


class MathModel(ABC):
    """
    Opinion dynamics model on a fixed row-stochastic weight matrix

    A model defines three things the experiment pipeline relies on:
    the numeric update rule (step), the stopping criterion (check_stop), which the
    LLM runner applies to judge scores as well, and the model-specific information
    shown to a participant agent (participant_prompt_fields)
    """
    name: str = ""

    def __init__(self, weights: np.ndarray):
        weights = np.asarray(weights, dtype=float)
        validate_weights(weights)
        self.weights = weights

    @property
    def size(self) -> int:
        return self.weights.shape[0]

    @abstractmethod
    def step(self, opinions: np.ndarray, initial_opinions: np.ndarray) -> np.ndarray:
        """Return the opinions after one update step"""

    @abstractmethod
    def check_stop(self, previous: np.ndarray | None, current: np.ndarray, eps: float) -> bool:
        """Stopping criterion. `previous` is None when checking the initial state"""

    def participant_prompt_fields(self, agent_index: int, agent: AgentState) -> dict[str, str]:
        """Model-specific placeholders for the participant prompt block. Empty by default"""
        return {}

    def describe(self) -> dict:
        """JSON-serializable description for experiment metadata"""
        return {"name": self.name, "params": self.params()}

    def params(self) -> dict:
        """Model parameters as JSON-serializable values"""
        return {}

    def simulate(self, initial_opinions: np.ndarray, max_steps: int, eps: float = 1e-6) -> TrajectoryResult:
        """Run the model from the initial opinions until the stopping criterion holds or max_steps is reached"""
        if max_steps < 0:
            raise ValueError("Maximum number of steps has to be non-negative")

        if eps < 0:
            raise ValueError("Epsilon has to be non-negative")

        initial_opinions = np.asarray(initial_opinions, dtype=float)
        validate_opinions(self.weights, initial_opinions)

        result = TrajectoryResult()
        current = initial_opinions.copy()
        result.trajectory.append(current.copy())

        if self.check_stop(None, current, eps):
            result.converged = True
            result.convergence_step = 0

            return result

        for step in range(1, max_steps + 1):
            previous = current
            current = self.step(previous, initial_opinions)
            result.trajectory.append(current.copy())
            result.total_steps = step

            if self.check_stop(previous, current, eps):
                result.converged = True
                result.convergence_step = step

                return result

        return result

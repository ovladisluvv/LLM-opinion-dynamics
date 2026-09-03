import numpy as np

from agents.agent_state import AgentState
from math_models.common import MathModel, TrajectoryResult, has_converged


def validate_susceptibility(weights: np.ndarray, susceptibility: np.ndarray) -> None:
    """Validate the vector of agents' susceptibility to social influence"""
    if susceptibility.ndim != 1:
        raise ValueError("Susceptibility vector has to be one-dimensional")

    if weights.shape[0] != susceptibility.shape[0]:
        raise ValueError(
            f"The size of the susceptibility vector has to match the weight matrix size. "
            f"Got {susceptibility.shape[0]} values for {weights.shape[0]} agents"
        )

    if np.any((susceptibility < 0) | (susceptibility > 1)):
        raise ValueError("Each susceptibility value has to be in the interval [0, 1]")


class FriedkinJohnsenModel(MathModel):
    """
    Friedkin-Johnsen model. The update rule is:
        x(t + 1) = Λ @ W @ x(t) + (I - Λ) @ x(0)
    where Λ is a diagonal matrix of agents' susceptibility to social influence.
    The run stops when the opinions become stationary (maximum change at most eps);
    the stationary state is generally not a consensus
    """
    name = "friedkin_johnsen"

    def __init__(self, weights: np.ndarray, susceptibility: list[float]):
        super().__init__(weights)

        susceptibility = np.asarray(susceptibility, dtype=float)
        validate_susceptibility(self.weights, susceptibility)
        self.susceptibility = susceptibility

    def step(self, opinions: np.ndarray, initial_opinions: np.ndarray) -> np.ndarray:
        social_influence = self.weights @ opinions
        return self.susceptibility * social_influence + (1.0 - self.susceptibility) * initial_opinions

    def check_stop(self, previous: np.ndarray | None, current: np.ndarray, eps: float) -> bool:
        if previous is None:
            return False

        return has_converged(previous, current, eps)

    def participant_prompt_fields(self, agent_index: int, agent: AgentState) -> dict[str, str]:
        susceptibility = float(self.susceptibility[agent_index])

        return {
            "initial_opinion_text": agent.initial_opinion_text,
            "susceptibility": f"{susceptibility:.4f}",
            "anchor_weight": f"{1.0 - susceptibility:.4f}",
        }

    def params(self) -> dict:
        return {"susceptibility": self.susceptibility.tolist()}


def friedkin_johnsen_step(
    weights: np.ndarray,
    opinions: np.ndarray,
    initial_opinions: np.ndarray,
    susceptibility: np.ndarray,
) -> np.ndarray:
    """Perform one Friedkin-Johnsen update step x(t + 1) = Λ @ W @ x(t) + (I - Λ) @ x(0)"""
    social_influence = weights @ opinions
    return susceptibility * social_influence + (1.0 - susceptibility) * initial_opinions


def simulate_friedkin_johnsen(
    weights: np.ndarray,
    opinions: np.ndarray,
    susceptibility: np.ndarray,
    max_steps: int,
    eps: float = 1e-6,
) -> TrajectoryResult:
    """Simulate the Friedkin-Johnsen opinion dynamics model"""
    return FriedkinJohnsenModel(weights, susceptibility=susceptibility).simulate(opinions, max_steps, eps)


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

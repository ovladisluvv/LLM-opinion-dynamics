import numpy as np

from agents.agent_state import AgentState, NeighborState
from math_models.degroot import validate_weights


def validate_network(agents: list[AgentState], weights: np.ndarray) -> np.ndarray:
    agent_ids = [agent.agent_id for agent in agents]

    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError(f"Agent IDs has to be unique. Found duplicate IDs in {agent_ids}")

    weights = np.asarray(weights, dtype=float)

    validate_weights(weights)

    if len(agents) != weights.shape[0]:
        raise ValueError(
            f"The number of agents has to match the dimensions of the weights matrix. "
            f"Got {len(agents)} agents and weight matrix of shape {weights.shape}"
        )

    return weights


class AgentNetwork:
    def __init__(self, agents: list[AgentState], weights: np.ndarray):
        self.agents = agents
        self.weights = validate_network(agents, weights)
        self.agent_positions = {agent.agent_id: index for index, agent in enumerate(agents)}
        self.sync_self_trust()

    def get_agent_index(self, agent_id: int) -> int:
        if agent_id not in self.agent_positions:
            raise ValueError(f"Agent with ID {agent_id} not found in the network")

        return self.agent_positions[agent_id]

    def get_agent(self, agent_id: int) -> AgentState:
        agent_index = self.get_agent_index(agent_id)
        return self.agents[agent_index]

    def get_self_trust(self, agent_id: int) -> float:
        agent_index = self.get_agent_index(agent_id)
        return self.weights[agent_index, agent_index]

    def sync_self_trust(self) -> None:
        for agent in self.agents:
            agent.self_trust = self.get_self_trust(agent.agent_id)

    def get_neighbors(self, agent_id: int) -> list[NeighborState]:
        agent_index = self.get_agent_index(agent_id)
        neighbors = []

        for neighbor_index, weight in enumerate(self.weights[agent_index]):
            if weight > 0 and neighbor_index != agent_index:
                neighbor_agent = self.agents[neighbor_index]
                neighbors.append(
                    NeighborState(
                        agent_id=neighbor_agent.agent_id,
                        weight=weight,
                        current_opinion_text=neighbor_agent.current_opinion_text,
                        current_opinion_score=neighbor_agent.current_opinion_score
                    )
                )

        return neighbors

    def update_agent_opinion(self, agent_id: int, new_opinion_text: str, new_opinion_score: float | None = None) -> None:
        agent = self.get_agent(agent_id)
        agent.current_opinion_text = new_opinion_text
        agent.current_opinion_score = new_opinion_score

    def update_network_opinions(self, updates: dict[int, tuple[str, float]]) -> None:
        for agent_id, (new_opinion_text, new_opinion_score) in updates.items():
            self.update_agent_opinion(
                agent_id=agent_id,
                new_opinion_text=new_opinion_text,
                new_opinion_score=new_opinion_score
            )

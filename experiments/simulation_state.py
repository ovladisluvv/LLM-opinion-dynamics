from agents.network import AgentNetwork


class SimulationResult:
    def __init__(
        self,
        score_trajectory: list[list[float]],
        text_trajectory: list[list[str]],
        agent_ids: list[int],
        total_steps: int = 0,
        consensus_reached: bool = False,
        consensus_step: int | None = None
    ):
        self.score_trajectory = score_trajectory
        self.text_trajectory = text_trajectory
        self.agent_ids = agent_ids
        self.total_steps = total_steps
        self.consensus_reached = consensus_reached
        self.consensus_step = consensus_step
        

    def add_trajectory_step(self, network: AgentNetwork) -> None:
        step_scores = []
        step_texts = []

        for agent in network.agents:
            if agent.current_opinion_score is None:
                raise ValueError(f"Agent {agent.agent_id} has no current_opinion_score")
            
            step_scores.append(agent.current_opinion_score)
            step_texts.append(agent.current_opinion_text)

        self.agent_ids = [agent.agent_id for agent in network.agents]
        self.score_trajectory.append(step_scores)
        self.text_trajectory.append(step_texts)

import time
import numpy as np

from agents import (
    AgentNetwork,
    AgentState,
    JudgeAgent,
    JudgeResult,
    NeighborState,
    ParticipantAgent,
    ParticipantResult,
)
from math_models import MathModel
from simulations.simulation_state import SimulationResult
from utils import Logger


class SimulationRunner:
    """
    Synchronous LLM simulation on an agent network

    The math model plays two roles here: its stopping criterion is applied to the judge
    scores (so both trajectories stop on the same rule), and its participant_prompt_fields
    are inserted into the participant prompt (so LLM agents see the same information as the formula)
    """
    def __init__(
        self,
        thesis: str,
        network: AgentNetwork,
        participant_agents: dict[int, ParticipantAgent],
        judge_agent: JudgeAgent,
        logger: Logger,
        math_model: MathModel,
        max_steps: int = 10,
        eps: float = 0.03
    ):
        if max_steps < 0:
            raise ValueError("max_steps has to be non-negative")

        if eps < 0:
            raise ValueError("eps has to be non-negative")

        if math_model.size != len(network.agents):
            raise ValueError(f"Math model is defined for {math_model.size} agents, but the network has {len(network.agents)}")

        self.thesis = thesis
        self.network = network.deepcopy()
        self.participant_agents = participant_agents
        self.judge_agent = judge_agent
        self.logger = logger
        self.math_model = math_model
        self.max_steps = max_steps
        self.eps = eps

        self.validate_participant_agents()

    def validate_participant_agents(self) -> None:
        """Ensure that participant_agents keys match the network agent IDs and that each ParticipantAgent's internal agent_id matches its key"""
        network_agent_ids = {agent.agent_id for agent in self.network.agents}
        participant_agent_ids = set(self.participant_agents.keys())

        missing_ids = network_agent_ids - participant_agent_ids
        extra_ids = participant_agent_ids - network_agent_ids

        if missing_ids:
            raise ValueError(f"Missing participant agents for IDs: {sorted(missing_ids)}")

        if extra_ids:
            raise ValueError(f"Participant agents not present in network: {sorted(extra_ids)}")

        for agent_id, participant_agent in self.participant_agents.items():
            if participant_agent.agent_id != agent_id:
                raise ValueError(f"Participant agent key does not match its internal agent_id: key={agent_id}, participant_agent.agent_id={participant_agent.agent_id}")

    def run(self) -> SimulationResult:
        """Run the simulation until convergence or max_steps is reached"""
        self.evaluate_initial_scores()

        result = SimulationResult(
            score_trajectory=[],
            text_trajectory=[],
            agent_ids=[agent.agent_id for agent in self.network.agents],
        )
        result.add_trajectory_step(self.network)

        if self.check_stop(previous_scores=None):
            result.converged = True
            result.convergence_step = 0
            result.total_steps = 0
            return result

        for step in range(1, self.max_steps + 1):
            previous_scores = self.current_scores()
            self.run_step(step)

            result.total_steps = step
            result.add_trajectory_step(self.network)

            if self.check_stop(previous_scores):
                result.converged = True
                result.convergence_step = step

                return result

        return result

    def evaluate_initial_scores(self) -> None:
        """Score initial textual opinions with the judge for agents that have no numeric score yet"""
        for agent in self.network.agents:
            if agent.current_opinion_score is not None:
                continue

            judge_result = self.run_judge(step=0, agent=agent, participant_text=agent.current_opinion_text)

            self.network.update_agent_opinion(
                agent_id=agent.agent_id,
                new_opinion_text=agent.current_opinion_text,
                new_opinion_score=judge_result.opinion_score,
            )

    def current_scores(self) -> np.ndarray:
        """Return the current opinion scores of all agents in the network as a numpy array"""
        scores = []

        for agent in self.network.agents:
            if agent.current_opinion_score is None:
                raise ValueError(f"Agent {agent.agent_id} has no current_opinion_score")

            scores.append(agent.current_opinion_score)

        return np.asarray(scores, dtype=float)

    def check_stop(self, previous_scores: np.ndarray | None) -> bool:
        """Apply the math model's stopping criterion to the judge scores"""
        return self.math_model.check_stop(previous_scores, self.current_scores(), self.eps)

    def run_step(self, step: int) -> None:
        """Run a single simulation step, updating all agents' opinions based on their neighbors and the judge's evaluation"""
        updates = {}

        for agent in self.network.agents:
            updates[agent.agent_id] = self.process_agent(agent, step)

        self.network.update_network_opinions(updates)

    def process_agent(self, agent: AgentState, step: int) -> tuple[str, float]:
        """Process a single agent's opinion update for the current step, returning the new opinion text and score"""
        participant_agent = self.participant_agents[agent.agent_id]
        neighbors = self.network.get_neighbors(agent.agent_id)
        agent_index = self.network.get_agent_index(agent.agent_id)
        model_fields = self.math_model.participant_prompt_fields(agent_index, agent)

        participant_result = self.run_participant(
            step=step,
            agent=agent,
            participant_agent=participant_agent,
            neighbors=neighbors,
            model_fields=model_fields,
        )

        judge_result = self.run_judge(
            step=step,
            agent=agent,
            participant_text=participant_result.response,
        )

        if judge_result.opinion_score is None:
            raise ValueError(f"Judge agent returned no opinion score for agent_id={agent.agent_id} at step={step}")

        return (participant_result.response, judge_result.opinion_score)

    def run_participant(
        self,
        step: int,
        agent: AgentState,
        participant_agent: ParticipantAgent,
        neighbors: list[NeighborState],
        model_fields: dict[str, str]
    ) -> ParticipantResult:
        """Run the participant agent to process neighbors' opinions and return the new opinion text"""
        start_time = time.perf_counter()
        participant_result = participant_agent.process_neighbors_opinions(
            thesis=self.thesis,
            current_opinion_text=agent.current_opinion_text,
            neighbors=neighbors,
            self_trust=agent.self_trust,
            model_name=self.math_model.name,
            model_fields=model_fields,
        )
        latency = time.perf_counter() - start_time

        self.logger.log_participant_response(
            step=step,
            agent_id=agent.agent_id,
            model_name=participant_agent.model_name,
            temperature=participant_agent.temperature,
            thesis=self.thesis,
            self_trust=agent.self_trust,
            current_opinion_text=agent.current_opinion_text,
            neighbors=neighbors,
            prompt=participant_result.prompt,
            response=participant_result.response,
            latency=latency,
        )

        return participant_result

    def run_judge(
        self,
        step: int,
        agent: AgentState,
        participant_text: str
    ) -> JudgeResult:
        """Run the judge agent to evaluate the participant's opinion text and return the opinion score"""
        start_time = time.perf_counter()
        judge_result = self.judge_agent.extract_opinion_score(
            thesis=self.thesis,
            participant_opinion_text=participant_text,
        )
        latency = time.perf_counter() - start_time

        if judge_result.opinion_score is None:
            raise ValueError(f"Judge agent returned no opinion score for agent_id={agent.agent_id} at step={step}")

        self.logger.log_judge_response(
            step=step,
            agent_id=agent.agent_id,
            model_name=self.judge_agent.model_name,
            temperature=self.judge_agent.temperature,
            thesis=self.thesis,
            self_trust=agent.self_trust,
            participant_text=participant_text,
            prompt=judge_result.prompt,
            raw_response=judge_result.raw_response,
            score=judge_result.opinion_score,
            latency=latency,
        )

        return judge_result

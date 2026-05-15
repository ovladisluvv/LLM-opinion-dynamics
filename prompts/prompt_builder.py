import yaml
from pathlib import Path

from agents.agent_state import NeighborState


class PromptBuilder():
    """Utility class to build prompts for participant and judge agents based on a YAML configuration file"""
    def __init__(self, config_path: str | Path = "prompts/config.yaml"):
        self.config_path = Path(config_path)

        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        if "prompts" not in self.config:
            raise ValueError('Config must contain a "prompts" mapping')

        self.prompts = self.config["prompts"]

    def build_neighbors_block(self, neighbors: list[NeighborState]) -> str:
        """Builds the block of neighbors' opinions for the participant prompt"""
        if not neighbors:
            return "No neighbor opinions are available"

        return "\n".join(
            f"Agent {neighbor.agent_id}\n"
            f"Your trust rating of that agent: {neighbor.weight:.4f}\n"
            f"Opinion: {neighbor.current_opinion_text}\n"
            for neighbor in neighbors
        )

    def build_participant_prompt(
        self,
        thesis: str,
        current_opinion_text: str,
        neighbors: list[NeighborState],
        self_trust: float
    ) -> str:
        """Builds the participant prompt by filling in the template with the provided information"""
        neighbors_block = self.build_neighbors_block(neighbors)
        prompt = self.prompts["participant_prompt"]

        return prompt.format(
            thesis=thesis,
            current_opinion_text=current_opinion_text,
            neighbors_block=neighbors_block,
            self_trust=f"{self_trust:.4f}"
        ).strip()

    def build_judge_prompt(self, thesis: str, participant_opinion: str) -> str:
        """Builds the judge prompt by filling in the template with the provided information"""
        prompt = self.prompts["judge_prompt"]

        return prompt.format(
            thesis=thesis,
            participant_opinion=participant_opinion
        ).strip()

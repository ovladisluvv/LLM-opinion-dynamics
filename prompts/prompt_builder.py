import re
from pathlib import Path
import yaml

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
        self.model_blocks: dict[str, str] = self.config.get("model_blocks") or {}

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

    def build_model_block(self, model_name: str, model_fields: dict[str, str]) -> str:
        """Builds the math-model-specific block of the participant prompt. Empty for models without a block"""
        template = self.model_blocks.get(model_name)

        if not template:
            return ""

        try:
            return template.format(**model_fields).rstrip()
        except KeyError as error:
            raise ValueError(
                f"Model block for {model_name!r} needs placeholder {error.args[0]!r}, "
                f"but the model provided only {sorted(model_fields)}"
            )

    def build_participant_prompt(
        self,
        thesis: str,
        current_opinion_text: str,
        neighbors: list[NeighborState],
        self_trust: float,
        model_name: str = "",
        model_fields: dict[str, str] | None = None
    ) -> str:
        """Builds the participant prompt by filling in the template with the provided information"""
        neighbors_block = self.build_neighbors_block(neighbors)
        model_block = self.build_model_block(model_name, model_fields or {})
        prompt = self.prompts["participant_prompt"]

        prompt = prompt.format(
            thesis=thesis,
            current_opinion_text=current_opinion_text,
            model_block=model_block,
            neighbors_block=neighbors_block,
            self_trust=f"{self_trust:.4f}"
        )

        return re.sub(r"\n{3,}", "\n\n", prompt).strip()

    def build_judge_prompt(self, thesis: str, participant_opinion: str) -> str:
        """Builds the judge prompt by filling in the template with the provided information"""
        prompt = self.prompts["judge_prompt"]

        return prompt.format(
            thesis=thesis,
            participant_opinion=participant_opinion
        ).strip()

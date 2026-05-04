import yaml
from pathlib import Path


class PromptBuilder():
    """Utility class to build prompts for participant and judge agents based on a YAML configuration file"""
    def __init__(self, config_path: str | Path = "prompts/config.yaml"):
        self.config_path = Path(config_path)

        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        if "prompts" not in self.config:
            raise ValueError('Config must contain a "prompts" mapping')

        self.prompts = self.config["prompts"]

    def build_neighbors_block(self, neighbor_opinions: list[str]) -> str:
        """Builds the block of neighbors' opinions for the participant prompt"""
        if not neighbor_opinions:
            return "No neighbor opinions are available"

        return "\n".join(f"{i + 1}. {opinion}" for i, opinion in enumerate(neighbor_opinions))

    def build_participant_prompt(
        self,
        thesis: str,
        current_opinion_text: str,
        neighbor_opinions: list[str],
        stance_description: str
    ) -> str:
        """Builds the participant prompt by filling in the template with the provided information"""
        neighbors_block = self.build_neighbors_block(neighbor_opinions)
        prompt = self.prompts["participant_prompt"]

        prompt = prompt.format(
            thesis=thesis,
            current_opinion_text=current_opinion_text,
            neighbors_block=neighbors_block,
            stance_description=stance_description
        ).strip()

        return prompt

    def build_judge_prompt(self, thesis: str, participant_opinion: str) -> str:
        """Builds the judge prompt by filling in the template with the provided information"""
        prompt = self.prompts["judge_prompt"]

        prompt = prompt.format(
            thesis=thesis,
            participant_opinion=participant_opinion
        ).strip()

        return prompt

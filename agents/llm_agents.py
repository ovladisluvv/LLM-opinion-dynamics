import re
from abc import ABC, abstractmethod

from prompts.prompt_builder import PromptBuilder


class BaseAgent(ABC):
    """Abstract base class for LLM agents used in the simulation"""
    def __init__(self, model_name: str, temperature: float = 0.2, prompt_builder: PromptBuilder | None = None):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_builder = prompt_builder or PromptBuilder()

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Abstract method to call the LLM API or local model to generate a response based on the prompt"""
        pass


class ParticipantAgent(BaseAgent):
    """Network node that updates its textual opinion under social influence from its neighbors"""
    def __init__(self, agent_id: int, model_name: str, temperature: float = 0.2, prompt_builder: PromptBuilder | None = None):
        super().__init__(model_name, temperature, prompt_builder)
        self.agent_id = agent_id

    def process_neighbors_opinions(
        self,
        thesis: str,
        current_opinion_text: str,
        neighbors_opinions: list[str],
        stance_description: str
    ) -> str:
        """Constructs the prompt and gets the updated opinion in text format"""
        prompt = self.prompt_builder.build_participant_prompt(
            thesis=thesis,
            current_opinion_text=current_opinion_text,
            neighbor_opinions=neighbors_opinions,
            stance_description=stance_description,
        )

        return self.generate_response(prompt)


class JudgeAgent(BaseAgent):
    """External evaluator agent that maps textual opinions to a continuous numerical opinion in the range [0.0, 1.0]"""
    def __init__(self, model_name: str, temperature: float = 0.0, prompt_builder: PromptBuilder | None = None):
        super().__init__(model_name, temperature=0.0, prompt_builder=prompt_builder)

    def extract_opinion_score(self, thesis: str, participant_text: str) -> float:
        """Evaluates the participant's text and returns a float score"""
        prompt = self.prompt_builder.build_judge_prompt(
            thesis=thesis,
            participant_opinion=participant_text
        )

        raw_response = self.generate_response(prompt)
        return self.parse_response(raw_response)

    def parse_response(self, response: str) -> float:
        """Helper method to extract a float number from the LLM's raw text response"""
        response = response.strip()

        if not response:
            raise ValueError("Judge returned an empty response")

        if not re.fullmatch(r"0(\.\d+)?|1(\.0+)?", response):
            raise ValueError(f"Judge failed to return a valid score. Raw response: {response}")

        score = max(0.0, min(1.0, float(response)))

        return score

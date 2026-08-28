import re

from prompts.prompt_builder import PromptBuilder
from agents.agent_state import ParticipantResult, JudgeResult, NeighborState
from agents.llm_client import LLMClient


class BaseAgent:
    """Base class for LLM agents used in the simulation. Delegates API calls to an injected LLM client"""
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        prompt_builder: PromptBuilder | None = None,
        client: LLMClient | None = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.client = client

    def generate_response(self, prompt: str) -> str:
        """Call the LLM API through the injected client to generate a response based on the prompt"""
        if self.client is None:
            raise ValueError(f"Agent for model {self.model_name} has no LLM client. Pass a client to the agent constructor")

        return self.client.generate(prompt, temperature=self.temperature)


class ParticipantAgent(BaseAgent):
    """Network node that updates its textual opinion under social influence from its neighbors"""
    def __init__(
        self,
        agent_id: int,
        model_name: str,
        temperature: float = 0.2,
        prompt_builder: PromptBuilder | None = None,
        client: LLMClient | None = None
    ):
        super().__init__(model_name, temperature, prompt_builder, client)
        self.agent_id = agent_id

    def process_neighbors_opinions(
        self,
        thesis: str,
        current_opinion_text: str,
        neighbors: list[NeighborState],
        self_trust: float
    ) -> ParticipantResult:
        """Constructs the prompt and gets the updated opinion in text format"""
        prompt = self.prompt_builder.build_participant_prompt(
            thesis=thesis,
            current_opinion_text=current_opinion_text,
            neighbors=neighbors,
            self_trust=self_trust,
        )

        result = ParticipantResult(
            prompt=prompt,
            response=self.generate_response(prompt)
        )

        return result


class JudgeAgent(BaseAgent):
    """External evaluator agent that maps textual opinions to a continuous numerical opinion in the range [0.0, 1.0]"""
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        prompt_builder: PromptBuilder | None = None,
        client: LLMClient | None = None
    ):
        super().__init__(model_name, temperature=temperature, prompt_builder=prompt_builder, client=client)

    def extract_opinion_score(self, thesis: str, participant_opinion_text: str) -> JudgeResult:
        """Evaluates the participant's text and returns a float score"""
        prompt = self.prompt_builder.build_judge_prompt(
            thesis=thesis,
            participant_opinion=participant_opinion_text
        )

        raw_response = self.generate_response(prompt)

        result = JudgeResult(
            prompt=prompt,
            raw_response=raw_response,
            opinion_score=self.parse_response(raw_response)
        )

        return result

    def parse_response(self, raw_response: str, eps: float = 1e-7) -> float:
        """Parse a LLM-judge raw text response to a float and validate it's in the range [0.0, 1.0]"""
        raw_response = raw_response.strip()

        if not raw_response:
            raise ValueError("Judge returned an empty response")

        if not re.fullmatch(r"0(\.\d+)?|1(\.0+)?", raw_response):
            raise ValueError(f"Judge failed to return a valid score. Raw response: {raw_response}")

        score = float(raw_response)

        if score < -eps or score - 1.0 > eps:
            raise ValueError(f"Judge returned a score out of bounds [0.0, 1.0]. Raw response: {raw_response}")
        
        return score

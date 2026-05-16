class ParticipantResult:
    def __init__(self, prompt: str, response: str):
        self.prompt = prompt
        self.response = response


class JudgeResult:
    def __init__(self, prompt: str, raw_response: str, opinion_score: float | None = None):
        self.prompt = prompt
        self.raw_response = raw_response
        self.opinion_score = opinion_score


class AgentState:
    def __init__(
        self,
        agent_id: int,
        initial_opinion_text: str,
        current_opinion_text: str,
        initial_opinion_score: float | None = None,
        current_opinion_score: float | None = None,
        self_trust: float = 1.0
    ):
        self.agent_id = agent_id
        self.initial_opinion_text = initial_opinion_text
        self.current_opinion_text = current_opinion_text
        self.initial_opinion_score = initial_opinion_score
        self.current_opinion_score = current_opinion_score
        self.self_trust = self_trust


class NeighborState:
    def __init__(
        self,
        agent_id: int,
        weight: float,
        current_opinion_text: str,
        current_opinion_score: float | None = None
    ):
        self.agent_id = agent_id
        self.weight = weight
        self.current_opinion_text = current_opinion_text
        self.current_opinion_score = current_opinion_score

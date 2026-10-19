from dataclasses import dataclass


@dataclass
class ParticipantResult:
    prompt: str
    response: str


@dataclass
class JudgeResult:
    prompt: str
    raw_response: str
    opinion_score: float | None = None


@dataclass
class AgentState:
    agent_id: int
    initial_opinion_text: str
    current_opinion_text: str
    initial_opinion_score: float | None = None
    current_opinion_score: float | None = None
    self_trust: float = 1.0


@dataclass
class NeighborState:
    agent_id: int
    weight: float
    current_opinion_text: str
    current_opinion_score: float | None = None

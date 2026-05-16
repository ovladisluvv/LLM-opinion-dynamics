import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.agent_state import NeighborState


class Logger:
    FIELDNAMES = [
        "timestamp",
        "experiment_id",
        "event_type",
        "step",
        "agent_id",
        "model_name",
        "temperature",
        "thesis",
        "self_trust",
        "current_opinion_text",
        "neighbors",
        "prompt",
        "response",
        "judge_raw_response",
        "judge_score"
    ]

    def __init__(self, path: str | Path | None = None, experiment_id: str | None = None):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_id = experiment_id or current_time

        if path is None:
            path = f"results/logs/experiment_{self.experiment_id}.csv"

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_row(self, row: dict[str, Any]) -> None:
        row = self.prepare_row(row)
        file_exists = self.path.exists()

        with open(self.path, "a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def prepare_row(self, row: dict[str, Any]) -> dict[str, str]:
        unknown_fields = set(row) - set(self.FIELDNAMES)
        if unknown_fields:
            raise ValueError(f"Unknown log fields: {sorted(unknown_fields)}")
        
        ready_row = {}

        for key, value in row.items():
            ready_row[key] = self.format_value(value)

        ready_row["experiment_id"] = self.experiment_id

        return ready_row
    
    def format_value(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, float):
            return f"{value:.4f}"

        if isinstance(value, list):
            return "\n---\n".join(self.format_value(item) for item in value)

        if isinstance(value, NeighborState):
            return self.format_neighbor(value)

        return str(value)
    
    def format_neighbor(self, neighbor: NeighborState) -> str:
        score = "" if neighbor.current_opinion_score is None else f"{neighbor.current_opinion_score:.4f}"
        
        return (
            f"agent_id={neighbor.agent_id}; "
            f"trust_rating={neighbor.weight:.4f}; "
            f"opinion_score={score}; "
            f"opinion_text={neighbor.current_opinion_text}"
        )

    def log_participant_response(
        self,
        step: int,
        agent_id: int,
        model_name: str,
        temperature: float,
        thesis: str,
        self_trust: float,
        current_opinion_text: str,
        neighbors: list[NeighborState],
        prompt: str,
        response: str,
    ) -> None:
        self.write_row({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": "participant_response",
            "step": step,
            "agent_id": agent_id,
            "model_name": model_name,
            "temperature": temperature,
            "thesis": thesis,
            "self_trust": self_trust,
            "current_opinion_text": current_opinion_text,
            "neighbors": neighbors,
            "prompt": prompt,
            "response": response,
            "judge_raw_response": "",
            "judge_score": "",
        })

    def log_judge_response(
        self,
        step: int,
        agent_id: int,
        model_name: str,
        temperature: float,
        thesis: str,
        self_trust: float,
        participant_text: str,
        prompt: str,
        raw_response: str,
        score: float,
    ) -> None:
        self.write_row({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": "judge_response",
            "step": step,
            "agent_id": agent_id,
            "model_name": model_name,
            "temperature": temperature,
            "thesis": thesis,
            "self_trust": self_trust,
            "current_opinion_text": participant_text,
            "neighbors": "",
            "prompt": prompt,
            "response": "",
            "judge_raw_response": raw_response,
            "judge_score": score,
        })

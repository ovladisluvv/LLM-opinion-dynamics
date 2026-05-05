import csv
from datetime import datetime
from pathlib import Path
from typing import Any


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
        "stance_description",
        "current_opinion_text",
        "neighbors_opinions",
        "prompt",
        "response",
        "judge_raw_response",
        "judge_score"
    ]

    def __init__(self, path: str | Path | None = None, experiment_id: str | None = None):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if experiment_id is None:
            experiment_id = current_time

        self.experiment_id = experiment_id

        if path is None:
            path = f"results/logs/experiment_{self.experiment_id}.csv"

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_row(self, row: dict[str, Any]) -> None:
        row = self.prepare_row(row)
        row["experiment_id"] = self.experiment_id   
        file_exists = self.path.exists()

        with open(self.path, "a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def prepare_row(self, row: dict[str, Any]) -> dict[str, str]:
        ready_row = {}

        for key, value in row.items():
            if isinstance(value, list):
                ready_row[key] = "\n---\n".join(map(str, value))
            elif value is None:
                ready_row[key] = ""
            else:
                ready_row[key] = str(value)

        return ready_row

    def log_participant_response(
        self,
        step: int,
        agent_id: int,
        model_name: str,
        temperature: float,
        thesis: str,
        stance_description: str,
        current_opinion_text: str,
        neighbors_opinions: list[str],
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
            "stance_description": stance_description,
            "current_opinion_text": current_opinion_text,
            "neighbors_opinions": neighbors_opinions,
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
            "stance_description": "",
            "current_opinion_text": participant_text,
            "neighbors_opinions": "",
            "prompt": prompt,
            "response": "",
            "judge_raw_response": raw_response,
            "judge_score": score,
        })

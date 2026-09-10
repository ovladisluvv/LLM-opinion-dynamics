import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.simulation_state import SimulationResult
from math_models.common import TrajectoryResult


def prompt_config_hash(path: str | Path) -> str:
    """SHA-256 hash of the prompt config file for reproducibility metadata"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_run_dir(results_dir: str | Path, experiment_id: str, run_index: int) -> Path:
    """Create the directory for one run without overwriting previous results"""
    run_dir = Path(results_dir) / experiment_id / f"run_{run_index:03d}"

    if run_dir.exists():
        raise ValueError(f"Run directory already exists, refusing to overwrite: {run_dir}")

    run_dir.mkdir(parents=True)

    return run_dir


def save_json(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_config_snapshot(run_dir: Path, config_path: str | Path) -> None:
    shutil.copyfile(config_path, run_dir / "config_snapshot.yaml")


def save_score_trajectory_csv(path: str | Path, trajectory: list[list[float]], agent_ids: list[int]) -> None:
    """Save a score trajectory as a CSV with one row per step and one column per agent"""
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step"] + [f"agent_{agent_id}" for agent_id in agent_ids])

        for step, scores in enumerate(trajectory):
            writer.writerow([step] + [f"{score:.6f}" for score in scores])


def save_llm_trajectories(run_dir: Path, llm_result: SimulationResult) -> None:
    save_score_trajectory_csv(
        run_dir / "llm_score_trajectory.csv",
        llm_result.score_trajectory,
        llm_result.agent_ids
    )

    text_steps = []
    for step, texts in enumerate(llm_result.text_trajectory):
        text_steps.append({
            "step": step,
            "opinions": {str(agent_id): text for agent_id, text in zip(llm_result.agent_ids, texts)},
        })

    save_json(run_dir / "llm_text_trajectory.json", {
        "agent_ids": llm_result.agent_ids,
        "steps": text_steps,
    })


def save_math_trajectory(run_dir: Path, math_result: TrajectoryResult, agent_ids: list[int]) -> None:
    """Save the math-model trajectory; the model name is recorded in metadata.json"""
    trajectory = [np.asarray(step).tolist() for step in math_result.trajectory]
    save_score_trajectory_csv(run_dir / "math_trajectory.csv", trajectory, agent_ids)


def save_run_status(run_dir: Path, status: str, error: str | None = None) -> None:
    """Record whether a run finished ("ok") or was aborted ("failed") and why"""
    save_json(run_dir / "run_status.json", {"status": status, "error": error})

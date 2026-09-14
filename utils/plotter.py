from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from math_models import TrajectoryResult
from simulations import SimulationResult


AGENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

MODEL_LABELS = {
    "degroot": "DeGroot",
    "friedkin_johnsen": "Friedkin-Johnsen",
    "fj": "Friedkin-Johnsen"
}


def model_label(model_name: str) -> str:
    """Return a human-readable label for the given model name"""
    return MODEL_LABELS.get(model_name, model_name)


def plot_opinion_trajectories(
    llm_result: SimulationResult,
    math_result: TrajectoryResult,
    model_name: str,
    experiment_id: str,
    run_id: str,
    output_path: str | Path
) -> Path:
    """Plot LLM (solid) and math-model (dashed) score trajectories per agent and save the figure as PNG"""
    llm_traj = np.asarray(llm_result.score_trajectory, dtype=float)
    math_traj = np.asarray(math_result.trajectory, dtype=float)
    label = model_label(model_name)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, agent_id in enumerate(llm_result.agent_ids):
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
        ax.plot(
            range(llm_traj.shape[0]),
            llm_traj[:, idx],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"Agent {agent_id} (LLM)"
        )
        ax.plot(
            range(math_traj.shape[0]),
            math_traj[:, idx],
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.7,
            label=f"Agent {agent_id} ({label})"
        )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Step")
    ax.set_ylabel("Opinion score (agreement with thesis)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"LLM vs {label} opinion trajectories\nexperiment {experiment_id}, {run_id}")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path

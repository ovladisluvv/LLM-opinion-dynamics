import numpy as np

from math_models import TrajectoryResult
from simulations.simulation_state import SimulationResult


def compare_trajectories(llm_result: SimulationResult, math_result: TrajectoryResult) -> dict:
    """
    Compare LLM and math-model score trajectories on the same network

    The trajectories may have different lengths because either side can stop
    earlier. Errors are computed only over the common prefix of both
    trajectories (steps 0..aligned_steps); no missing values are invented
    """
    llm_trajectory = np.asarray(llm_result.score_trajectory, dtype=float)
    math_trajectory = np.asarray(math_result.trajectory, dtype=float)

    if llm_trajectory.size == 0 or math_trajectory.size == 0:
        raise ValueError("Both trajectories have to contain at least the initial step")

    if llm_trajectory.shape[1] != math_trajectory.shape[1]:
        raise ValueError(
            f"Trajectories have to cover the same agents. "
            f"Got {llm_trajectory.shape[1]} and {math_trajectory.shape[1]} agents"
        )

    common_length = min(llm_trajectory.shape[0], math_trajectory.shape[0])
    llm_common = llm_trajectory[:common_length]
    math_common = math_trajectory[:common_length]

    abs_error = np.abs(llm_common - math_common)

    return {
        "alignment": "common_prefix",
        "aligned_steps": common_length - 1,
        "llm_total_steps": llm_trajectory.shape[0] - 1,
        "math_total_steps": math_trajectory.shape[0] - 1,
        "abs_error_per_step": abs_error.tolist(),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(((llm_common - math_common) ** 2).mean())),
        "llm_spread_per_step": np.ptp(llm_trajectory, axis=1).tolist(),
        "math_spread_per_step": np.ptp(math_trajectory, axis=1).tolist(),
        "llm_variance_per_step": llm_trajectory.var(axis=1).tolist(),
        "math_variance_per_step": math_trajectory.var(axis=1).tolist(),
        "llm_converged": llm_result.converged,
        "llm_convergence_step": llm_result.convergence_step,
        "math_converged": math_result.converged,
        "math_convergence_step": math_result.convergence_step,
    }


def aggregate_run_metrics(run_metrics: list[dict]) -> dict:
    """Aggregate mean/std of the main scalar metrics over several independent runs"""
    if not run_metrics:
        raise ValueError("run_metrics has to be non-empty")

    summary: dict[str, float | int] = {"num_runs": len(run_metrics)}

    for key in ("mae", "rmse", "llm_total_steps", "math_total_steps"):
        values = np.asarray([metrics[key] for metrics in run_metrics], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())

    for key in ("llm_converged", "math_converged"):
        summary[f"{key}_count"] = sum(1 for metrics in run_metrics if metrics[key])

    return summary

import argparse
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np

from agents import (
    AgentNetwork,
    AgentState,
    JudgeAgent,
    LLMClientError,
    LLMSettings,
    ParticipantAgent,
    create_llm_client,
)
from config import ExperimentConfig, LLMRoleConfig, load_experiment_config, AgentPromptBuilder
from math_models import MathModel, TrajectoryResult, create_math_model
from simulations import (
    SimulationResult,
    SimulationRunner,
    aggregate_run_metrics,
    compare_trajectories,
)
from utils import (
    Logger,
    create_run_dir,
    load_env_file,
    plot_opinion_trajectories,
    prompt_config_hash,
    save_config_snapshot,
    save_json,
    save_llm_trajectories,
    save_math_trajectory,
    save_run_status,
)


class ExperimentOutcome:
    """Result of running an experiment, including the directory where results are saved and any failed runs"""
    def __init__(self, experiment_dir: Path, failed_runs: list[str] | None = None):
        self.experiment_dir = experiment_dir
        self.failed_runs = failed_runs if failed_runs is not None else []


def build_network(config: ExperimentConfig) -> AgentNetwork:
    """Create a fresh agent network from the experiment config"""
    agents = [
        AgentState(
            agent_id=agent_config.agent_id,
            initial_opinion_text=agent_config.initial_opinion_text,
            current_opinion_text=agent_config.initial_opinion_text,
            initial_opinion_score=agent_config.initial_opinion_score,
            current_opinion_score=agent_config.initial_opinion_score,
        )
        for agent_config in config.agents
    ]

    return AgentNetwork(agents=agents, weights=config.weights)


def build_client(config: ExperimentConfig, role: LLMRoleConfig):
    """Resolve credentials for the role's provider from the environment and create its client"""
    settings = LLMSettings.from_env(role.provider, role.model)

    return create_llm_client(
        settings=settings,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
        retry_delay=config.retry_delay,
    )


def build_agents(
    config: ExperimentConfig,
    run_seed: int,
    agent_prompt_builder: AgentPromptBuilder
) -> tuple[dict[int, ParticipantAgent], JudgeAgent]:
    """Create participant agents and the judge agent with their own LLM clients"""
    participant_client = build_client(config, config.participant_llm)
    judge_client = build_client(config, config.judge_llm)

    participant_params = config.participant_llm.generation_params(seed=run_seed)
    judge_params = config.judge_llm.generation_params(seed=run_seed)

    participant_agents = {
        agent_config.agent_id: ParticipantAgent(
            agent_id=agent_config.agent_id,
            model_name=config.participant_llm.model,
            params=participant_params,
            agent_prompt_builder=agent_prompt_builder,
            client=participant_client,
        )
        for agent_config in config.agents
    }

    judge_agent = JudgeAgent(
        model_name=config.judge_llm.model,
        params=judge_params,
        agent_prompt_builder=agent_prompt_builder,
        client=judge_client,
    )

    return participant_agents, judge_agent


def role_metadata(role: LLMRoleConfig, run_seed: int) -> dict:
    """Collect metadata for a role's LLM client, excluding any sensitive tokens"""
    return {**role.to_dict(), "seed": run_seed}


def build_metadata(
    config: ExperimentConfig,
    experiment_id: str,
    run_id: str,
    run_seed: int,
    math_model: MathModel,
    prompt_hash: str,
    llm_result: SimulationResult,
    math_result: TrajectoryResult
) -> dict:
    """Collect run metadata for reproducibility. Tokens are never included"""
    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "random_seed": run_seed,
        "thesis": config.thesis,
        "agent_ids": [agent_config.agent_id for agent_config in config.agents],
        "llm": {
            "participant": role_metadata(config.participant_llm, run_seed),
            "judge": role_metadata(config.judge_llm, run_seed),
        },
        "math_model": math_model.describe(),
        "request_timeout": config.request_timeout,
        "max_retries": config.max_retries,
        "retry_delay": config.retry_delay,
        "convergence_epsilon": config.convergence_epsilon,
        "math_model_epsilon": config.math_model_epsilon,
        "network_weights": config.weights.tolist(),
        "llm_total_steps": llm_result.total_steps,
        "llm_converged": llm_result.converged,
        "llm_convergence_step": llm_result.convergence_step,
        "math_total_steps": math_result.total_steps,
        "math_converged": math_result.converged,
        "math_convergence_step": math_result.convergence_step,
        "prompt_config_sha256": prompt_hash,
        "resolved_config": config.to_dict(),
    }


def run_single(
    config: ExperimentConfig,
    config_path: str | Path,
    experiment_id: str,
    run_index: int,
    agent_prompt_builder: AgentPromptBuilder
) -> dict:
    """Execute one independent run: LLM simulation, math-model baseline, comparison, and saving"""
    run_id = f"run_{run_index:03d}"
    run_seed = config.random_seed + run_index

    random.seed(run_seed)
    np.random.seed(run_seed)

    run_dir = create_run_dir(config.results_dir, experiment_id, run_index)
    logger = Logger(path=run_dir / "events.csv", experiment_id=experiment_id, run_id=run_id)

    math_model = create_math_model(config.math_model_name, config.weights, config.math_model_params)
    participant_agents, judge_agent = build_agents(config, run_seed, agent_prompt_builder)
    network = build_network(config)

    runner = SimulationRunner(
        thesis=config.thesis,
        network=network,
        participant_agents=participant_agents,
        judge_agent=judge_agent,
        logger=logger,
        math_model=math_model,
        max_steps=config.max_steps,
        eps=config.convergence_epsilon,
    )

    llm_result = runner.run()

    initial_scores = np.asarray(llm_result.score_trajectory[0], dtype=float)
    math_result = math_model.simulate(initial_scores, max_steps=config.max_steps, eps=config.math_model_epsilon)

    metrics = compare_trajectories(llm_result, math_result)
    prompt_hash = prompt_config_hash(agent_prompt_builder.config_path)
    metadata = build_metadata(
        config=config,
        experiment_id=experiment_id,
        run_id=run_id,
        run_seed=run_seed,
        math_model=math_model,
        prompt_hash=prompt_hash,
        llm_result=llm_result,
        math_result=math_result,
    )

    save_json(run_dir / "metadata.json", metadata)
    save_config_snapshot(run_dir, config_path)
    save_llm_trajectories(run_dir, llm_result)
    save_math_trajectory(run_dir, math_result, llm_result.agent_ids)
    save_json(run_dir / "comparison_metrics.json", metrics)
    plot_opinion_trajectories(
        llm_result=llm_result,
        math_result=math_result,
        model_name=math_model.name,
        experiment_id=experiment_id,
        run_id=run_id,
        output_path=run_dir / "opinion_trajectories.png",
    )
    save_run_status(run_dir, "ok")

    print(f"{run_id}: LLM steps={llm_result.total_steps} "
          f"(converged={llm_result.converged}), "
          f"{math_model.name} steps={math_result.total_steps} "
          f"(converged={math_result.converged}), "
          f"MAE={metrics['mae']:.4f}, results in {run_dir}")

    return metrics


def run_experiment(config_path: str | Path, env_path: str | Path = ".env") -> ExperimentOutcome:
    """
    Run the full experiment described by the config file

    A run that fails (invalid judge output, exhausted API retries) is recorded in its
    run_status.json and in summary.json. The remaining runs still execute
    """
    load_env_file(env_path)

    config = load_experiment_config(config_path)
    agent_prompt_builder = AgentPromptBuilder()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_id = f"{config.experiment_name}_{timestamp}"
    experiment_dir = Path(config.results_dir) / experiment_id

    print(f"Experiment {experiment_id}: {config.num_runs} run(s), {len(config.agents)} agents, max {config.max_steps} steps, math model: {config.math_model_name}")
    print(f"Participant model: {config.participant_llm.provider}/{config.participant_llm.model}, judge model: {config.judge_llm.provider}/{config.judge_llm.model}")

    run_metrics = []
    failed_runs = []
    for run_index in range(config.num_runs):
        run_id = f"run_{run_index:03d}"

        try:
            metrics = run_single(
                config=config,
                config_path=config_path,
                experiment_id=experiment_id,
                run_index=run_index,
                agent_prompt_builder=agent_prompt_builder,
            )
        except (ValueError, LLMClientError) as error:
            failed_runs.append(run_id)
            save_run_status(experiment_dir / run_id, "failed", error=f"{type(error).__name__}: {error}")
            print(f"{run_id}: FAILED: {error}")
            traceback.print_exc()
            continue

        run_metrics.append(metrics)

    summary: dict[str, Any] = aggregate_run_metrics(run_metrics) if run_metrics else {"num_runs": 0}
    summary["failed_runs"] = failed_runs
    save_json(experiment_dir / "summary.json", summary)

    if run_metrics:
        print(f"Summary over {len(run_metrics)} successful run(s): "
              f"MAE={summary['mae_mean']:.4f}±{summary['mae_std']:.4f}, "
              f"RMSE={summary['rmse_mean']:.4f}±{summary['rmse_std']:.4f}")

    if failed_runs:
        print(f"Failed runs: {failed_runs}")

    print(f"All results saved to {experiment_dir}")

    return ExperimentOutcome(experiment_dir=experiment_dir, failed_runs=failed_runs)


def main() -> None:
    """Parse command-line arguments and run the experiment"""
    parser = argparse.ArgumentParser(description="Run an LLM opinion dynamics experiment against a mathematical model")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config")
    parser.add_argument("--env", default=".env", help="Path to the .env file with provider tokens and endpoints")

    args = parser.parse_args()

    try:
        outcome = run_experiment(config_path=args.config, env_path=args.env)
    except (ValueError, LLMClientError) as error:
        raise SystemExit(f"Error: {error}")

    if outcome.failed_runs:
        raise SystemExit(f"Error: {len(outcome.failed_runs)} run(s) failed: {outcome.failed_runs}")


if __name__ == "__main__":
    main()

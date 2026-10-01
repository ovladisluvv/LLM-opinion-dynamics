import argparse
import os

from agents import LLMClientError
from experiments import run_experiment
from utils import load_env_file

# Environment variable with the experiment YAML path, used when --config is not passed
CONFIG_ENV_VAR = "EXPERIMENT_CONFIG"


parser = argparse.ArgumentParser(description="Run an LLM opinion dynamics experiment against a mathematical model")
parser.add_argument("--config", help=f"Path to the experiment YAML config, overrides {CONFIG_ENV_VAR} from .env")
parser.add_argument("--env", default=".env", help="Path to the .env file with provider tokens, endpoints and the config path")

args = parser.parse_args()

load_env_file(args.env)
config_path = args.config or os.environ.get(CONFIG_ENV_VAR, "").strip()

if not config_path:
    raise SystemExit(f"Error: no experiment config given. Pass --config or set {CONFIG_ENV_VAR} in {args.env}")

try:
    outcome = run_experiment(config_path=config_path, env_path=args.env)
except (ValueError, LLMClientError) as error:
    raise SystemExit(f"Error: {error}")

if outcome.failed_runs:
    raise SystemExit(f"Error: {len(outcome.failed_runs)} run(s) failed: {outcome.failed_runs}")

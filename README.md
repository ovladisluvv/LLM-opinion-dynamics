# LLM-opinion-dynamics
## Modeling opinion dynamics in social networks with LLM-based agents

LLM-opinion-dynamics runs a discussion between LLM agents on a weighted social network, turns their
textual opinions into numeric scores with a separate judge model, and compares the resulting trajectory with a classical opinion dynamics model (DeGroot, Friedkin-Johnsen) run on the same network. The research question behind it: for which parameters (temperature, the LLM itself, prompt design) does the LLM dynamics reproduce a given formal model

> **Note.** External LLM APIs are not fully deterministic even with fixed temperatures and seeds. Instead of claiming determinism, every run records its parameters, seed, exact prompts and raw responses, and experiments can be repeated over several independent runs

## Features

### Simulation
* Participant agents update their opinions in text, reading their neighbors' opinions together with the influence weights from a row-stochastic matrix `W` (`W[i][i]` is the agent's self-trust)
* A separate judge model scores every opinion from `0.0` (disagrees with the thesis) to `1.0` (agrees). Anything other than a bare number is an error - failed calls are never replaced with a neutral score
* Synchronous steps, like `x(t + 1) = W x(t)`: the whole network is updated only after every agent has answered, so nobody sees a half-updated state
* Participant and judge are configured independently: different providers, models and temperatures

### Mathematical models
* **DeGroot** - `x(t + 1) = W x(t)`, stops at consensus ($\max_i x_i(t) - \min_i x_i(t) \leqslant \varepsilon$)
* **Friedkin-Johnsen** - `x(t + 1) = Λ W x(t) + (I - Λ) x(0)`, stops at stationarity ($\max_i |x_i(t + 1) - x_i(t)| \leqslant \varepsilon$)
  * LLM agents are also reminded of their initial opinion and how strongly they are anchored to it
* The LLM simulation stops on the same criterion as the chosen model, and the model starts from the judge's step-0 scores
* New models are added by subclassing `MathModel` and registering it in `math_models/factory.py`

### Results
* MAE and RMSE between the LLM and model trajectories, spread, variance, convergence steps
* A plot of both trajectories
* Each run in its own directory: metadata, config snapshot, CSV log of every LLM call, trajectories, metrics. A failed run does not stop the others, and `summary.json` aggregates all runs

## Project architecture

* **`agents`** - the LLM side of the simulation:
  * `agent_state.py` - agent, neighbors, participant result and judge result data classes
  * `agent_network.py` - agents plus the weight matrix: neighbor lookup and synchronous update
  * `llm_agents.py` - participant agent and judge agent with strict score parsing
  * `llm_client.py` - chat-completions client with retries, provider registry, credentials from the environment
* **`math_models`** - classical models without any LLM:
  * `common.py` - `MathModel` base class, weight validation, consensus and stationarity checks
  * `degroot.py`, `friedkin_johnsen.py` - the built-in models, each runnable as a standalone demo
  * `factory.py` - model registry used by the experiment config
* **`simulations`** - the LLM simulation and its comparison with the model:
  * `simulation_runner.py` - scores initial opinions missing a score with the judge, runs synchronous steps and stops on the math model's criterion
  * `simulation_state.py` - judge score and text trajectories of one run with its convergence info
  * `result_comparator.py` - MAE, RMSE, spread and variance over the common prefix of both trajectories, mean and std over several runs
* **`config`** - experiment and prompt configuration:
  * `config_loader.py` - loads and validates the experiment YAML: agents, weights, LLM roles, math model and its parameters. Rejects keys that look like tokens
  * `agent_prompt_config.yaml` - participant and judge prompt templates and model-specific blocks
  * `agent_prompt_builder.py` - fills the templates with the thesis, opinions, neighbors and the model block
  * `example_experiment_config_degroot.yaml`, `example_experiment_config_fj.yaml` - example experiments
* **`utils`** - supporting code:
  * `env_loader.py` - loads `.env` into the environment without overriding already set variables
  * `logger.py` - CSV log with one row per LLM call: prompt, response, judge output, score, latency
  * `plotter.py` - plot of the LLM (solid) and model (dashed) trajectories
  * `result_saver.py` - run directories, trajectories, metadata, config snapshot and prompt config hash
* **`experiments/experiment_runner.py`** - experiment pipeline: LLM runs, model baseline, comparison, saving
* **`main.py`** - command-line entry point: resolves the experiment config from `--config` or `EXPERIMENT_CONFIG` and runs it

## Installation and running

Dependencies (numpy, PyYAML, matplotlib) are listed in `requirements.txt`:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the credentials for the providers used in the experiment
(`openai`, `openai_compatible`, `ollama`, `vllm`) and, optionally, `EXPERIMENT_CONFIG` - the experiment
to run by default. Tokens stay in `.env` only and are never saved to logs or results

The experiment itself (thesis, agents, weights, LLMs and their temperatures, math model, number of runs) is
described in a YAML file - see `config/example_experiment_config_degroot.yaml` and
`config/example_experiment_config_fj.yaml`

Running from the repo root:
```bash
python main.py                      # experiment from EXPERIMENT_CONFIG
python main.py --config <path.yaml> # or an explicit one
python main.py --env <path.env>     # a different .env file
```

Results are saved to `results/<experiment_name>_<timestamp>/run_NNN/`

The mathematical models also run on their own, without any API:
```bash
python -m math_models.degroot
python -m math_models.friedkin_johnsen
```

## Requirements

* Python 3.11+
* Access to an OpenAI-compatible chat-completions API or a local server (Ollama, vLLM, llama.cpp, LM Studio)

## Licence

LLM-opinion-dynamics is distributed under the MIT licence - see `LICENCE`

## Author

Vladislav Ogai ([ovladisluvv](https://github.com/ovladisluvv)), 2026

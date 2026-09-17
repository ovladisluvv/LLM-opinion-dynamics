from pathlib import Path
import numpy as np
import yaml

from agents import DEFAULT_PROVIDER, GenerationParams, get_provider_spec
from math_models import create_math_model, validate_weights

# YAML keys whose name contains one of these fragments has to stay in .env
SECRET_KEY_FRAGMENTS = ("token", "api_key", "secret")


class AgentConfig:
    """Initial configuration of one participant agent"""
    def __init__(
        self,
        agent_id: int,
        initial_opinion_text: str,
        initial_opinion_score: float | None = None,
        persona: str | None = None
    ):
        if not isinstance(agent_id, int) or isinstance(agent_id, bool):
            raise ValueError(f"agent_id has to be an integer. Got: {agent_id!r}")

        if not initial_opinion_text or not initial_opinion_text.strip():
            raise ValueError(f"Agent {agent_id} has an empty initial_opinion_text")

        if initial_opinion_score is not None and not 0.0 <= initial_opinion_score <= 1.0:
            raise ValueError(f"Agent {agent_id} has initial_opinion_score outside [0.0, 1.0]: {initial_opinion_score}")

        self.agent_id = agent_id
        self.initial_opinion_text = initial_opinion_text.strip()
        self.initial_opinion_score = initial_opinion_score
        self.persona = persona

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "initial_opinion_text": self.initial_opinion_text,
            "initial_opinion_score": self.initial_opinion_score,
            "persona": self.persona,
        }


class LLMRoleConfig:
    """Model choice and sampling settings of one LLM role (participant or judge). Contains no credentials"""
    def __init__(
        self,
        model: str,
        provider: str = DEFAULT_PROVIDER,
        temperature: float = 0.2,
        max_tokens: int | None = None
    ):
        if not model or not str(model).strip():
            raise ValueError("LLM role has to specify a non-empty model name")

        get_provider_spec(provider)

        self.model = str(model).strip()
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        # validates temperature and max_tokens
        self.generation_params(seed=None)

    def generation_params(self, seed: int | None) -> GenerationParams:
        return GenerationParams(temperature=self.temperature, max_tokens=self.max_tokens, seed=seed)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class ExperimentConfig:
    """Validated experiment configuration loaded from a YAML file. Contains no tokens"""
    def __init__(
        self,
        thesis: str,
        agents: list[AgentConfig],
        weights: list[list[float]],
        participant_llm: LLMRoleConfig,
        judge_llm: LLMRoleConfig,
        math_model_name: str = "degroot",
        math_model_params: dict | None = None,
        math_model_epsilon: float | None = None,
        max_steps: int = 10,
        convergence_epsilon: float = 0.03,
        random_seed: int = 0,
        num_runs: int = 1,
        request_timeout: float = 60.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        results_dir: str = "results",
        experiment_name: str = "experiment"
    ):
        if not thesis or not thesis.strip():
            raise ValueError("thesis has to be a non-empty string")

        if not agents:
            raise ValueError("agents list has to be non-empty")

        agent_ids = [agent.agent_id for agent in agents]
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError(f"Agent IDs have to be unique. Got: {agent_ids}")

        weights_array = np.asarray(weights, dtype=float)
        validate_weights(weights_array)

        if weights_array.shape[0] != len(agents):
            raise ValueError(f"The number of agents has to match the weight matrix size. Got {len(agents)} agents and weight matrix of shape {weights_array.shape}")

        if max_steps < 1:
            raise ValueError("max_steps has to be at least 1")

        if convergence_epsilon < 0:
            raise ValueError("convergence_epsilon has to be non-negative")

        if math_model_epsilon is not None and math_model_epsilon < 0:
            raise ValueError("math_model.epsilon has to be non-negative")

        if num_runs < 1:
            raise ValueError("num_runs has to be at least 1")

        math_model_params = dict(math_model_params or {})
        # fail early on an unknown model or invalid parameters
        create_math_model(math_model_name, weights_array, math_model_params)

        self.thesis = thesis.strip()
        self.agents = agents
        self.weights = weights_array
        self.participant_llm = participant_llm
        self.judge_llm = judge_llm
        self.math_model_name = math_model_name
        self.math_model_params = math_model_params
        self.math_model_epsilon = math_model_epsilon if math_model_epsilon is not None else convergence_epsilon
        self.max_steps = max_steps
        self.convergence_epsilon = convergence_epsilon
        self.random_seed = random_seed
        self.num_runs = num_runs
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.results_dir = results_dir
        self.experiment_name = experiment_name

    def to_dict(self) -> dict:
        """Resolved config as a JSON-serializable dict for experiment metadata"""
        return {
            "thesis": self.thesis,
            "agents": [agent.to_dict() for agent in self.agents],
            "weights": self.weights.tolist(),
            "llm": {
                "participant": self.participant_llm.to_dict(),
                "judge": self.judge_llm.to_dict(),
            },
            "math_model": {
                "name": self.math_model_name,
                "params": self.math_model_params,
                "epsilon": self.math_model_epsilon,
            },
            "max_steps": self.max_steps,
            "convergence_epsilon": self.convergence_epsilon,
            "random_seed": self.random_seed,
            "num_runs": self.num_runs,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "results_dir": self.results_dir,
            "experiment_name": self.experiment_name,
        }


def find_secret_keys(data, path: str = "") -> list[str]:
    """Recursively collect YAML keys that look like credentials"""
    found = []

    if isinstance(data, dict):
        for key, value in data.items():
            key_path = f"{path}.{key}" if path else str(key)
            lowered = str(key).lower()

            if lowered != "max_tokens" and any(fragment in lowered for fragment in SECRET_KEY_FRAGMENTS):
                found.append(key_path)

            found.extend(find_secret_keys(value, key_path))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            found.extend(find_secret_keys(item, f"{path}[{index}]"))

    return found


def parse_llm_role(raw_llm: dict, role: str, default_temperature: float) -> LLMRoleConfig:
    raw_role = raw_llm.get(role)

    if not isinstance(raw_role, dict):
        raise ValueError(f'Experiment config has to contain an "llm.{role}" mapping with at least a "model" key')

    raw_role = {"temperature": default_temperature, **raw_role}

    try:
        return LLMRoleConfig(**raw_role)
    except TypeError as error:
        raise ValueError(f"Invalid llm.{role} field: {error}")


def parse_math_model(raw_math_model) -> dict:
    if raw_math_model is None:
        return {}

    if not isinstance(raw_math_model, dict):
        raise ValueError(f'"math_model" has to be a mapping. Got: {raw_math_model!r}')

    allowed = {"name", "params", "epsilon"}
    unknown = sorted(set(raw_math_model) - allowed)
    if unknown:
        raise ValueError(f"Unknown math_model fields: {unknown}. Allowed: {sorted(allowed)}")

    params = raw_math_model.get("params")
    if params is not None and not isinstance(params, dict):
        raise ValueError(f'"math_model.params" has to be a mapping. Got: {params!r}')

    resolved = {}
    if "name" in raw_math_model:
        resolved["math_model_name"] = raw_math_model["name"]
    if params is not None:
        resolved["math_model_params"] = params
    if "epsilon" in raw_math_model:
        resolved["math_model_epsilon"] = raw_math_model["epsilon"]

    return resolved


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file"""
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Experiment config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Experiment config has to be a YAML mapping. Got: {type(data).__name__}")

    secret_keys = find_secret_keys(data)
    if secret_keys:
        raise ValueError(f"Tokens and API keys belong in .env, not in the experiment config. Remove keys: {secret_keys}")

    raw_agents = data.pop("agents", None)
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ValueError('Experiment config has to contain a non-empty "agents" list')

    agents = []
    for raw_agent in raw_agents:
        if not isinstance(raw_agent, dict):
            raise ValueError(f"Each agent entry has to be a mapping. Got: {raw_agent!r}")

        try:
            agents.append(AgentConfig(**raw_agent))
        except TypeError as error:
            raise ValueError(f"Invalid agent field: {error}")

    raw_llm = data.pop("llm", None)
    if not isinstance(raw_llm, dict):
        raise ValueError('Experiment config has to contain an "llm" mapping with "participant" and "judge" roles')

    unknown_roles = sorted(set(raw_llm) - {"participant", "judge"})
    if unknown_roles:
        raise ValueError(f"Unknown llm roles: {unknown_roles}. Allowed: ['judge', 'participant']")

    participant_llm = parse_llm_role(raw_llm, "participant", default_temperature=0.2)
    judge_llm = parse_llm_role(raw_llm, "judge", default_temperature=0.0)

    data.update(parse_math_model(data.pop("math_model", None)))

    try:
        return ExperimentConfig(agents=agents, participant_llm=participant_llm, judge_llm=judge_llm, **data)
    except TypeError as error:
        raise ValueError(f"Invalid experiment config field: {error}")

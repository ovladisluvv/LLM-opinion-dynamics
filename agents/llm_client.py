import json
import os
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class LLMClientError(Exception):
    """Raised when an LLM API call fails after all retries or returns an invalid payload"""
    pass


class GenerationParams:
    """Sampling parameters of one LLM call. Kept separate from the client so they can be swept per experiment"""
    def __init__(self, temperature: float, max_tokens: int | None = None, seed: int | None = None, top_p: float | None = None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.top_p = top_p

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature has to be non-negative")

        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens has to be positive when set")

        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p has to be in (0, 1] when set")


class ProviderSpec:
    """How a provider name maps to a client class and to its connection defaults"""
    def __init__(self, client_class: type["LLMClient"], default_base_url: str | None, requires_token: bool):
        self.client_class = client_class
        self.default_base_url = default_base_url
        self.requires_token = requires_token


class LLMSettings:
    """Connection settings for one LLM role: model name plus provider credentials from the environment"""
    def __init__(
        self,
        model_name: str,
        provider: str,
        token: str | None = None,
        base_url: str | None = None
    ):
        if not model_name:
            raise ValueError("model_name has to be a non-empty string")

        if not provider:
            raise ValueError("provider has to be a non-empty string")

        self.model_name = model_name
        self.provider = provider
        self.token = token or None
        self.base_url = base_url or None

    @classmethod
    def from_env(cls, provider: str, model_name: str) -> "LLMSettings":
        """
        Resolve credentials for a provider from {PROVIDER}_LLM_TOKEN and {PROVIDER}_LLM_BASE_URL,
        e.g. OPENAI_LLM_TOKEN or OLLAMA_LLM_BASE_URL. The base URL falls back to the provider default
        """
        spec = get_provider_spec(provider)
        prefix = f"{provider.upper()}_LLM"

        token = os.environ.get(f"{prefix}_TOKEN", "").strip() or None
        base_url = os.environ.get(f"{prefix}_BASE_URL", "").strip() or spec.default_base_url

        if spec.requires_token and token is None:
            raise ValueError(f"Missing required environment variable {prefix}_TOKEN for provider {provider!r}. Set it in .env (see .env.example)")

        if base_url is None:
            raise ValueError(f"Provider {provider!r} has no default endpoint. Set {prefix}_BASE_URL in .env")

        return cls(model_name=model_name, provider=provider, token=token, base_url=base_url)


class LLMClient(ABC):
    """Provider-independent LLM client with a retry loop around a single abstract request method"""
    def __init__(
        self,
        settings: LLMSettings,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ):
        if timeout <= 0:
            raise ValueError("timeout has to be positive")

        if max_retries < 0:
            raise ValueError("max_retries has to be non-negative")

        if retry_delay < 0:
            raise ValueError("retry_delay has to be non-negative")

        self.settings = settings
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    def generate(self, prompt: str, params: GenerationParams) -> str:
        """Send the prompt to the model, retrying failed requests, and return the response text"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return self.send_request(prompt, params)
            except LLMClientError as error:
                last_error = error

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        raise LLMClientError(
            f"LLM request for model {self.settings.model_name} failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    @abstractmethod
    def send_request(self, prompt: str, params: GenerationParams) -> str:
        """Perform one API request and return the response text. Raises LLMClientError on failure"""
        pass


class OpenAICompatibleClient(LLMClient):
    """
    Client for any endpoint speaking the OpenAI chat completions protocol:
    OpenAI itself, Ollama, vLLM, llama.cpp server, LM Studio and similar local servers
    """
    def build_request(self, prompt: str, params: GenerationParams) -> tuple[str, dict[str, str], dict]:
        """Return the URL, headers and JSON body of one chat completion request"""
        url = self.settings.base_url.rstrip("/") + "/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.settings.token:
            headers["Authorization"] = f"Bearer {self.settings.token}"

        body = {
            "model": self.settings.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.temperature,
        }

        for field in ("max_tokens", "seed", "top_p"):
            value = getattr(params, field)
            if value is not None:
                body[field] = value

        return url, headers, body

    def send_request(self, prompt: str, params: GenerationParams) -> str:
        url, headers, body = self.build_request(prompt, params)
        request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            details = self.read_error_body(error)
            raise LLMClientError(f"HTTP {error.code} from {url}: {details}")
        except urllib.error.URLError as error:
            raise LLMClientError(f"Network error for {url}: {error.reason}")
        except (TimeoutError, socket.timeout):
            raise LLMClientError(f"Request to {url} timed out after {self.timeout} seconds")

        return self.parse_body(raw_body)

    @staticmethod
    def read_error_body(error: urllib.error.HTTPError, limit: int = 500) -> str:
        try:
            return error.read().decode("utf-8", errors="replace")[:limit]
        except Exception:
            return error.reason if isinstance(error.reason, str) else ""

    def parse_body(self, raw_body: str) -> str:
        """Extract the assistant message text from a chat completions payload"""
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as error:
            raise LLMClientError(f"Invalid JSON in API response: {error}")

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMClientError(f"Unexpected API response structure: {raw_body[:500]}")

        if not isinstance(content, str):
            raise LLMClientError(f"API response content is not a string: {content!r}")

        return content


# Provider registry. Every entry maps a provider name to a client class and its defaults;
# a new adapter (for example an in-process transformers backend) is registered by adding one entry
PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(OpenAICompatibleClient, "https://api.openai.com/v1", requires_token=True),
    "openai_compatible": ProviderSpec(OpenAICompatibleClient, None, requires_token=False),
    "ollama": ProviderSpec(OpenAICompatibleClient, "http://localhost:11434/v1", requires_token=False),
    "vllm": ProviderSpec(OpenAICompatibleClient, "http://localhost:8000/v1", requires_token=False),
}

DEFAULT_PROVIDER = "openai"


def get_provider_spec(provider: str) -> ProviderSpec:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Supported providers: {sorted(PROVIDERS)}")

    return PROVIDERS[provider]


def create_llm_client(
    settings: LLMSettings,
    timeout: float = 60.0,
    max_retries: int = 2,
    retry_delay: float = 1.0
) -> LLMClient:
    """Create the client registered for the provider in the settings"""
    spec = get_provider_spec(settings.provider)

    return spec.client_class(
        settings=settings,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

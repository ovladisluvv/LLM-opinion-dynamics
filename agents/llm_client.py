import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class LLMClientError(Exception):
    """Raised when an LLM API call fails after all retries or returns an invalid payload"""
    pass


class LLMSettings:
    """Connection settings for one LLM role, loaded from environment variables"""
    ENV_SUFFIXES = ("PROVIDER", "MODEL", "TOKEN", "BASE_URL")

    def __init__(
        self,
        model_name: str,
        token: str,
        provider: str | None = None,
        base_url: str | None = None
    ):
        if not model_name:
            raise ValueError("model_name has to be a non-empty string")

        if not token:
            raise ValueError("token has to be a non-empty string")

        self.model_name = model_name
        self.token = token
        self.provider = provider
        self.base_url = base_url

    @classmethod
    def from_env(cls, role: str) -> "LLMSettings":
        """Load settings for a role from {role}_LLM_* environment variables, e.g. PARTICIPANT_LLM_MODEL"""
        values = {suffix: os.environ.get(f"{role}_LLM_{suffix}", "").strip() for suffix in cls.ENV_SUFFIXES}

        missing = [f"{role}_LLM_{suffix}" for suffix in ("MODEL", "TOKEN") if not values[suffix]]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}. Set them in .env (see .env.example)")

        return cls(
            model_name=values["MODEL"],
            token=values["TOKEN"],
            provider=values["PROVIDER"] or None,
            base_url=values["BASE_URL"] or None
        )


class LLMClient(ABC):
    """Provider-independent LLM client with a retry loop around a single abstract request method"""
    def __init__(
        self,
        settings: LLMSettings,
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        max_tokens: int | None = None
    ):
        if timeout <= 0:
            raise ValueError("timeout has to be positive")

        if max_retries < 0:
            raise ValueError("max_retries has to be non-negative")

        if retry_delay < 0:
            raise ValueError("retry_delay has to be non-negative")

        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens has to be positive when set")

        self.settings = settings
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    def generate(self, prompt: str, temperature: float) -> str:
        """Send the prompt to the model, retrying failed requests, and return the response text"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return self.send_request(prompt, temperature)
            except LLMClientError as error:
                last_error = error

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        raise LLMClientError(f"LLM request for model {self.settings.model_name} failed after {self.max_retries + 1} attempts. Last error: {last_error}")

    @abstractmethod
    def send_request(self, prompt: str, temperature: float) -> str:
        """Perform one API request and return the response text. Raises LLMClientError on failure"""
        pass

def create_llm_client(
    settings: LLMSettings,
    timeout: float = 60.0,
    max_retries: int = 2,
    retry_delay: float = 1.0,
    max_tokens: int | None = None
) -> LLMClient:
    """Create an LLM client for the provider in the settings

    An unset provider defaults to the OpenAI-compatible client, which supports
    any endpoint speaking the chat completions protocol via base_url.
    New provider adapters should be registered here
    """
    providers = {
        ...
    }

    if settings.provider not in providers:
        supported = sorted(name for name in providers if name is not None)
        raise ValueError(
            f"Unknown LLM provider: {settings.provider}"
            f"Supported providers: {supported}"
        )

    client_class = providers[settings.provider]

    return client_class(
        settings=settings,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        max_tokens=max_tokens
    )

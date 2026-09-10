import os
from pathlib import Path


def parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse one KEY=VALUE line, returning None for blank lines and comments"""
    stripped = line.strip()

    if not stripped or stripped.startswith("#"):
        return None

    if "=" not in stripped:
        raise ValueError(f"Invalid .env line (expected KEY=VALUE): {stripped}")

    key, _, value = stripped.partition("=")
    key = key.strip()
    value = value.strip()

    if not key:
        raise ValueError(f"Invalid .env line (empty key): {stripped}")

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]

    return key, value


def load_env_file(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """
    Load environment variables from a .env file into os.environ

    Variables already present in os.environ are kept unless override is True.
    A missing file is not an error: variables may be provided by the environment itself.
    Returns the variables read from the file
    """
    path = Path(path)
    loaded = {}

    if not path.exists():
        return loaded

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parsed = parse_env_line(line)

            if parsed is None:
                continue

            key, value = parsed
            loaded[key] = value

            if override or key not in os.environ:
                os.environ[key] = value

    return loaded

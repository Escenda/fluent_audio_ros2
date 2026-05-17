from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAiCredentialConfig:
    api_key_env: str


def load_openai_credential_config(
    *, parameter_name: str, api_key_env: str
) -> OpenAiCredentialConfig:
    env_name = api_key_env.strip()
    if not env_name:
        raise RuntimeError(f"{parameter_name} is required")
    if not os.environ.get(env_name, "").strip():
        raise RuntimeError(
            f"environment variable {env_name} referenced by {parameter_name} is required"
        )
    return OpenAiCredentialConfig(api_key_env=env_name)

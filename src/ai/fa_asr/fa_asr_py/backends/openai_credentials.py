from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAiCredentialConfig:
    api_key_env: str
    api_key_value: str


def load_openai_credential_config(
    *, parameter_name: str, api_key_env: str
) -> OpenAiCredentialConfig:
    env_name = api_key_env.strip()
    if not env_name:
        raise RuntimeError(f"{parameter_name} is required")
    api_key_value = os.environ.get(env_name, "").strip()
    if not api_key_value:
        raise RuntimeError(
            f"environment variable {env_name} referenced by {parameter_name} is required"
        )
    return OpenAiCredentialConfig(api_key_env=env_name, api_key_value=api_key_value)

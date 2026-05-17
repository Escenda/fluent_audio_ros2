from pathlib import Path

import pytest
import yaml

from fa_vad_py.backends.silero import SileroVAD


def test_default_config_requires_explicit_silero_repo_dir() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default_vad.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_vad_node"]["ros__parameters"]

    assert params["backend.name"] == "silero"
    assert params["silero"]["allow_online"] is False
    assert params["silero"]["repo_dir"] == ""


def test_silero_backend_rejects_missing_repo_dir_when_offline() -> None:
    with pytest.raises(RuntimeError, match="silero.repo_dir is required"):
        SileroVAD(silero_repo_dir="", allow_online=False)


def test_silero_backend_rejects_missing_local_repo_even_when_online_allowed() -> None:
    missing_repo = "/tmp/fluent_audio_missing_silero_repo"

    with pytest.raises(RuntimeError, match="silero.repo_dir does not exist"):
        SileroVAD(silero_repo_dir=missing_repo, allow_online=True)

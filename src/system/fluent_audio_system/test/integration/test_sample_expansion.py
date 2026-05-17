from pathlib import Path

import pytest
import yaml

from fluent_audio_system import config_schema
from fluent_audio_system.config_schema import load_system_config


PACKAGE_ROOT = Path(__file__).parents[2]
FIXTURE_DIR = PACKAGE_ROOT / "test" / "fixtures"


def _patch_fluent_audio_system_share(monkeypatch: pytest.MonkeyPatch) -> None:
    def package_share(package_name: str) -> str:
        if package_name != "fluent_audio_system":
            raise RuntimeError(f"unexpected package share lookup: {package_name}")
        return str(PACKAGE_ROOT)

    monkeypatch.setattr(config_schema, "_get_package_share_directory", package_share)


def test_valid_fixture_expands_enabled_nodes_and_remappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    spec = load_system_config(
        "${share:fluent_audio_system}/test/fixtures/valid_io_system.yaml"
    )

    assert len(spec.groups) == 1
    assert spec.groups[0].id == "io"
    assert len(spec.groups[0].nodes) == 1
    node = spec.groups[0].nodes[0]
    assert node.id == "fa_in"
    assert node.package == "fa_in"
    assert node.executable == "fa_in_node"
    assert node.params_file == str(FIXTURE_DIR / "fa_in.params.yaml")
    assert node.launch_remappings() == [("audio/frame", "robot/audio/input")]

    params = _load_fixture_params("fa_in.params.yaml", "fa_in_node")
    assert params["audio.stream_id"] == "audio/frame"
    assert params["audio.layout"] == "interleaved"


def test_missing_fixture_params_file_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    with pytest.raises(RuntimeError, match="params_file not found"):
        load_system_config(
            "${share:fluent_audio_system}/test/fixtures/missing_params_system.yaml"
        )


def test_invalid_schema_fixture_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    with pytest.raises(RuntimeError, match="system.inter_group_delay is required"):
        load_system_config(
            "${share:fluent_audio_system}/test/fixtures/invalid_schema_system.yaml"
        )


def test_sequence_remapping_fixture_expands_to_launch_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    spec = load_system_config(
        "${share:fluent_audio_system}/test/fixtures/remapping_system.yaml"
    )

    assert len(spec.groups) == 1
    assert len(spec.groups[0].nodes) == 1
    assert spec.groups[0].nodes[0].launch_remappings() == [
        ("audio/frame", "robot/audio/input"),
        ("vad/state", "robot/audio/vad/state"),
    ]


def _load_fixture_params(fixture_name: str, node_name: str) -> dict[str, str | int | bool]:
    raw = yaml.safe_load((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))
    return raw[node_name]["ros__parameters"]

from pathlib import Path

import pytest
import yaml

from fluent_audio_system.config_schema import AudioNodeSpec
from fluent_audio_system.site_binding import (
    build_site_binding_overrides,
    parse_bool_launch_arg_value,
)
from fluent_audio_system.site_binding_launch import (
    SOURCE_BOUND_AUDIO_AI_PACKAGES,
    node_enabled_by_site_binding,
    node_launch_parameters,
)


PACKAGE_ROOT = Path(__file__).parents[2]


def _node(
    *,
    package: str,
    backend_name: str | None,
    params_file: str = "/tmp/node.params.yaml",
) -> AudioNodeSpec:
    return AudioNodeSpec(
        id=package,
        package=package,
        executable=f"{package}_node",
        node_name=package,
        namespace="",
        output="screen",
        params_file=params_file,
        parameters={},
        remappings=[],
        backend_name=backend_name,
    )
def test_site_binding_launch_gates_only_alsa_site_bound_io() -> None:
    overrides = build_site_binding_overrides(
        fa_in_enabled=False,
        fa_out_enabled=False,
        fa_in_source_id="",
        fa_out_sink_id="",
    )

    assert not node_enabled_by_site_binding(
        _node(package="fa_in", backend_name="alsa_capture"),
        overrides,
    )
    assert not node_enabled_by_site_binding(
        _node(package="fa_out", backend_name="alsa_playback"),
        overrides,
    )
    assert node_enabled_by_site_binding(
        _node(package="fa_in", backend_name="pcm_file_reader"),
        overrides,
    )
    assert node_enabled_by_site_binding(
        _node(package="fa_out", backend_name="pcm_file_writer"),
        overrides,
    )


def test_site_binding_launch_applies_alsa_source_and_sink_ids() -> None:
    overrides = build_site_binding_overrides(
        fa_in_enabled=True,
        fa_out_enabled=True,
        fa_in_source_id="hw:CARD=Mic,DEV=0",
        fa_out_sink_id="hw:CARD=Speaker,DEV=0",
    )

    fa_in_params = node_launch_parameters(
        _node(package="fa_in", backend_name="alsa_capture"),
        overrides,
    )
    fa_out_params = node_launch_parameters(
        _node(package="fa_out", backend_name="alsa_playback"),
        overrides,
    )

    assert fa_in_params == [
        "/tmp/node.params.yaml",
        {
            "audio.device_selector.mode": "id",
            "audio.device_selector.identifier": "hw:CARD=Mic,DEV=0",
        },
    ]
    assert fa_out_params == [
        "/tmp/node.params.yaml",
        {"audio.device_id": "hw:CARD=Speaker,DEV=0"},
    ]


@pytest.mark.parametrize("package_name", sorted(SOURCE_BOUND_AUDIO_AI_PACKAGES))
def test_site_binding_launch_applies_source_id_to_audio_ai_nodes(
    package_name: str,
) -> None:
    overrides = build_site_binding_overrides(
        fa_in_enabled=True,
        fa_out_enabled=False,
        fa_in_source_id="hw:CARD=Mic,DEV=0",
        fa_out_sink_id="",
    )

    assert node_launch_parameters(
        _node(package=package_name, backend_name="test_backend"),
        overrides,
    ) == [
        "/tmp/node.params.yaml",
        {"expected_source_id": "hw:CARD=Mic,DEV=0"},
    ]


def test_sample_config_keeps_device_site_binding_out_of_system_config() -> None:
    config_path = PACKAGE_ROOT / "config" / "fluent_audio_system.sample.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    io_group = next(group for group in config["groups"] if group["id"] == "io")
    fa_in = next(node for node in io_group["nodes"] if node["id"] == "fa_in")
    fa_out = next(node for node in io_group["nodes"] if node["id"] == "fa_out")

    assert io_group["enable"] is True
    assert fa_in["enable"] is True
    assert fa_out["enable"] is True
    assert fa_in["parameters"] == {
        "output_topic": "audio/frame",
        "audio.stream_id": "audio/raw/mic",
    }
    assert fa_out["parameters"] == {
        "input_topic": "audio/output/frame",
        "input_stream_id": "audio/playback/main",
        "playback_done_topic": "audio/output/playback_done",
        "playback_control_service": "audio/output/playback_control",
    }
    assert "audio.device_selector.identifier" not in fa_in["parameters"]
    assert "audio.device_id" not in fa_out["parameters"]
def test_site_binding_bool_values_are_strict() -> None:
    assert parse_bool_launch_arg_value("fa_in_enabled", "true") is True
    assert parse_bool_launch_arg_value("fa_in_enabled", " false ") is False

    with pytest.raises(RuntimeError, match="fa_in_enabled must be true or false"):
        parse_bool_launch_arg_value("fa_in_enabled", "")


def test_enabled_source_requires_source_id() -> None:
    with pytest.raises(RuntimeError, match="fa_in_source_id is required"):
        build_site_binding_overrides(
            fa_in_enabled=True,
            fa_out_enabled=False,
            fa_in_source_id="",
            fa_out_sink_id="",
        )


def test_enabled_sink_requires_sink_id() -> None:
    with pytest.raises(RuntimeError, match="fa_out_sink_id is required"):
        build_site_binding_overrides(
            fa_in_enabled=False,
            fa_out_enabled=True,
            fa_in_source_id="",
            fa_out_sink_id="",
        )


def test_disabled_source_sink_allow_empty_binding() -> None:
    overrides = build_site_binding_overrides(
        fa_in_enabled=False,
        fa_out_enabled=False,
        fa_in_source_id="",
        fa_out_sink_id="",
    )

    assert overrides.fa_in_enabled is False
    assert overrides.fa_out_enabled is False
    assert overrides.fa_in_source_id == ""
    assert overrides.fa_out_sink_id == ""

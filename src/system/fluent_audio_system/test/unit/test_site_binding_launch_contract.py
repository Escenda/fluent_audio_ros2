from pathlib import Path

import pytest
import yaml

from fluent_audio_system.site_binding import (
    build_site_binding_overrides,
    parse_bool_launch_arg_value,
)


PACKAGE_ROOT = Path(__file__).parents[2]


def test_system_launch_declares_site_binding_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fluent_audio_system.launch.py").read_text(
        encoding="utf-8"
    )
    run_text = (PACKAGE_ROOT / "launch" / "run.py").read_text(encoding="utf-8")

    for arg_name in ("fa_in_enabled", "fa_out_enabled", "fa_in_source_id", "fa_out_sink_id"):
        assert f'"{arg_name}"' in launch_text
        assert f'"{arg_name}"' in run_text

    assert 'default_value="false"' in launch_text
    assert "_required_bool_launch_arg" in launch_text
    assert "_node_enabled_by_site_binding" in launch_text
    assert "_node_launch_parameters" in launch_text
    assert 'node.package == "fa_in"' in launch_text
    assert 'node.package == "fa_out"' in launch_text
    assert "_SOURCE_BOUND_AUDIO_AI_PACKAGES" in launch_text
    assert '"fa_asr"' in launch_text
    assert '"fa_kws"' in launch_text
    assert '"fa_turn_detector"' in launch_text
    assert '"fa_vad"' in launch_text
    assert 'override_params["audio.device_selector.mode"] = "id"' in launch_text
    assert 'override_params["expected_source_id"] = overrides.fa_in_source_id' in launch_text
    assert 'node.id == "fa_in"' not in launch_text
    assert 'node.id == "fa_out"' not in launch_text


def test_sample_config_keeps_site_binding_out_of_system_config() -> None:
    config_path = PACKAGE_ROOT / "config" / "fluent_audio_system.sample.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    io_group = next(group for group in config["groups"] if group["id"] == "io")
    fa_in = next(node for node in io_group["nodes"] if node["id"] == "fa_in")
    fa_out = next(node for node in io_group["nodes"] if node["id"] == "fa_out")

    assert io_group["enable"] is True
    assert fa_in["enable"] is True
    assert fa_out["enable"] is True
    assert fa_in["parameters"] == {}
    assert fa_out["parameters"] == {}


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

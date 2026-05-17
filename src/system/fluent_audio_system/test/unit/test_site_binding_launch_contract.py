from pathlib import Path

import yaml


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

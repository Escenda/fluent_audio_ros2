from pathlib import Path

import yaml


def test_default_config_requires_explicit_sink_device() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    params = config["fa_out"]["ros__parameters"]

    assert params["backend.name"] == "alsa_playback"
    assert params["audio.device_id"] == ""
    assert '"default"' not in config_text


def test_sink_backend_has_no_struct_default() -> None:
    header_path = Path(__file__).parents[2] / "include" / "fa_out" / "fa_out_node.hpp"

    assert "std::string backend_name{};" in header_path.read_text(encoding="utf-8")


def test_alsa_sink_rejects_plugin_pcm_devices() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "isRawAlsaPlaybackDevice" in source
    assert 'rfind("hw:", 0)' in source
    assert "audio.device_id must be an ALSA raw hardware id" in source

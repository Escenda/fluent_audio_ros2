from pathlib import Path

import yaml


def test_default_config_requires_explicit_sink_device() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    params = config["fa_out"]["ros__parameters"]

    assert params["backend.name"] == "alsa_playback"
    assert params["audio.device_id"] == ""
    assert params["audio.chunk_duration_ms"] == 30
    assert params["audio.qos.depth"] == 10
    assert params["audio.qos.reliable"] is True
    assert '"default"' not in config_text


def test_sink_backend_has_no_struct_default() -> None:
    header_path = Path(__file__).parents[2] / "include" / "fa_out" / "fa_out_node.hpp"
    header_text = header_path.read_text(encoding="utf-8")

    assert "std::string backend_name{};" in header_text
    assert "uint32_t sample_rate{0};" in header_text
    assert "uint32_t channels{0};" in header_text
    assert "uint32_t bit_depth{0};" in header_text
    assert "size_t max_queue_frames{0};" in header_text
    assert "uint32_t chunk_duration_ms{0};" in header_text
    assert "size_t qos_depth{0};" in header_text


def test_alsa_sink_rejects_plugin_pcm_devices() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "isRawAlsaPlaybackDevice" in source
    assert 'rfind("hw:", 0)' in source
    assert "audio.device_id must be an ALSA raw hardware id" in source


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert 'declare_parameter<int>("audio.sample_rate")' in source
    assert 'declare_parameter<int>("audio.channels")' in source
    assert 'declare_parameter<int>("audio.bit_depth")' in source
    assert 'declare_parameter<int>("queue.max_frames")' in source
    assert 'declare_parameter<int>("audio.chunk_duration_ms")' in source
    assert 'declare_parameter<int>("audio.qos.depth")' in source
    assert 'declare_parameter<bool>("audio.qos.reliable")' in source


def test_runtime_write_failure_fails_closed_without_reopen_retry() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    source = source_path.read_text(encoding="utf-8")
    playback_thread = source.split("void FaOutNode::playbackThread()")[1].split(
        "}  // namespace fa_out"
    )[0]

    assert "failClosed(" in playback_thread
    assert "openDevice()" not in playback_thread
    assert "ALSA device unavailable, dropping frame" not in playback_thread
    assert "snd_pcm_prepare(pcm_handle_);" not in playback_thread
    assert "rclcpp::shutdown()" in source

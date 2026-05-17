from pathlib import Path

import yaml


def test_default_config_requires_explicit_source_identifier() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    params = config["fa_in_node"]["ros__parameters"]
    selector = params["audio"]["device_selector"]

    assert params["backend"]["name"] == "alsa_capture"
    assert params["audio"]["encoding"] == "PCM16LE"
    assert params["audio"]["bit_depth"] == 16
    assert params["audio"]["stream_id"] == "audio/frame"
    assert params["audio"]["layout"] == "interleaved"
    assert selector["mode"] == "name"
    assert selector["identifier"] == ""
    assert '"default"' not in config_text


def test_source_backend_has_no_struct_default() -> None:
    package_root = Path(__file__).parents[2]
    header_path = package_root / "include" / "fa_in" / "fa_in_node.hpp"
    source_path = package_root / "src" / "fa_in_node.cpp"

    header_text = header_path.read_text(encoding="utf-8")
    source_text = source_path.read_text(encoding="utf-8")

    assert "std::string backend_name{};" in header_text
    assert "std::string device_mode{};" in header_text
    assert "uint32_t sample_rate{0};" in header_text
    assert "uint32_t channels{0};" in header_text
    assert "uint32_t bit_depth{0};" in header_text
    assert "uint32_t chunk_ms{0};" in header_text
    assert "std::string encoding{};" in header_text
    assert "std::string stream_id{};" in header_text
    assert "std::string layout{};" in header_text
    assert "uint32_t diag_period_ms{0};" in header_text
    assert 'declare_parameter<int>("audio.sample_rate")' in source_text
    assert 'declare_parameter<int>("audio.channels")' in source_text
    assert 'declare_parameter<int>("audio.bit_depth")' in source_text
    assert 'declare_parameter<int>("audio.chunk_ms")' in source_text
    assert 'declare_parameter<std::string>("audio.encoding")' in source_text
    assert 'declare_parameter<std::string>("audio.stream_id")' in source_text
    assert 'declare_parameter<std::string>("audio.layout")' in source_text
    assert 'declare_parameter<int>("diagnostics.publish_period_ms")' in source_text


def test_publish_frame_sets_required_audio_frame_identity_without_analysis_fields() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")
    publish_frame = source.split("void FaInNode::publishFrame")[1].split(
        "void FaInNode::publishDiagnostics"
    )[0]

    assert "frame_msg.source_id = active_device_id_;" in publish_frame
    assert "frame_msg.stream_id = config_.stream_id;" in publish_frame
    assert "frame_msg.layout = config_.layout;" in publish_frame
    assert ".rms" not in publish_frame
    assert ".peak" not in publish_frame
    assert ".vad" not in publish_frame


def test_alsa_backend_filters_plugin_pcm_sources() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "isRawAlsaHardwareSource" in source
    assert 'rfind("hw:", 0)' in source
    assert "devices.emplace_back(source_id" in source


def test_alsa_backend_validates_format_contract_and_disables_resampling() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "alsaFormatForConfig" in source
    assert '"audio.encoding/audio.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32"' in source
    assert "snd_pcm_hw_params_set_rate_resample(pcm_handle_, params, 0)" in source


def test_runtime_read_failure_fails_closed_without_prepare_retry() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")
    capture_loop = source.split("void FaInNode::captureLoop()")[1].split(
        "void FaInNode::publishFrame"
    )[0]

    assert "failClosed(" in capture_loop
    assert "snd_pcm_prepare" not in capture_loop
    assert "std::this_thread::sleep_for" not in capture_loop
    assert "rclcpp::shutdown()" in source


def test_colcon_runs_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

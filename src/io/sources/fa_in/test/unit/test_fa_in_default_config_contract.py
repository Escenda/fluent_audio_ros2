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
    validation_header_path = package_root / "include" / "fa_in" / "audio_config_validation.hpp"
    source_path = package_root / "src" / "fa_in_node.cpp"

    header_text = header_path.read_text(encoding="utf-8")
    validation_text = validation_header_path.read_text(encoding="utf-8")
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
    assert "size_t bytes_per_buffer_{0};" in header_text
    assert 'declare_parameter<int>("audio.sample_rate")' in source_text
    assert 'declare_parameter<int>("audio.channels")' in source_text
    assert 'declare_parameter<int>("audio.bit_depth")' in source_text
    assert 'declare_parameter<int>("audio.chunk_ms")' in source_text
    assert 'declare_parameter<std::string>("audio.encoding")' in source_text
    assert 'declare_parameter<std::string>("audio.stream_id")' in source_text
    assert 'declare_parameter<std::string>("audio.layout")' in source_text
    assert 'declare_parameter<int>("diagnostics.publish_period_ms")' in source_text
    assert "requirePositiveUint32" in source_text
    assert "std::max<uint32_t>" not in source_text
    assert "validation::bytesPerFrame" in source_text
    assert "validation::bytesForFrames" in source_text
    assert "audio.sample_rate * audio.chunk_ms must produce an integer frame count" in validation_text
    assert "audio.chunk_ms produces zero capture frames" in validation_text
    assert "audio.channels * audio.bit_depth exceeds size_t range" in validation_text


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
    source_path = Path(__file__).parents[2] / "src" / "backends" / "alsa_capture_backend.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "isRawAlsaHardwareSource" in source
    assert 'rfind("hw:", 0)' in source
    assert "devices.push_back(DeviceInfo{source_id" in source
    assert "throw BackendError(alsaError(\"snd_device_name_hint failed\", err));" in source


def test_alsa_backend_validates_format_contract_and_disables_resampling() -> None:
    source_path = Path(__file__).parents[2] / "src" / "backends" / "alsa_capture_backend.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "alsaFormatForConfig" in source
    assert '"audio.encoding/audio.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32"' in source
    assert "snd_pcm_hw_params_set_rate_resample(pcm_handle_, params, 0)" in source
    assert "ALSA period size negotiation changed requested capture chunk" in source
    assert "requested capture chunk exceeds ALSA frame count range" in source


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


def test_device_services_surface_enumeration_failure() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")
    list_devices = source.split("void FaInNode::handleListDevices")[1].split(
        "void FaInNode::handleSwitchDevice"
    )[0]
    switch_device = source.split("void FaInNode::handleSwitchDevice")[1].split(
        "bool FaInNode::reopenStream"
    )[0]

    assert "throw backends::BackendError(\"source backend is not initialized\")" in source
    assert "response->success = false;" in list_devices
    assert "response->message = e.what();" in list_devices
    assert "response->success = true;" in list_devices
    assert "response->message = \"ok\";" in list_devices
    assert "response->success = false;" in switch_device
    assert "response->message = e.what();" in switch_device


def test_backend_implementation_files_are_ros_free() -> None:
    package_root = Path(__file__).parents[2]
    backend_paths = sorted((package_root / "src" / "backends").glob("*.cpp"))
    forbidden_tokens = (
        "rclcpp",
        "fa_interfaces",
        "diagnostic_msgs",
        "std_msgs/msg",
    )

    assert backend_paths
    for path in backend_paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source


def test_backend_builds_as_separate_library() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_library(fa_in_backends" in cmake_text
    assert "src/backends/alsa_capture_backend.cpp" in cmake_text
    assert "target_link_libraries(fa_in_node fa_in_backends)" in cmake_text


def test_node_header_does_not_store_alsa_pcm_handle() -> None:
    header_path = Path(__file__).parents[2] / "include" / "fa_in" / "fa_in_node.hpp"
    header = header_path.read_text(encoding="utf-8")

    assert "snd_pcm_t" not in header
    assert "pcm_handle_" not in header
    assert "alsa/asoundlib.h" not in header
    assert "std::unique_ptr<fa_in::backends::SourceBackend> source_backend_;" in header


def test_colcon_runs_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_audio_config_validation_test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

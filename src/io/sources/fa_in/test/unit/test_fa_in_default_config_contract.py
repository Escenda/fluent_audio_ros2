from pathlib import Path

import yaml


def test_default_config_requires_explicit_source_identifier() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    assert "fa_in" in config
    assert "fa_in_node" not in config
    params = config["fa_in"]["ros__parameters"]
    selector = params["audio"]["device_selector"]

    assert params["backend.name"] == "alsa_capture"
    assert params["output_topic"] == "fa_in/output"
    assert "backend" not in params
    assert params["audio"]["encoding"] == "PCM16LE"
    assert params["audio"]["bit_depth"] == 16
    assert params["audio"]["stream_id"] == "audio/raw/mic"
    assert params["output_topic"] != params["audio"]["stream_id"]
    assert params["audio"]["layout"] == "interleaved"
    assert params["audio"]["qos"]["depth"] == 10
    assert params["audio"]["qos"]["reliable"] is False
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert selector["mode"] == "id"
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
    assert "std::string output_topic{};" in header_text
    assert "std::string device_mode{};" in header_text
    assert "std::string file_path{};" in header_text
    assert "std::string endpoint_uri{};" in header_text
    assert "std::string transport_identity{};" in header_text
    assert "std::string source_id{};" in header_text
    assert "bool playback_loop{false};" in header_text
    assert "uint32_t sample_rate{0};" in header_text
    assert "uint32_t channels{0};" in header_text
    assert "uint32_t bit_depth{0};" in header_text
    assert "uint32_t chunk_ms{0};" in header_text
    assert "std::string encoding{};" in header_text
    assert "std::string stream_id{};" in header_text
    assert "std::string layout{};" in header_text
    assert "uint32_t audio_qos_depth{0};" in header_text
    assert "bool audio_qos_reliable{false};" in header_text
    assert "uint32_t diag_period_ms{0};" in header_text
    assert "uint32_t network_max_packet_bytes{0};" in header_text
    assert "uint32_t polling_period_ms{0};" in header_text
    assert "size_t bytes_per_buffer_{0};" in header_text
    assert 'declare_parameter<std::string>("output_topic")' in source_text
    assert 'declare_parameter<int>("audio.sample_rate")' in source_text
    assert 'declare_parameter<int>("audio.channels")' in source_text
    assert 'declare_parameter<int>("audio.bit_depth")' in source_text
    assert 'declare_parameter<int>("audio.chunk_ms")' in source_text
    assert 'declare_parameter<std::string>("audio.encoding")' in source_text
    assert 'declare_parameter<std::string>("audio.stream_id")' in source_text
    assert 'declare_parameter<std::string>("audio.layout")' in source_text
    assert 'declare_parameter("endpoint.uri", rclcpp::ParameterValue{}, dynamic_parameter)' in source_text
    assert 'declare_parameter("transport.identity", rclcpp::ParameterValue{}, dynamic_parameter)' in source_text
    assert 'declare_parameter("network.max_packet_bytes", rclcpp::ParameterValue{}, dynamic_parameter)' in source_text
    assert 'declare_parameter("polling.period_ms", rclcpp::ParameterValue{}, dynamic_parameter)' in source_text
    assert 'declare_parameter<int>("audio.qos.depth")' in source_text
    assert 'declare_parameter<bool>("audio.qos.reliable")' in source_text
    assert 'declare_parameter<int>("diagnostics.qos.depth")' in source_text
    assert 'declare_parameter<bool>("diagnostics.qos.reliable")' in source_text
    assert 'declare_parameter<int>("diagnostics.publish_period_ms")' in source_text
    assert "readRequiredString(*this, \"output_topic\")" in source_text
    assert "readRequiredBool(*this, \"audio.qos.reliable\")" in source_text
    assert "readRequiredBool(*this, \"diagnostics.qos.reliable\")" in source_text
    forbidden_identifier_default = "declare_parameter(\"audio.device_selector.identifier\", "
    forbidden_identifier_default += "config_"
    assert forbidden_identifier_default not in source_text
    assert "requirePositiveUint32" in source_text
    assert "std::max<uint32_t>" not in source_text
    assert "validation::bytesPerFrame" in source_text
    assert "validation::bytesForFrames" in source_text
    assert "audio.sample_rate * audio.chunk_ms must produce an integer frame count" in validation_text
    assert "audio.chunk_ms produces zero capture frames" in validation_text
    assert "audio.channels * audio.bit_depth exceeds size_t range" in validation_text


def test_launch_requires_explicit_node_name_and_config_file() -> None:
    package_root = Path(__file__).parents[2]
    launch_text = (package_root / "launch" / "fa_in.launch.py").read_text(encoding="utf-8")
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    node_source = (package_root / "src" / "fa_in_node.cpp").read_text(encoding="utf-8")
    main_source = (package_root / "src" / "main.cpp").read_text(encoding="utf-8")

    assert ("default_" + "value") not in launch_text
    assert ("FindPackage" + "Share") not in launch_text
    assert ("PathJoin" + "Substitution") not in launch_text
    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'rclcpp::Node("fa_in", options)' in node_source
    assert 'status.name = "fa_in";' in node_source
    assert 'get_logger("fa_in")' in main_source
    assert set(config.keys()) == {"fa_in"}


def test_publish_frame_sets_required_audio_frame_identity_without_analysis_fields() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")
    publish_frame = source.split("void FaInNode::publishFrame")[1].split(
        "void FaInNode::publishDiagnostics"
    )[0]

    assert "frame_msg.source_id = active_source_id_;" in publish_frame
    assert "frame_msg.stream_id = config_.stream_id;" in publish_frame
    assert "frame_msg.layout = config_.layout;" in publish_frame
    assert ".rms" not in publish_frame
    assert ".peak" not in publish_frame
    assert ".vad" not in publish_frame


def test_publishers_use_explicit_output_topic_and_transport_qos() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "create_publisher<fa_interfaces::msg::AudioFrame>(" in source
    assert "config_.output_topic" in source
    assert "makeExplicitQos(config_.audio_qos_depth, config_.audio_qos_reliable)" in source
    assert "makeExplicitQos(config_.diagnostics_qos_depth, config_.diagnostics_qos_reliable)" in source
    assert "rclcpp::SensorDataQoS()" not in source
    assert "System" + "Defaults" + "QoS" not in source
    forbidden_hardcoded_audio_topic = (
        "audio_pub_ = this->create_publisher<fa_interfaces::msg::AudioFrame>(\""
    )
    forbidden_hardcoded_audio_topic += "audio/"
    forbidden_hardcoded_audio_topic += "frame\""
    assert forbidden_hardcoded_audio_topic not in source


def test_alsa_backend_filters_plugin_pcm_sources() -> None:
    source_path = Path(__file__).parents[2] / "src" / "backends" / "alsa_capture_backend.cpp"
    validation_header_path = Path(__file__).parents[2] / "include" / "fa_in" / "audio_config_validation.hpp"
    source = source_path.read_text(encoding="utf-8")
    validation_header = validation_header_path.read_text(encoding="utf-8")

    assert "isRawAlsaHardwareSource" in validation_header
    assert 'rfind("hw:", 0)' in validation_header
    assert "validation::isRawAlsaHardwareSource(source_id)" in source
    assert "validation::requireRawAlsaHardwareSource(device_id);" in source
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
    assert "capture loop started without required source backend" in capture_loop
    assert "ReadStatus::kNoData" in capture_loop
    assert "std::this_thread::sleep_for" in capture_loop
    assert "read_result.frames != frames_per_buffer_" in capture_loop
    assert "expected configured capture chunk" in capture_loop
    assert "snd_pcm_prepare" not in capture_loop
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
    assert "response->max_input_channels.push_back(dev.max_input_channels);" in list_devices
    assert "response->default_sample_rates.push_back(dev.default_sample_rate);" in list_devices
    assert "response->max_input_channels.push_back(config_.channels);" not in list_devices
    assert "response->default_sample_rates.push_back(config_.sample_rate);" not in list_devices
    assert "response->success = false;" in switch_device
    assert "response->message = e.what();" in switch_device
    assert "validation::requireSwitchDeviceSelector" in switch_device
    assert "request->target_selector_mode" in switch_device
    assert "device index not found" in switch_device
    assert "device id not found" in switch_device
    assert "device name is ambiguous" in switch_device
    assert "restart=false" not in switch_device
    assert "request->restart" not in switch_device
    assert "else if (!request->target_identifier.empty())" not in switch_device
    assert 'request->target_selector_mode == "id"' in switch_device
    assert 'request->target_selector_mode == "name"' in switch_device
    assert 'request->target_selector_mode == "index"' in switch_device


def test_alsa_name_selector_fails_closed_on_duplicate_display_names() -> None:
    package_root = Path(__file__).parents[2]
    backend_source = (
        package_root / "src" / "backends" / "alsa_capture_backend.cpp"
    ).read_text(encoding="utf-8")
    node_source = (package_root / "src" / "fa_in_node.cpp").read_text(
        encoding="utf-8"
    )
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")
    backend_doc = (package_root / "docs" / "backends" / "alsa.md").read_text(
        encoding="utf-8"
    )
    validation_header = (
        package_root / "include" / "fa_in" / "audio_config_validation.hpp"
    ).read_text(encoding="utf-8")
    node_display_name = node_source.split("std::string displayName")[1].split(
        "FaInNode::BackendFactory"
    )[0]
    backend_display_name = backend_source.split("std::string displayName")[1].split(
        "snd_pcm_format_t"
    )[0]
    id_selector_block = backend_source.split('if (selector.mode == "id")')[1].split(
        'if (selector.mode == "name")'
    )[0]
    name_selector_block = backend_source.split('if (selector.mode == "name")')[1].split(
        'throw BackendError("unsupported audio.device_selector.mode'
    )[0]

    assert 'selector.mode == "id"' in backend_source
    assert "Configured ALSA input source id was not found" in backend_source
    assert "device.id == selector.identifier" in id_selector_block
    assert "device.id == selector.identifier" not in name_selector_block
    assert "displayName(device) == selector.identifier" in name_selector_block
    assert "display_name_matches.size() == 1" in backend_source
    assert "display_name_matches.size() > 1" in backend_source
    assert "Configured ALSA input source name is ambiguous" in backend_source
    assert "must be a display name, not a raw hw: source id" in validation_header
    forbidden_display_name_fallback = "return device"
    forbidden_display_name_fallback += ".id;"
    assert forbidden_display_name_fallback not in node_display_name
    assert forbidden_display_name_fallback not in backend_display_name
    assert "display_name_matches.size() == 1" in node_source
    assert "display_name_matches.size() > 1" in node_source
    assert "device name is ambiguous" in node_source
    switch_device = node_source.split("void FaInNode::handleSwitchDevice")[1].split(
        "bool FaInNode::reopenStream"
    )[0]
    id_switch_block = switch_device.split(
        'request->target_selector_mode == "id"'
    )[1].split('request->target_selector_mode == "name"')[0]
    id_miss_block = id_switch_block.split("if (device_id.empty())")[1]
    assert "display_name_matches" not in id_switch_block
    assert "displayName(dev)" not in id_miss_block
    assert "device id not found" in id_switch_block
    assert "一意に解決できる表示名" in spec
    assert "重複表示名では fail closed" in spec
    assert "configured display name が複数 source に一致する" in backend_doc


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
    assert "src/backends/pcm_file_reader_backend.cpp" in cmake_text
    assert "add_library(fa_in_node_core" in cmake_text
    assert "src/fa_in_node.cpp" in cmake_text
    assert "src/main.cpp" in cmake_text
    assert "target_link_libraries(fa_in_node_core fa_in_backends)" in cmake_text
    assert "target_link_libraries(fa_in_node fa_in_node_core)" in cmake_text


def test_node_header_does_not_store_alsa_pcm_handle() -> None:
    header_path = Path(__file__).parents[2] / "include" / "fa_in" / "fa_in_node.hpp"
    header = header_path.read_text(encoding="utf-8")

    assert "snd_pcm_t" not in header
    assert "pcm_handle_" not in header
    assert "alsa/asoundlib.h" not in header
    assert "std::unique_ptr<fa_in::backends::SourceBackend> source_backend_;" in header
    assert "using BackendFactory =" in header
    assert "BackendFactory backend_factory_;" in header


def test_colcon_runs_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_audio_config_validation_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_node_contract_test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml


def test_source_adapter_exposes_file_and_network_backends_but_no_dsp_surface() -> None:
    package_root = Path(__file__).parents[2]
    source_text = (package_root / "src" / "fa_in_node.cpp").read_text(encoding="utf-8")
    header_text = (package_root / "include" / "fa_in" / "fa_in_node.hpp").read_text(
        encoding="utf-8"
    )
    config_text = (package_root / "config" / "default.yaml").read_text(encoding="utf-8")
    spec_text = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")
    test_plan_text = (package_root / "docs" / "テスト設計.md").read_text(
        encoding="utf-8"
    )
    combined = "\n".join([source_text, header_text, config_text])

    forbidden_tokens = [
        "audio.file",
        "decode",
        "encoder",
        "gain",
        "normalize",
        "limiter",
        "resample",
        "volume",
        "sample_format",
        "channel_convert",
        "bit_depth_convert",
    ]
    for token in forbidden_tokens:
        assert token not in combined
    assert "pcm_file_reader" in combined
    assert "network_pcm_receiver" in combined
    assert "file.path" in combined
    assert "endpoint.uri" in combined
    assert "network.max_packet_bytes" in combined
    assert "FA-IN-SPEC-023" in spec_text
    assert "FA-IN-SPEC-023" in test_plan_text
    assert (package_root / "docs" / "backends" / "network_pcm_receiver.md").is_file()

from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_sidechain_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_sidechain"
        / "backends"
        / "internal_sidechain_detector.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_sidechain_detector.cpp"
    ).read_text(encoding="utf-8")


def test_example_config_separates_topics_from_stream_ids() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_sidechain"]["ros__parameters"]

    assert params["sidechain_topic"] == "fa_sidechain/input"
    assert params["control_topic"] == "fa_sidechain/control"
    assert params["sidechain_stream_id"] == "audio/sidechain/frame"
    assert params["control"]["stream_id"] == "audio/sidechain/control"
    assert params["sidechain_topic"] != params["sidechain_stream_id"]
    assert params["control_topic"] != params["control"]["stream_id"]
    assert params["sidechain_stream_id"] != params["control"]["stream_id"]
    assert params["detector"]["threshold_rms"] == 0.05
    assert params["detector"]["active_gain_db"] == -12.0
    assert params["detector"]["inactive_gain_db"] == 0.0
    assert params["control"]["sample_rate"] == 1000
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_launch_requires_explicit_config_file_and_node_name_without_package_default() -> None:
    launch_source = (package_root() / "launch" / "fa_sidechain.launch.py").read_text(encoding="utf-8")
    node_name_argument = launch_source.split('DeclareLaunchArgument(\n            "node_name"')[1].split(
        "        ),",
        1,
    )[0]
    config_argument = launch_source.split('DeclareLaunchArgument(\n            "config_file"')[1].split(
        "        ),",
        1,
    )[0]

    assert "default_value" not in node_name_argument
    assert "default_value" not in config_argument
    assert "FindPackageShare" not in launch_source
    assert "PathJoinSubstitution" not in launch_source
    assert "config/default.yaml" not in launch_source
    assert "parameters=[config_file]" in launch_source


def test_node_requires_parameters_without_runtime_defaults_and_validates_identity() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaSidechainNode::loadParameters")[1].split(
        "void FaSidechainNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("sidechain_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("control_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("sidechain_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("control.stream_id");' in load_parameters
    assert 'this->declare_parameter<int>("diagnostics.qos.depth");' in load_parameters
    assert 'this->declare_parameter<bool>("diagnostics.qos.reliable");' in load_parameters
    assert "readRequiredString(*this, \"sidechain_topic\")" in load_parameters
    assert "readRequiredString(*this, \"sidechain_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"control.stream_id\")" in load_parameters
    assert "readRequiredDouble(*this, \"detector.threshold_rms\")" in load_parameters
    assert "readRequiredBool(*this, \"diagnostics.qos.reliable\")" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "resolve_topic_name(config_.sidechain_topic)" in load_parameters
    assert "resolved_sidechain == resolved_control" in load_parameters
    assert "streamMatchesTopic(" in load_parameters
    assert "config_.sidechain_stream_id == config_.control_stream_id" in load_parameters
    assert "config_.expected_sample_rate > kMaxExpectedSampleRate" in load_parameters
    assert "config_.expected_channels > kMaxExpectedChannels" in load_parameters
    assert "config_.diagnostics_qos_depth <= 0" in load_parameters
    assert "rclcpp::SystemDefaultsQoS" not in source
    assert "std::max<int>(1, config_.qos_depth)" not in source


def test_node_validates_frame_contract_before_backend() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaSidechainNode::validateFrame")[1].split(
        "bool FaSidechainNode::buildControlFrame"
    )[0]
    handle_sidechain = source.split("void FaSidechainNode::handleSidechainFrame")[1].split(
        "bool FaSidechainNode::validateFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.sidechain_stream_id" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "validateFrame(*msg)" in handle_sidechain
    assert "backend_->detect(msg->data, control_data)" in handle_sidechain
    assert handle_sidechain.index("validateFrame(*msg)") < handle_sidechain.index(
        "backend_->detect(msg->data, control_data)"
    )


def test_backend_is_ros_free_and_owns_sample_rms_gain_bytes() -> None:
    node_source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()

    forbidden = ("rclcpp", "fa_interfaces", "AudioFrame", "diagnostic_msgs", "topic")
    for text in (backend_header, backend_source):
        for token in forbidden:
            assert token not in text
    assert "readSamples" not in node_source
    assert "calculateFrameRms" not in node_source
    assert "targetGainForRms" not in node_source
    assert "std::memcpy(output.data.data()" not in node_source
    assert "std::vector<uint8_t> & control_data" in backend_header
    assert "const ProcessStatus status = validateAndMeasure(input, rms, frame_count);" in backend_source
    assert "control_data = float32LeBytes(gain_sample);" in backend_source


def test_backend_drops_invalid_samples_without_clamp_and_commits_only_on_success() -> None:
    backend_source = read_backend_source()
    detect_code = backend_source.split("DetectionResult InternalSidechainDetectorBackend::detect")[1].split(
        "double dbToLinear"
    )[0]

    assert "std::clamp" not in backend_source
    assert "ProcessStatus::kNonFiniteInput" in backend_source
    assert "ProcessStatus::kOutOfRangeInput" in backend_source
    assert "ProcessStatus::kNonFiniteOutput" in detect_code
    assert "ProcessStatus::kOutOfRangeOutput" in detect_code
    assert "control_data = float32LeBytes(gain_sample);" in detect_code
    assert detect_code.index("control_data = float32LeBytes(gain_sample);") < detect_code.index(
        "last_rms_ = rms;"
    )
    assert "throw std::logic_error(\"unhandled sidechain detector backend process status\")" in backend_source


def test_node_output_metadata_and_diagnostics_use_backend_state() -> None:
    source = read_node_source()
    control_code = source.split("bool FaSidechainNode::buildControlFrame")[1].split(
        "size_t FaSidechainNode::bytesPerFrame"
    )[0]
    diagnostics = source.split("void FaSidechainNode::publishDiagnostics")[1].split(
        "}  // namespace fa_sidechain"
    )[0]

    assert "output.header = input.header;" in control_code
    assert "output.source_id = input.source_id;" in control_code
    assert "output.stream_id = config_.control_stream_id;" in control_code
    assert "output.sample_rate = static_cast<uint32_t>(config_.control_sample_rate);" in control_code
    assert "output.channels = 1;" in control_code
    assert "output.encoding = kEncodingFloat32;" in control_code
    assert "output.bit_depth = 32;" in control_code
    assert "output.layout = kInterleavedLayout;" in control_code
    assert "output.epoch = input.epoch;" in control_code
    assert "output.data = control_data;" in control_code
    assert "backend_->lastRms()" in diagnostics
    assert "backend_->lastGainLinear()" in diagnostics
    assert "backend_->lastActive()" in diagnostics
    assert "sidechain_stream_id" in diagnostics
    assert "control_stream_id" in diagnostics
    assert "diagnostics_qos_depth" in diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_sidechain_detector.md",
        "config/default.yaml",
        "launch/fa_sidechain.launch.py",
        "include/fa_sidechain/fa_sidechain_node.hpp",
        "include/fa_sidechain/backends/internal_sidechain_detector.hpp",
        "src/fa_sidechain_node.cpp",
        "src/backends/internal_sidechain_detector.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_sidechain_detector_backend.cpp",
        "test/unit/test_fa_sidechain_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_and_backend_gtest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

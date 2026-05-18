from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_ducking_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_ducking"
        / "backends"
        / "internal_sidechain_ducking.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_sidechain_ducking.cpp"
    ).read_text(encoding="utf-8")


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_ducking"]["ros__parameters"]

    assert params["program_topic"] == "fa_ducking/program"
    assert params["sidechain_topic"] == "fa_ducking/sidechain"
    assert params["output_topic"] == "fa_ducking/output"
    assert params["program_stream_id"] == "audio/program/frame"
    assert params["sidechain_stream_id"] == "audio/sidechain/frame"
    assert params["output"]["stream_id"] == "audio/ducked/frame"
    assert params["program_topic"] != params["program_stream_id"]
    assert params["sidechain_topic"] != params["sidechain_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert len(
        {
            params["program_stream_id"],
            params["sidechain_stream_id"],
            params["output"]["stream_id"],
        }
    ) == 3
    assert params["sidechain"]["threshold_rms"] == 0.05
    assert params["sidechain"]["max_age_ms"] == 100
    assert params["ducking"]["gain_db"] == -12.0
    assert params["ducking"]["attack_ms"] == 10.0
    assert params["ducking"]["release_ms"] == 250.0
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
    launch_source = (package_root() / "launch" / "fa_ducking.launch.py").read_text(encoding="utf-8")
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
    load_parameters = source.split("void FaDuckingNode::loadParameters")[1].split(
        "void FaDuckingNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("program_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("sidechain_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("program_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("sidechain_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<int>("diagnostics.qos.depth");' in load_parameters
    assert 'this->declare_parameter<bool>("diagnostics.qos.reliable");' in load_parameters
    assert "readRequiredString(*this, \"program_topic\")" in load_parameters
    assert "readRequiredString(*this, \"program_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredDouble(*this, \"sidechain.threshold_rms\")" in load_parameters
    assert "readRequiredBool(*this, \"diagnostics.qos.reliable\")" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "resolve_topic_name(config_.program_topic)" in load_parameters
    assert "resolved_program == resolved_sidechain" in load_parameters
    assert "streamMatchesTopic(" in load_parameters
    assert "config_.program_stream_id == config_.sidechain_stream_id" in load_parameters
    assert "config_.expected_sample_rate > kMaxExpectedSampleRate" in load_parameters
    assert "config_.expected_channels > kMaxExpectedChannels" in load_parameters
    assert "config_.diagnostics_qos_depth <= 0" in load_parameters
    assert "rclcpp::SystemDefaultsQoS" not in source
    assert "std::max<int>(1, config_.qos_depth)" not in source


def test_node_validates_frame_contract_before_backend() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaDuckingNode::validateFrame")[1].split(
        "size_t FaDuckingNode::bytesPerFrame"
    )[0]
    handle_program = source.split("void FaDuckingNode::handleProgramFrame")[1].split(
        "void FaDuckingNode::handleSidechainFrame"
    )[0]
    handle_sidechain = source.split("void FaDuckingNode::handleSidechainFrame")[1].split(
        "bool FaDuckingNode::validateFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != expected_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "validateFrame(*msg, config_.program_stream_id, \"program\")" in handle_program
    assert "backend_->processProgram(msg->data, nowNanoseconds(), output_data)" in handle_program
    assert handle_program.index("validateFrame(*msg, config_.program_stream_id") < handle_program.index(
        "backend_->processProgram(msg->data, nowNanoseconds(), output_data)"
    )
    assert "validateFrame(*msg, config_.sidechain_stream_id, \"sidechain\")" in handle_sidechain
    assert "backend_->observeSidechain(msg->data, nowNanoseconds())" in handle_sidechain


def test_backend_is_ros_free_and_owns_sample_state() -> None:
    node_source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()

    forbidden = ("rclcpp", "fa_interfaces", "AudioFrame", "diagnostic_msgs", "topic")
    for text in (backend_header, backend_source):
        for token in forbidden:
            assert token not in text
    assert "calculateFrameRms" not in node_source
    assert "readSamples" not in node_source
    assert "smoothingAlpha" not in node_source
    assert "smoothedGain" not in node_source
    assert "std::vector<uint8_t> & output" in backend_header
    assert "observeSidechain" in backend_header
    assert "processProgram" in backend_header
    assert "current_gain_" in backend_header
    assert "latest_sidechain_rms_" in backend_header
    assert "const ProcessStatus status = validateAndMeasure(input, rms, frame_count);" in backend_source


def test_backend_drops_invalid_samples_without_clamp_and_commits_only_on_success() -> None:
    backend_source = read_backend_source()
    process_program = backend_source.split("ProgramResult InternalSidechainDuckingBackend::processProgram")[1].split(
        "double dbToLinear"
    )[0]

    assert "std::clamp" not in backend_source
    assert "limiter" not in backend_source
    assert "normalize(" not in backend_source
    assert "ProcessStatus::kNonFiniteInput" in backend_source
    assert "ProcessStatus::kOutOfRangeInput" in backend_source
    assert "ProcessStatus::kNonFiniteOutput" in process_program
    assert "ProcessStatus::kOutOfRangeOutput" in process_program
    assert "output = std::move(candidate);" in process_program
    assert process_program.index("output = std::move(candidate);") < process_program.index(
        "current_gain_ = candidate_gain;"
    )
    assert "throw std::logic_error(\"unhandled sidechain ducking backend process status\")" in backend_source


def test_node_output_metadata_and_diagnostics_use_backend_state() -> None:
    source = read_node_source()
    handle_program = source.split("void FaDuckingNode::handleProgramFrame")[1].split(
        "void FaDuckingNode::handleSidechainFrame"
    )[0]
    diagnostics = source.split("void FaDuckingNode::publishDiagnostics")[1].split(
        "}  // namespace fa_ducking"
    )[0]

    assert "fa_interfaces::msg::AudioFrame out = *msg;" in handle_program
    assert "out.stream_id = config_.output_stream_id;" in handle_program
    assert "out.data = std::move(output_data);" in handle_program
    assert "backend_->currentGain()" in diagnostics
    assert "backend_->lastTargetGain()" in diagnostics
    assert "backend_->lastSidechainRms()" in diagnostics
    assert '"sidechain_max_age_ms"' in diagnostics
    assert '"last_sidechain_age_ms"' in diagnostics
    assert '"sidechain_max_age_ns"' not in diagnostics
    assert '"last_sidechain_age_ns"' not in diagnostics
    assert "nanosecondsToMillisecondsString(backend_->sidechainMaxAgeNs())" in diagnostics
    assert "nanosecondsToMillisecondsString(backend_->lastSidechainAgeNs())" in diagnostics
    assert "program_stream_id" in diagnostics
    assert "sidechain_stream_id" in diagnostics
    assert "output_stream_id" in diagnostics


def test_colcon_runs_pytest_and_backend_gtest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

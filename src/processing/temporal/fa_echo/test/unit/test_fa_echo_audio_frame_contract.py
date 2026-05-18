from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_echo_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_echo" / "backends" / "internal_feedback_echo.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_feedback_echo.cpp"
    ).read_text(encoding="utf-8")


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load(
        (package_root() / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_echo"]["ros__parameters"]

    assert params["input_topic"] == "fa_echo/input"
    assert params["output_topic"] == "fa_echo/output"
    assert params["input_stream_id"] == "audio/buffered/mic"
    assert params["output"]["stream_id"] == "audio/echo/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["output_topic"] != params["input_stream_id"]
    assert params["input_topic"] != params["output"]["stream_id"]
    assert params["echo"]["delay_ms"] == 250.0
    assert params["echo"]["feedback_gain"] == 0.35
    assert params["echo"]["wet_gain"] == 0.4
    assert params["echo"]["dry_gain"] == 0.8
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is True


def test_launch_requires_explicit_config_file_without_package_default() -> None:
    launch_source = (package_root() / "launch" / "fa_echo.launch.py").read_text(
        encoding="utf-8"
    )
    config_argument = launch_source.split('DeclareLaunchArgument(\n            "config_file"')[1].split(
        "        ),",
        1,
    )[0]

    assert "default_value" not in launch_source
    assert "FindPackageShare" not in launch_source
    assert "PathJoinSubstitution" not in launch_source
    assert "config/default.yaml" not in launch_source
    assert "parameters=[config_file]" in launch_source


def test_echo_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "std::clamp",
        "clip",
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "cutoff_hz",
        "center_hz",
        "denoise",
        "reverb",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_echo_parameters_are_required_without_runtime_defaults_and_range_checked() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaEchoNode::loadParameters")[1].split(
        "void FaEchoNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<double>("echo.delay_ms");' in load_parameters
    assert 'this->declare_parameter<double>("echo.feedback_gain");' in load_parameters
    assert 'this->declare_parameter<double>("echo.wet_gain");' in load_parameters
    assert 'this->declare_parameter<double>("echo.dry_gain");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredDouble(*this, \"echo.delay_ms\")" in load_parameters
    assert "readRequiredBool(*this, \"qos.reliable\")" in load_parameters
    assert "readRequiredInt(*this, \"diagnostics.qos.depth\")" in load_parameters
    assert "readRequiredBool(*this, \"diagnostics.qos.reliable\")" in load_parameters
    assert "rclcpp::SystemDefaultsQoS()" not in source
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "config_.input_topic == config_.output_topic" in load_parameters
    assert "config_.input_stream_id.empty()" in load_parameters
    assert "config_.output_stream_id.empty()" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, config_.input_topic)" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, config_.output_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, config_.input_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, config_.output_topic)" in load_parameters
    assert "resolve_topic_name(config_.input_topic)" in load_parameters
    assert "resolve_topic_name(config_.output_topic)" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, resolved_input_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, resolved_output_topic)" in load_parameters
    assert "config_.input_stream_id == config_.output_stream_id" in load_parameters
    assert "!isFinite(config_.delay_ms) || config_.delay_ms <= 0.0" in load_parameters
    assert "!isFinite(config_.feedback_gain) || std::abs(config_.feedback_gain) >= 1.0" in load_parameters
    assert "!isFinite(config_.wet_gain)" in load_parameters
    assert "!isFinite(config_.dry_gain)" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_sample_rate > kMaxExpectedSampleRate" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_channels > kMaxExpectedChannels" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "config_.diagnostics_qos_depth <= 0" in load_parameters
    assert "delay_samples == 0" in load_parameters
    assert "std::max<int>(1, config_.qos_depth)" not in source


def test_echo_validates_frame_contract_before_processing() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaEchoNode::validateFrame")[1].split(
        "bool FaEchoNode::applyEcho"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_echo_delegates_sample_validation_and_state_mutation_to_backend() -> None:
    source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()
    handle_frame = source.split("void FaEchoNode::handleFrame")[1].split(
        "bool FaEchoNode::validateFrame"
    )[0]
    apply_echo = source.split("bool FaEchoNode::applyEcho")[1].split(
        "size_t FaEchoNode::bytesPerFrame"
    )[0]
    process = backend_source.split("ProcessResult InternalFeedbackEchoBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "validateSamples" not in source
    assert "if (!applyEcho(*msg, out))" in handle_frame
    assert "backend_->process(in.source_id, in.data, out.data)" in apply_echo
    assert "const char * status_message = backends::processStatusMessage(result.status);" in apply_echo
    assert "enum class ProcessStatus" in backend_header
    assert "ProcessResult" in backend_header
    assert "!isNormalizedSample(input_sample)" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeInput, false};" in process
    assert "std::clamp" not in process


def test_echo_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_echo = source.split("bool FaEchoNode::applyEcho")[1].split(
        "size_t FaEchoNode::bytesPerFrame"
    )[0]

    assert "out = in;" in apply_echo
    assert "out.stream_id = config_.output_stream_id;" in apply_echo
    assert ".rms" not in apply_echo
    assert ".peak" not in apply_echo
    assert ".vad" not in apply_echo


def test_echo_uses_per_channel_ring_buffers_and_feedback_recurrence() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalFeedbackEchoBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "std::vector<std::vector<float>> delay_buffers_{};" in header
    assert "std::vector<size_t> delay_positions_{};" in header
    assert "std::vector<std::vector<float>> next_buffers = delay_buffers_;" in process
    assert "std::vector<size_t> next_positions = delay_positions_;" in process
    assert "const size_t delay_index = next_positions[channel_index];" in process
    assert "const float delayed_sample = next_buffers[channel_index][delay_index];" in process
    assert "config_.dry_gain * static_cast<double>(input_sample)" in process
    assert "config_.wet_gain * static_cast<double>(delayed_sample)" in process
    assert "static_cast<double>(input_sample) +" in process
    assert "config_.feedback_gain * static_cast<double>(delayed_sample)" in process
    assert "next_buffers[channel_index][delay_index] = next_state_float;" in process
    assert "next_positions[channel_index] = (delay_index + 1U) % config_.delay_samples;" in process


def test_echo_rejects_invalid_output_or_state_without_clipping() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalFeedbackEchoBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "bool isNormalizedSample(float value)" in backend_source
    assert "value >= kMinNormalizedSample && value <= kMaxNormalizedSample" in backend_source
    assert "!isFinite(output_sample)" in process
    assert "!isFinite(next_state)" in process
    assert "!isNormalizedSample(output_float)" in process
    assert "!isNormalizedSample(next_state_float)" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeOutput, false};" in process
    assert "std::clamp" not in process


def test_echo_resets_state_when_source_id_changes() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    reset_state = backend_source.split("void InternalFeedbackEchoBackend::resetDelayState")[1].split(
        "ProcessResult InternalFeedbackEchoBackend::process"
    )[0]
    process = backend_source.split("ProcessResult InternalFeedbackEchoBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_echo = source.split("bool FaEchoNode::applyEcho")[1].split(
        "size_t FaEchoNode::bytesPerFrame"
    )[0]

    assert "std::vector<float>(config_.delay_samples, kSilenceSample)" in reset_state
    assert "positions.assign(static_cast<size_t>(config_.channels), 0U);" in reset_state
    assert "const bool needs_initialization = current_source_id_.empty();" in process
    assert "const bool source_changed = !current_source_id_.empty() && source_id != current_source_id_;" in process
    assert "resetDelayState(next_buffers, next_positions);" in process
    assert "return ProcessResult{ProcessStatus::kInvalidState, false};" in process
    assert "source_resets_.fetch_add(1);" in apply_echo
    assert "current_source_id_ = source_id;" in process
    assert "delay_buffers_ = std::move(next_buffers);" in process
    assert "delay_positions_ = std::move(next_positions);" in process


def test_diagnostics_include_echo_source_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaEchoNode::publishDiagnostics")[1].split(
        "}  // namespace fa_echo"
    )[0]

    assert 'status.name = "fa_echo";' in diagnostics
    assert 'pushKeyValue(status, "delay_ms", std::to_string(config_.delay_ms));' in diagnostics
    assert 'pushKeyValue(status, "delay_samples", std::to_string(backend_->delaySamples()));' in diagnostics
    assert 'pushKeyValue(status, "feedback_gain", std::to_string(backend_->feedbackGain()));' in diagnostics
    assert 'pushKeyValue(status, "wet_gain", std::to_string(backend_->wetGain()));' in diagnostics
    assert 'pushKeyValue(status, "dry_gain", std::to_string(backend_->dryGain()));' in diagnostics
    assert 'pushKeyValue(status, "current_source_id", backend_->currentSourceId());' in diagnostics
    assert 'pushKeyValue(status, "input_stream_id", config_.input_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "output_stream_id", config_.output_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "messages_in", std::to_string(messages_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "messages_out", std::to_string(messages_out_.load()));' in diagnostics
    assert (
        'pushKeyValue(status, "messages_dropped", std::to_string(messages_dropped_.load()));'
        in diagnostics
    )
    assert 'pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_feedback_echo.md",
        "config/default.yaml",
        "launch/fa_echo.launch.py",
        "include/fa_echo/fa_echo_node.hpp",
        "include/fa_echo/backends/internal_feedback_echo.hpp",
        "src/fa_echo_node.cpp",
        "src/backends/internal_feedback_echo.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_feedback_echo_backend.cpp",
        "test/cpp/test_echo_graph.cpp",
        "test/unit/test_fa_echo_audio_frame_contract.py",
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
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

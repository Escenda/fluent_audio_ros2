from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_reverb_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_reverb" / "backends" / "internal_feedback_delay.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_feedback_delay.cpp"
    ).read_text(encoding="utf-8")


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load(
        (package_root() / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_reverb"]["ros__parameters"]

    assert params["input_topic"] == "fa_reverb/input"
    assert params["output_topic"] == "fa_reverb/output"
    assert params["input_stream_id"] == "audio/echo/mic"
    assert params["output"]["stream_id"] == "audio/reverb/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["output_topic"] != params["input_stream_id"]
    assert params["input_topic"] != params["output"]["stream_id"]
    assert params["reverb"]["room_size"] == 0.72
    assert params["reverb"]["damping"] == 0.35
    assert params["reverb"]["wet_gain"] == 0.32
    assert params["reverb"]["dry_gain"] == 0.68
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_launch_requires_explicit_config_file_without_package_default() -> None:
    launch_source = (package_root() / "launch" / "fa_reverb.launch.py").read_text(
        encoding="utf-8"
    )
    config_argument = launch_source.split('DeclareLaunchArgument(\n            "config_file"')[1].split(
        "        ),",
        1,
    )[0]

    assert "default_value" not in config_argument
    assert "FindPackageShare" not in launch_source
    assert "PathJoinSubstitution" not in launch_source
    assert "config/default.yaml" not in launch_source
    assert "parameters=[config_file]" in launch_source


def test_reverb_does_not_hide_other_processing_or_io_responsibilities() -> None:
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
        "alsa",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_reverb_parameters_are_declared_without_runtime_defaults() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaReverbNode::loadParameters")[1].split(
        "void FaReverbNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<double>("reverb.room_size");' in load_parameters
    assert 'this->declare_parameter<double>("reverb.damping");' in load_parameters
    assert 'this->declare_parameter<double>("reverb.wet_gain");' in load_parameters
    assert 'this->declare_parameter<double>("reverb.dry_gain");' in load_parameters
    assert 'this->declare_parameter<int>("expected.sample_rate");' in load_parameters
    assert 'this->declare_parameter<int>("expected.channels");' in load_parameters
    assert 'this->declare_parameter<std::string>("expected.encoding");' in load_parameters
    assert 'this->declare_parameter<int>("expected.bit_depth");' in load_parameters
    assert 'this->declare_parameter<std::string>("expected.layout");' in load_parameters
    assert 'this->declare_parameter<int>("qos.depth");' in load_parameters
    assert 'this->declare_parameter<bool>("qos.reliable");' in load_parameters
    assert 'this->declare_parameter<int>("diagnostics.publish_period_ms");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredDouble(*this, \"reverb.room_size\")" in load_parameters
    assert "readRequiredBool(*this, \"qos.reliable\")" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line


def test_reverb_parameters_are_range_checked() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaReverbNode::loadParameters")[1].split(
        "void FaReverbNode::configureBackend"
    )[0]

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
    assert "!isFinite(config_.room_size) || config_.room_size < 0.0 || config_.room_size > 1.0" in load_parameters
    assert "!isFinite(config_.damping) || config_.damping < 0.0 || config_.damping > 1.0" in load_parameters
    assert "!isFinite(config_.wet_gain) || config_.wet_gain < 0.0 || config_.wet_gain > 1.0" in load_parameters
    assert "!isFinite(config_.dry_gain) || config_.dry_gain < 0.0 || config_.dry_gain > 1.0" in load_parameters
    assert "(config_.wet_gain + config_.dry_gain) > 1.0" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_sample_rate > kMaxExpectedSampleRate" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_channels > kMaxExpectedChannels" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "std::max<int>(1, config_.qos_depth)" not in source


def test_reverb_validates_float32_interleaved_frame_contract() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaReverbNode::validateFrame")[1].split(
        "bool FaReverbNode::applyReverb"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_reverb_delegates_sample_validation_and_state_mutation_to_backend() -> None:
    source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()
    handle_frame = source.split("void FaReverbNode::handleFrame")[1].split(
        "bool FaReverbNode::validateFrame"
    )[0]
    apply_reverb = source.split("bool FaReverbNode::applyReverb")[1].split(
        "size_t FaReverbNode::bytesPerFrame"
    )[0]
    process = backend_source.split("ProcessResult InternalFeedbackDelayBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "validateSamples" not in source
    assert "if (!applyReverb(*msg, out))" in handle_frame
    assert "backend_->process(in.source_id, in.data, out.data)" in apply_reverb
    assert "const char * status_message = backends::processStatusMessage(result.status);" in apply_reverb
    assert "enum class ProcessStatus" in backend_header
    assert "ProcessResult" in backend_header
    assert "!isNormalizedSample(input_sample)" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeInput, false};" in process
    assert "std::clamp" not in process


def test_reverb_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_reverb = source.split("bool FaReverbNode::applyReverb")[1].split(
        "size_t FaReverbNode::bytesPerFrame"
    )[0]

    assert "out = in;" in apply_reverb
    assert "out.stream_id = config_.output_stream_id;" in apply_reverb
    assert ".rms" not in apply_reverb
    assert ".peak" not in apply_reverb
    assert ".vad" not in apply_reverb


def test_reverb_uses_per_channel_multi_tap_feedback_delay_backend() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalFeedbackDelayBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "std::vector<std::vector<DelayLineState>> delay_lines_{};" in header
    assert "std::vector<size_t> delay_samples_{};" in header
    assert "std::vector<std::vector<DelayLineState>> next_state = delay_lines_;" in process
    assert "resetReverbState(next_state);" in process
    assert "for (DelayLineState & line : next_state[channel_index])" in process
    assert "const float delayed_sample = line.buffer[delay_index];" in process
    assert "wet_sum += static_cast<double>(delayed_sample);" in process
    assert "effective_feedback_gain_ * filtered_sample" in process
    assert "line.buffer[delay_index] = next_state_float;" in process
    assert "line.position = (delay_index + 1U) % line.buffer.size();" in process
    assert "const double wet_sample = wet_sum / static_cast<double>(delay_samples_.size());" in process


def test_reverb_rejects_invalid_output_or_state_without_clipping() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalFeedbackDelayBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "bool isNormalizedSample(float value)" in backend_source
    assert "value >= kMinNormalizedSample && value <= kMaxNormalizedSample" in backend_source
    assert "!isFinite(filtered_sample)" in process
    assert "!isFinite(next_feedback_state)" in process
    assert "!isNormalizedSample(filtered_float) || !isNormalizedSample(next_state_float)" in process
    assert "!isFinite(output_sample)" in process
    assert "!isNormalizedSample(output_float)" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeOutput, false};" in process
    assert "std::clamp" not in process
    assert "delay_lines_ = std::move(next_state);" in process


def test_reverb_resets_state_when_source_id_changes() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    reset_state = backend_source.split("void InternalFeedbackDelayBackend::resetReverbState")[1].split(
        "bool InternalFeedbackDelayBackend::validateReverbState"
    )[0]
    process = backend_source.split("ProcessResult InternalFeedbackDelayBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_reverb = source.split("bool FaReverbNode::applyReverb")[1].split(
        "size_t FaReverbNode::bytesPerFrame"
    )[0]

    assert "state.assign(" in reset_state
    assert "channel_state[line_index].buffer.assign(delay_samples_[line_index], kSilenceSample);" in reset_state
    assert "channel_state[line_index].position = 0U;" in reset_state
    assert "channel_state[line_index].filter_state = kSilenceSample;" in reset_state
    assert "const bool needs_initialization = current_source_id_.empty();" in process
    assert "const bool source_changed = !current_source_id_.empty() && source_id != current_source_id_;" in process
    assert "source_resets_.fetch_add(1);" in apply_reverb
    assert "current_source_id_ = source_id;" in process
    assert "delay_lines_ = std::move(next_state);" in process


def test_diagnostics_include_reverb_source_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaReverbNode::publishDiagnostics")[1].split(
        "}  // namespace fa_reverb"
    )[0]

    assert 'status.name = "fa_reverb";' in diagnostics
    assert 'pushKeyValue(status, "room_size", std::to_string(config_.room_size));' in diagnostics
    assert 'pushKeyValue(status, "damping", std::to_string(config_.damping));' in diagnostics
    assert 'pushKeyValue(status, "wet_gain", std::to_string(backend_->wetGain()));' in diagnostics
    assert 'pushKeyValue(status, "dry_gain", std::to_string(backend_->dryGain()));' in diagnostics
    assert 'std::to_string(backend_->effectiveFeedbackGain())' in diagnostics
    assert 'pushKeyValue(status, "delay_lines", std::to_string(backend_->delayLineCount()));' in diagnostics
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
        "docs/backends/internal_feedback_delay.md",
        "config/default.yaml",
        "launch/fa_reverb.launch.py",
        "include/fa_reverb/fa_reverb_node.hpp",
        "include/fa_reverb/backends/internal_feedback_delay.hpp",
        "src/fa_reverb_node.cpp",
        "src/backends/internal_feedback_delay.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_feedback_delay_backend.cpp",
        "test/unit/test_fa_reverb_audio_frame_contract.py",
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
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

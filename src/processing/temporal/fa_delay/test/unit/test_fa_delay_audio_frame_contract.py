from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_delay_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_delay" / "backends" / "internal_sample_delay.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_sample_delay.cpp"
    ).read_text(encoding="utf-8")


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_delay"]["ros__parameters"]

    assert params["input_topic"] == "fa_delay/input"
    assert params["output_topic"] == "fa_delay/output"
    assert params["input_stream_id"] == "audio/buffered/mic"
    assert params["output"]["stream_id"] == "audio/delayed/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["delay"]["ms"] == 250.0
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
    launch_source = (package_root() / "launch" / "fa_delay.launch.py").read_text(
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


def test_delay_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "std::clamp",
        "clip",
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "gain.linear",
        "threshold.linear",
        "cutoff_hz",
        "center_hz",
        "denoise",
        "reverb",
        "echo",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_delay_parameters_are_required_without_runtime_defaults_and_range_checked() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaDelayNode::loadParameters")[1].split(
        "void FaDelayNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<double>("delay.ms");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredDouble(*this, \"delay.ms\")" in load_parameters
    assert "readRequiredBool(*this, \"qos.reliable\")" in load_parameters
    assert "readRequiredInt(*this, \"diagnostics.qos.depth\")" in load_parameters
    assert "readRequiredBool(*this, \"diagnostics.qos.reliable\")" in load_parameters
    assert "rclcpp::SystemDefaultsQoS()" not in source
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "sameIdentityString(config_.input_stream_id, config_.input_topic)" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, resolved_input_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, config_.output_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, resolved_output_topic)" in load_parameters
    assert "config_.input_stream_id == config_.output_stream_id" in load_parameters
    assert "!isFinite(config_.delay_ms) || config_.delay_ms <= 0.0" in load_parameters
    assert "delay.ms must be > 0 and finite" in load_parameters
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


def test_delay_converts_ms_to_whole_samples_from_expected_sample_rate() -> None:
    source = read_node_source()
    configure_backend = source.split("void FaDelayNode::configureBackend")[1].split(
        "void FaDelayNode::setupInterfaces"
    )[0]

    assert (
        "config_.delay_ms * static_cast<double>(config_.expected_sample_rate) / 1000.0"
        in configure_backend
    )
    assert "const size_t delay_samples = static_cast<size_t>(std::llround(raw_delay_samples));" in configure_backend
    assert "delay.ms must convert to at least 1 sample" in configure_backend


def test_delay_validates_frame_contract_before_processing() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaDelayNode::validateFrame")[1].split(
        "bool FaDelayNode::applyDelay"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_delay_delegates_sample_validation_and_state_mutation_to_backend() -> None:
    source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()
    handle_frame = source.split("void FaDelayNode::handleFrame")[1].split(
        "bool FaDelayNode::validateFrame"
    )[0]
    apply_delay = source.split("bool FaDelayNode::applyDelay")[1].split(
        "size_t FaDelayNode::bytesPerFrame"
    )[0]
    process = backend_source.split("ProcessResult InternalSampleDelayBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "validateSamples" not in source
    assert "ensureDelayState" not in source
    assert "if (!applyDelay(*msg, out))" in handle_frame
    assert "backend_->process(in.source_id, in.data, out.data)" in apply_delay
    assert "const char * status_message = backends::processStatusMessage(result.status);" in apply_delay
    assert "enum class ProcessStatus" in backend_header
    assert "!std::isfinite(input_sample)" in process
    assert "!isNormalizedSample(input_sample)" in process
    assert "std::clamp" not in process


def test_delay_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_delay = source.split("bool FaDelayNode::applyDelay")[1].split(
        "size_t FaDelayNode::bytesPerFrame"
    )[0]

    assert "out = in;" in apply_delay
    assert "out.stream_id = config_.output_stream_id;" in apply_delay
    assert ".rms" not in apply_delay
    assert ".peak" not in apply_delay
    assert ".vad" not in apply_delay


def test_delay_uses_per_channel_buffers_initialized_with_silence() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    reset_state = backend_source.split("void InternalSampleDelayBackend::resetDelayState")[1].split(
        "bool InternalSampleDelayBackend::validateDelayState"
    )[0]
    process = backend_source.split("ProcessResult InternalSampleDelayBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "std::vector<std::deque<float>> delay_buffers_{};" in header
    assert "std::deque<float>(config_.delay_samples, kSilenceSample)" in reset_state
    assert "std::vector<std::deque<float>> next_buffers = delay_buffers_;" in process
    assert "const float delayed_sample = next_buffers[channel_index].front();" in process
    assert "next_buffers[channel_index].pop_front();" in process
    assert "next_buffers[channel_index].push_back(input_sample);" in process
    assert "writeFloat32Le(next_output, sample_index, delayed_sample);" in process


def test_delay_resets_state_when_source_id_changes() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalSampleDelayBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_delay = source.split("bool FaDelayNode::applyDelay")[1].split(
        "size_t FaDelayNode::bytesPerFrame"
    )[0]

    assert "const bool needs_initialization = current_source_id_.empty();" in process
    assert "const bool source_changed = !current_source_id_.empty() && source_id != current_source_id_;" in process
    assert "resetDelayState(next_buffers);" in process
    assert "current_source_id_ = source_id;" in process
    assert "delay_buffers_ = std::move(next_buffers);" in process
    assert "source_resets_.fetch_add(1);" in apply_delay


def test_diagnostics_include_delay_source_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaDelayNode::publishDiagnostics")[1].split(
        "}  // namespace fa_delay"
    )[0]

    assert 'status.name = "fa_delay";' in diagnostics
    assert 'pushKeyValue(status, "delay_ms", std::to_string(config_.delay_ms));' in diagnostics
    assert 'pushKeyValue(status, "delay_samples", std::to_string(backend_->delaySamples()));' in diagnostics
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
        "docs/backends/internal_sample_delay.md",
        "config/default.yaml",
        "launch/fa_delay.launch.py",
        "include/fa_delay/fa_delay_node.hpp",
        "include/fa_delay/backends/internal_sample_delay.hpp",
        "src/fa_delay_node.cpp",
        "src/backends/internal_sample_delay.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_sample_delay_backend.cpp",
        "test/unit/test_fa_delay_audio_frame_contract.py",
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
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml

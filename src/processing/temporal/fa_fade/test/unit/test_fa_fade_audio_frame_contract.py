from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_fade_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_fade" / "backends" / "internal_linear_fade.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_linear_fade.cpp"
    ).read_text(encoding="utf-8")


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_fade"]["ros__parameters"]

    assert params["input_topic"] == "fa_fade/input"
    assert params["output_topic"] == "fa_fade/output"
    assert params["input_stream_id"] == "audio/buffered/mic"
    assert params["output"]["stream_id"] == "audio/faded/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["fade"]["mode"] == "fade_in"
    assert params["fade"]["duration_frames"] == 16000
    assert params["fade"]["initial_position_frames"] == 0
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
    launch_source = (package_root() / "launch" / "fa_fade.launch.py").read_text(
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


def test_fade_does_not_hide_other_processing_or_io_responsibilities() -> None:
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
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_fade_parameters_are_required_without_runtime_defaults_and_range_checked() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaFadeNode::loadParameters")[1].split(
        "void FaFadeNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("fade.mode");' in load_parameters
    assert 'this->declare_parameter<int>("fade.duration_frames");' in load_parameters
    assert 'this->declare_parameter<int>("fade.initial_position_frames");' in load_parameters
    assert 'this->declare_parameter<bool>("qos.reliable");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"fade.mode\")" in load_parameters
    assert "readRequiredInt(*this, \"fade.duration_frames\")" in load_parameters
    assert "readRequiredInt(*this, \"fade.initial_position_frames\")" in load_parameters
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
    assert "parseFadeMode(config_.mode);" in load_parameters
    assert "fade.mode must be one of fade_in, fade_out" in source
    assert "config_.duration_frames <= 0" in load_parameters
    assert "fade.duration_frames must be > 0" in load_parameters
    assert "config_.initial_position_frames < 0" in load_parameters
    assert "fade.initial_position_frames must be >= 0" in load_parameters
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


def test_fade_validates_frame_contract_before_backend() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaFadeNode::validateFrame")[1].split(
        "bool FaFadeNode::applyFade"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_fade_backend_owns_sample_processing_and_position_state() -> None:
    source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()
    handle_frame = source.split("void FaFadeNode::handleFrame")[1].split(
        "bool FaFadeNode::validateFrame"
    )[0]
    apply_fade = source.split("bool FaFadeNode::applyFade")[1].split(
        "size_t FaFadeNode::bytesPerFrame"
    )[0]
    process = backend_source.split("ProcessResult InternalLinearFadeBackend::process")[1].split(
        "const char * fadeModeName"
    )[0]

    assert "gainAtPosition" not in source
    assert "uint64_t position_frames_{0U};" in backend_header
    assert "position_frames_(config.initial_position_frames)" in backend_source
    assert "if (!applyFade(*msg, out))" in handle_frame
    assert "backend_->process(in.data, output_data)" in apply_fade
    assert "const char * status_message = backends::processStatusMessage(result.status);" in apply_fade
    assert "enum class ProcessStatus" in backend_header
    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "output = std::move(next_output);" in process
    assert "position_frames_ += static_cast<uint64_t>(frame_count);" in process
    assert "std::clamp" not in process


def test_fade_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_fade = source.split("bool FaFadeNode::applyFade")[1].split(
        "size_t FaFadeNode::bytesPerFrame"
    )[0]

    assert "out = in;" in apply_fade
    assert "out.stream_id = config_.output_stream_id;" in apply_fade
    assert "out.data = std::move(output_data);" in apply_fade
    assert ".rms" not in apply_fade
    assert ".peak" not in apply_fade
    assert ".vad" not in apply_fade


def test_fade_uses_linear_position_counter_across_accepted_frames() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalLinearFadeBackend::process")[1].split(
        "const char * fadeModeName"
    )[0]
    gain_at_position = backend_source.split("double InternalLinearFadeBackend::gainAtPosition")[1].split(
        "ProcessResult InternalLinearFadeBackend::process"
    )[0]

    assert "position_frames_ + static_cast<uint64_t>(sample_index / channels)" in process
    assert "position_frames_ += static_cast<uint64_t>(frame_count);" in process
    assert "static_cast<double>(position_frames) / static_cast<double>(config_.duration_frames)" in gain_at_position
    assert "return 1.0;" in gain_at_position
    assert "return progress;" in gain_at_position
    assert "return 0.0;" in gain_at_position
    assert "return 1.0 - progress;" in gain_at_position


def test_fade_drops_invalid_samples_instead_of_clamping_or_normalizing() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalLinearFadeBackend::process")[1].split(
        "const char * fadeModeName"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "const double faded = static_cast<double>(sample) * gainAtPosition" in process
    assert "!std::isfinite(faded)" in process
    assert "faded < static_cast<double>(kMinNormalizedSample)" in process
    assert "faded > static_cast<double>(kMaxNormalizedSample)" in process
    assert "!std::isfinite(output_sample)" in process
    assert "std::clamp" not in process


def test_diagnostics_include_mode_duration_position_stream_identity_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaFadeNode::publishDiagnostics")[1].split(
        "}  // namespace fa_fade"
    )[0]

    assert 'status.name = "fa_fade";' in diagnostics
    assert 'pushKeyValue(status, "mode", config_.mode);' in diagnostics
    assert 'pushKeyValue(status, "duration_frames", std::to_string(config_.duration_frames));' in diagnostics
    assert 'backend_->positionFrames()' in diagnostics
    assert 'pushKeyValue(status, "input_stream_id", config_.input_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "output_stream_id", config_.output_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_linear_fade.md",
        "config/default.yaml",
        "launch/fa_fade.launch.py",
        "include/fa_fade/fa_fade_node.hpp",
        "include/fa_fade/backends/internal_linear_fade.hpp",
        "src/fa_fade_node.cpp",
        "src/backends/internal_linear_fade.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_linear_fade_backend.cpp",
        "test/unit/test_fa_fade_audio_frame_contract.py",
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
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

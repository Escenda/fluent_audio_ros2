from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_window_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_window" / "backends" / "internal_window_function.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_window_function.cpp"
    ).read_text(encoding="utf-8")


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_window"]["ros__parameters"]

    assert params["input_topic"] == "fa_window/input"
    assert params["output_topic"] == "fa_window/output"
    assert params["input_stream_id"] == "audio/buffered/mic"
    assert params["output"]["stream_id"] == "audio/windowed/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["window"]["type"] == "hann"
    assert params["window"]["expected_frames"] == 512
    assert params["window"]["strict_frame_count"] is True
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_launch_requires_explicit_config_file_without_package_default() -> None:
    launch_source = (package_root() / "launch" / "fa_window.launch.py").read_text(
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


def test_window_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "std::clamp",
        "clip",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "gain.linear",
        "threshold.linear",
        "cutoff_hz",
        "center_hz",
        "denoise",
        "limiter",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_window_parameters_are_required_without_runtime_defaults_and_range_checked() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaWindowNode::loadParameters")[1].split(
        "void FaWindowNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("window.type");' in load_parameters
    assert 'this->declare_parameter<int>("window.expected_frames");' in load_parameters
    assert 'this->declare_parameter<bool>("window.strict_frame_count");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"window.type\")" in load_parameters
    assert "readRequiredInt(*this, \"window.expected_frames\")" in load_parameters
    assert "readRequiredBool(*this, \"window.strict_frame_count\")" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line

    assert "sameIdentityString(config_.input_stream_id, config_.input_topic)" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, resolved_input_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, config_.output_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, resolved_output_topic)" in load_parameters
    assert "config_.input_stream_id == config_.output_stream_id" in load_parameters
    assert "parseWindowType(config_.window_type);" in load_parameters
    assert "window.type must be one of hann, hamming" in source
    assert "config_.expected_frames <= 1" in load_parameters
    assert "window.expected_frames must be > 1" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_sample_rate > kMaxExpectedSampleRate" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_channels > kMaxExpectedChannels" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters


def test_window_validates_frame_contract_before_backend() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaWindowNode::validateFrame")[1].split(
        "bool FaWindowNode::applyWindow"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_window_backend_owns_frame_policy_coefficients_and_sample_processing() -> None:
    source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()
    apply_window = source.split("bool FaWindowNode::applyWindow")[1].split(
        "size_t FaWindowNode::bytesPerFrame"
    )[0]
    process = backend_source.split("ProcessResult InternalWindowFunctionBackend::process")[1].split(
        "const char * windowTypeName"
    )[0]
    coefficient_at = backend_source.split("double InternalWindowFunctionBackend::coefficientAt")[1].split(
        "ProcessResult InternalWindowFunctionBackend::process"
    )[0]

    assert "computeCoefficients" not in source
    assert "coefficientAt" not in source
    assert "backend_->process(in.data, output_data)" in apply_window
    assert "const char * status_message = backends::processStatusMessage(result.status);" in apply_window
    assert "enum class ProcessStatus" in backend_header
    assert "config_.strict_frame_count && frame_count != config_.expected_frames" in process
    assert "!config_.strict_frame_count && frame_count <= 1U" in process
    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "output = std::move(next_output);" in process
    assert "return 0.5 * (1.0 - std::cos(phase));" in coefficient_at
    assert "return 0.54 - (0.46 * std::cos(phase));" in coefficient_at


def test_window_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_window = source.split("bool FaWindowNode::applyWindow")[1].split(
        "size_t FaWindowNode::bytesPerFrame"
    )[0]

    assert "out = in;" in apply_window
    assert "out.stream_id = config_.output_stream_id;" in apply_window
    assert "out.data = std::move(output_data);" in apply_window
    assert ".rms" not in apply_window
    assert ".peak" not in apply_window
    assert ".vad" not in apply_window


def test_window_drops_invalid_samples_instead_of_clamping_or_normalizing() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalWindowFunctionBackend::process")[1].split(
        "const char * windowTypeName"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "const double windowed = static_cast<double>(sample) * coefficientAt" in process
    assert "!std::isfinite(windowed)" in process
    assert "windowed < static_cast<double>(kMinNormalizedSample)" in process
    assert "windowed > static_cast<double>(kMaxNormalizedSample)" in process
    assert "!std::isfinite(output_sample)" in process
    assert "std::clamp" not in process


def test_diagnostics_include_type_frame_policy_stream_identity_last_frame_count_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaWindowNode::publishDiagnostics")[1].split(
        "}  // namespace fa_window"
    )[0]

    assert 'status.name = "fa_window";' in diagnostics
    assert 'pushKeyValue(status, "window_type", config_.window_type);' in diagnostics
    assert 'pushKeyValue(status, "expected_frames", std::to_string(config_.expected_frames));' in diagnostics
    assert 'pushKeyValue(status, "strict_frame_count", config_.strict_frame_count ? "true" : "false");' in diagnostics
    assert 'pushKeyValue(status, "last_frame_count", std::to_string(last_frame_count_.load()));' in diagnostics
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
        "docs/backends/internal_window_function.md",
        "config/default.yaml",
        "launch/fa_window.launch.py",
        "include/fa_window/fa_window_node.hpp",
        "include/fa_window/backends/internal_window_function.hpp",
        "src/fa_window_node.cpp",
        "src/backends/internal_window_function.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_window_function_backend.cpp",
        "test/unit/test_fa_window_audio_frame_contract.py",
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

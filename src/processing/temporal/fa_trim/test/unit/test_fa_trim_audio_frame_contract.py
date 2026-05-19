from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_trim_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_trim" / "backends" / "internal_frame_trim.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_frame_trim.cpp"
    ).read_text(encoding="utf-8")


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_trim"]["ros__parameters"]

    assert params["input_topic"] == "fa_trim/input"
    assert params["output_topic"] == "fa_trim/output"
    assert params["input_stream_id"] == "audio/windowed/mic"
    assert params["output"]["stream_id"] == "audio/trimmed/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["trim"]["leading_frames"] == 16
    assert params["trim"]["trailing_frames"] == 16
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
    launch_source = (package_root() / "launch" / "fa_trim.launch.py").read_text(
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


def test_trim_does_not_hide_other_processing_or_io_responsibilities() -> None:
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
        "fade",
        "window_type",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_parameters_are_required_without_runtime_defaults_and_range_checked() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaTrimNode::loadParameters")[1].split(
        "void FaTrimNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<int>("trim.leading_frames");' in load_parameters
    assert 'this->declare_parameter<int>("trim.trailing_frames");' in load_parameters
    assert 'this->declare_parameter<bool>("qos.reliable");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredInt(*this, \"trim.leading_frames\")" in load_parameters
    assert "readRequiredInt(*this, \"trim.trailing_frames\")" in load_parameters
    assert "readRequiredBool(*this, \"qos.reliable\")" in load_parameters
    assert "readRequiredInt(*this, \"diagnostics.qos.depth\")" in load_parameters
    assert "readRequiredBool(*this, \"diagnostics.qos.reliable\")" in load_parameters
    assert "rclcpp::SystemDefaultsQoS()" not in source
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line

    assert "resolved_input_topic" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, config_.input_topic)" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, resolved_input_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, config_.output_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, resolved_output_topic)" in load_parameters
    assert "config_.input_stream_id == config_.output_stream_id" in load_parameters
    assert "config_.leading_frames < 0" in load_parameters
    assert "trim.leading_frames must be >= 0" in load_parameters
    assert "config_.trailing_frames < 0" in load_parameters
    assert "trim.trailing_frames must be >= 0" in load_parameters
    assert "config_.leading_frames == 0 && config_.trailing_frames == 0" in load_parameters
    assert "at least one of trim.leading_frames or trim.trailing_frames must be > 0" in load_parameters
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


def test_runtime_validates_audio_frame_contract_before_backend() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaTrimNode::validateFrame")[1].split(
        "bool FaTrimNode::applyTrim"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "contract_drops_.fetch_add(1);" in validate_frame


def test_sample_validation_and_payload_trim_are_backend_responsibilities() -> None:
    source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()
    handle_frame = source.split("void FaTrimNode::handleFrame")[1].split(
        "bool FaTrimNode::validateFrame"
    )[0]
    apply_trim = source.split("bool FaTrimNode::applyTrim")[1].split(
        "size_t FaTrimNode::bytesPerFrame"
    )[0]
    process = backend_source.split("ProcessResult InternalFrameTrimBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "validateSamples" not in source
    assert "trimFrame" not in source
    assert "if (!applyTrim(*msg, out))" in handle_frame
    assert "backend_->process(in.data, output_data)" in apply_trim
    assert "const char * status_message = backends::processStatusMessage(result.status);" in apply_trim
    assert "enum class ProcessStatus" in backend_header
    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "std::clamp" not in process
    assert "next_output.assign(" in process
    assert "output = std::move(next_output);" in process


def test_trim_preserves_declared_metadata_updates_stream_epoch_and_payload() -> None:
    source = read_node_source()
    apply_trim = source.split("bool FaTrimNode::applyTrim")[1].split(
        "size_t FaTrimNode::bytesPerFrame"
    )[0]

    assert "out = in;" in apply_trim
    assert "out.stream_id = config_.output_stream_id;" in apply_trim
    assert "out.epoch = in.epoch + 1U;" in apply_trim
    assert "out.data = std::move(output_data);" in apply_trim
    assert "out.source_id" not in apply_trim
    assert "out.header" not in apply_trim
    assert "out.encoding" not in apply_trim
    assert "out.sample_rate" not in apply_trim
    assert "out.channels" not in apply_trim
    assert "out.bit_depth" not in apply_trim
    assert "out.layout" not in apply_trim


def test_trim_drops_frames_that_would_be_empty() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalFrameTrimBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_trim = source.split("bool FaTrimNode::applyTrim")[1].split(
        "size_t FaTrimNode::bytesPerFrame"
    )[0]
    handle_frame = source.split("void FaTrimNode::handleFrame")[1].split(
        "bool FaTrimNode::validateFrame"
    )[0]

    assert "config_.leading_frames >= frame_count" in process
    assert "config_.trailing_frames >= (frame_count - config_.leading_frames)" in process
    assert "ProcessStatus::kTrimExhaustsInput" in process
    assert "trim_exhausted_drops_.fetch_add(1);" in apply_trim
    assert "last_output_frame_count_.store(0U);" in apply_trim
    assert "return false;" in apply_trim
    assert "if (!applyTrim(*msg, out))" in handle_frame
    assert "audio_pub_->publish(out);" in handle_frame
    assert handle_frame.index("if (!applyTrim(*msg, out))") < handle_frame.index(
        "audio_pub_->publish(out);"
    )


def test_epoch_increment_wrap_is_dropped_after_backend_acceptance() -> None:
    source = read_node_source()
    apply_trim = source.split("bool FaTrimNode::applyTrim")[1].split(
        "size_t FaTrimNode::bytesPerFrame"
    )[0]

    assert "in.epoch == std::numeric_limits<uint32_t>::max()" in apply_trim
    assert "epoch_overflow_drops_.fetch_add(1);" in apply_trim
    assert "Dropping frame because epoch increment would wrap uint32" in apply_trim
    assert "out.epoch = in.epoch + 1U;" in apply_trim
    assert apply_trim.index("backend_->process(in.data, output_data)") < apply_trim.index(
        "in.epoch == std::numeric_limits<uint32_t>::max()"
    )


def test_diagnostics_include_trim_policy_stream_identity_and_drop_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaTrimNode::publishDiagnostics")[1].split(
        "}  // namespace fa_trim"
    )[0]

    assert 'status.name = "fa_trim";' in diagnostics
    assert 'pushKeyValue(status, "input_stream_id", config_.input_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "output_stream_id", config_.output_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "leading_frames", std::to_string(config_.leading_frames));' in diagnostics
    assert 'pushKeyValue(status, "trailing_frames", std::to_string(config_.trailing_frames));' in diagnostics
    assert 'pushKeyValue(status, "last_input_frame_count",' in diagnostics
    assert 'pushKeyValue(status, "last_output_frame_count",' in diagnostics
    assert 'pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));' in diagnostics
    assert 'pushKeyValue(status, "contract_drops", std::to_string(contract_drops_.load()));' in diagnostics
    assert (
        'pushKeyValue(status, "invalid_sample_drops", '
        "std::to_string(invalid_sample_drops_.load()));"
    ) in diagnostics
    assert (
        'pushKeyValue(status, "trim_exhausted_drops", '
        "std::to_string(trim_exhausted_drops_.load()));"
    ) in diagnostics
    assert (
        'pushKeyValue(status, "epoch_overflow_drops", '
        "std::to_string(epoch_overflow_drops_.load()));"
    ) in diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "CMakeLists.txt",
        "package.xml",
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_frame_trim.md",
        "config/default.yaml",
        "config/profiles/.gitkeep",
        "launch/fa_trim.launch.py",
        "include/fa_trim/fa_trim_node.hpp",
        "include/fa_trim/backends/internal_frame_trim.hpp",
        "src/fa_trim_node.cpp",
        "src/backends/internal_frame_trim.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_frame_trim_backend.cpp",
        "test/unit/test_fa_trim_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()

    assert not (package_root() / "docs" / "backends" / "no_runtime_backend.md").exists()
    assert not (package_root() / "include" / "fa_trim" / "backends" / ".gitkeep").exists()
    assert not (package_root() / "src" / "backends" / ".gitkeep").exists()


def test_colcon_runs_pytest_and_backend_gtest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "find_package(ament_lint_auto REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_lint_auto_find_test_dependencies()" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

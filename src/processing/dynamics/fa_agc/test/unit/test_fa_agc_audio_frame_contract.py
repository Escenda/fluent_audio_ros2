from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_agc_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_agc" / "backends" / "internal_rms_agc.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_rms_agc.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_agc"]["ros__parameters"]

    assert params["input_topic"] == "audio/compressed/mic"
    assert params["output_topic"] == "audio/agc/mic"
    assert params["agc"]["target_rms"] == 0.1
    assert params["agc"]["min_gain"] == 0.25
    assert params["agc"]["max_gain"] == 4.0
    assert params["agc"]["attack_ms"] == 10.0
    assert params["agc"]["release_ms"] == 250.0
    assert 0.0 < params["agc"]["target_rms"] <= 1.0
    assert 0.0 < params["agc"]["min_gain"] <= 1.0
    assert params["agc"]["max_gain"] >= 1.0
    assert params["agc"]["min_gain"] <= params["agc"]["max_gain"]
    assert params["agc"]["attack_ms"] > 0.0
    assert params["agc"]["release_ms"] > 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_agc_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "std::clamp",
        "SND_PCM",
        "snd_pcm",
        "resample",
        "set_channels",
        "convert",
        "applyLimiter",
        "applyCompressor",
        "applyNormalize",
        "applyNoiseGate",
        "device_gain",
        "fa_in/",
        "fa_in::",
        'package="fa_in"',
        "low_pass",
        "high_pass",
        "denoise",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_config_validation_fails_closed() -> None:
    node_source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = node_source.split("void FaAgcNode::loadParameters")[1].split(
        "void FaAgcNode::configureBackend"
    )[0]

    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "config_.target_rms <= 0.0" in load_parameters
    assert "config_.target_rms > 1.0" in load_parameters
    assert "agc.target_rms must be finite and in (0.0, 1.0]" in load_parameters
    assert "config_.min_gain <= 0.0" in load_parameters
    assert "agc.min_gain must be finite and > 0.0" in load_parameters
    assert "config_.max_gain < config_.min_gain" in load_parameters
    assert "agc.max_gain must be finite and >= agc.min_gain" in load_parameters
    assert "config_.min_gain > 1.0 || config_.max_gain < 1.0" in load_parameters
    assert "agc.min_gain <= 1.0 <= agc.max_gain is required for initial gain" in load_parameters
    assert "config_.attack_ms <= 0.0" in load_parameters
    assert "config_.release_ms <= 0.0" in load_parameters
    assert "fa_agc requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_agc requires expected.bit_depth=32" in load_parameters
    assert "fa_agc requires expected.layout=interleaved" in load_parameters
    assert "config_.sample_rate <= 0" in backend_source
    assert "config_.channels <= 0" in backend_source
    assert "config_.target_rms <= 0.0" in backend_source
    assert "config_.max_gain < config_.min_gain" in backend_source


def test_runtime_frame_validation_drops_invalid_frames() -> None:
    source = read_node_source()
    handle_frame = source.split("void FaAgcNode::handleFrame")[1].split(
        "bool FaAgcNode::validateFrame"
    )[0]
    validate_frame = source.split("bool FaAgcNode::validateFrame")[1].split(
        "bool FaAgcNode::applyAgc"
    )[0]

    assert "if (!msg)" in handle_frame
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_agc_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_agc = source.split("bool FaAgcNode::applyAgc")[1].split(
        "void FaAgcNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_agc
    assert "out.stream_id = config_.output_topic;" in apply_agc
    assert "backend_->process(in.data, out.data)" in apply_agc
    assert ".vad" not in apply_agc
    assert ".asr" not in apply_agc


def test_frame_rms_target_gain_and_smoothing_are_backend_owned() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "class InternalRmsAgcBackend" in header
    assert "struct ProcessResult" in header
    assert "enum class GainDirection" in header
    assert "double InternalRmsAgcBackend::calculateFrameRms" in backend_source
    assert "square_sum += value * value;" in backend_source
    assert "return std::sqrt(mean_square);" in backend_source
    assert "double InternalRmsAgcBackend::boundedTargetGain" in backend_source
    assert "target_gain = config_.target_rms / frame_rms;" in backend_source
    assert "target_gain < config_.min_gain" in backend_source
    assert "target_gain > config_.max_gain" in backend_source
    assert "double InternalRmsAgcBackend::smoothingAlpha" in backend_source
    assert "1.0 - std::exp(-frame_seconds / time_constant_seconds)" in backend_source
    assert "target_gain < current_gain_" in backend_source
    assert "current_gain_ + (alpha * (target_gain - current_gain_))" in backend_source
    assert "calculateFrameRms" not in node_source
    assert "boundedTargetGain" not in node_source
    assert "smoothingAlpha" not in node_source


def test_agc_drops_non_finite_or_out_of_range_samples_instead_of_clamping() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalRmsAgcBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(output_sample)" in process
    assert "output_sample < kNormalizedMin || output_sample > kNormalizedMax" in process
    assert "ProcessStatus::kNonFiniteInput" in process
    assert "ProcessStatus::kOutOfRangeInput" in process
    assert "ProcessStatus::kOutOfRangeOutput" in process
    assert "std::clamp" not in backend_source


def test_output_range_drop_does_not_commit_candidate_gain_or_output() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalRmsAgcBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    validation_before_commit = process.split("current_gain_ = candidate_gain;")[0]
    assert "std::vector<uint8_t> next_output(input.size());" in validation_before_commit
    assert "ProcessStatus::kOutOfRangeOutput" in validation_before_commit
    assert "output = std::move(next_output);" not in validation_before_commit
    assert "current_gain_ = candidate_gain;" in process
    assert "last_frame_rms_ = frame_rms;" in process
    assert "last_target_gain_ = target_gain;" in process
    assert "output = std::move(next_output);" in process


def test_agc_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "enum class ProcessStatus" in header
    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteGain" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "ProcessStatus::kMisalignedInput" in backend_source
    assert "ProcessStatus::kNonFiniteInput" in backend_source
    assert "ProcessStatus::kOutOfRangeOutput" in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source
    assert "GainDirection::kReduction" in node_source
    assert "GainDirection::kIncrease" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_diagnostics_include_parameters_state_and_counters() -> None:
    source = read_node_source()
    publish_diagnostics = source.split("void FaAgcNode::publishDiagnostics")[1].split(
        "}  // namespace fa_agc"
    )[0]

    assert 'status.name = "fa_agc";' in publish_diagnostics
    assert '"target_rms"' in publish_diagnostics
    assert '"min_gain"' in publish_diagnostics
    assert '"max_gain"' in publish_diagnostics
    assert '"attack_ms"' in publish_diagnostics
    assert '"release_ms"' in publish_diagnostics
    assert '"current_gain"' in publish_diagnostics
    assert '"last_frame_rms"' in publish_diagnostics
    assert '"last_target_gain"' in publish_diagnostics
    assert '"frames_in"' in publish_diagnostics
    assert '"frames_out"' in publish_diagnostics
    assert '"frames_dropped"' in publish_diagnostics
    assert '"gain_reductions"' in publish_diagnostics
    assert '"gain_increases"' in publish_diagnostics
    assert '"output_topic"' in publish_diagnostics
    assert "backend_->currentGain()" in publish_diagnostics
    assert "backend_->lastFrameRms()" in publish_diagnostics
    assert "backend_->lastTargetGain()" in publish_diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_rms_agc.md",
        "config/default.yaml",
        "launch/fa_agc.launch.py",
        "include/fa_agc/fa_agc_node.hpp",
        "include/fa_agc/backends/internal_rms_agc.hpp",
        "src/fa_agc_node.cpp",
        "src/backends/internal_rms_agc.cpp",
        "test/cpp/test_internal_rms_agc_backend.cpp",
        "test/unit/test_fa_agc_audio_frame_contract.py",
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

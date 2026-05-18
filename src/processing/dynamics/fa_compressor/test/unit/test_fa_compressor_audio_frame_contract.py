from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_compressor_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_compressor"
        / "backends"
        / "internal_static_curve.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_static_curve.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_compressor"]["ros__parameters"]

    assert params["input_topic"] == "audio/normalized/mic"
    assert params["output_topic"] == "audio/compressed/mic"
    assert params["compressor"]["threshold_linear"] == 0.5
    assert params["compressor"]["ratio"] == 4.0
    assert params["compressor"]["makeup_gain_linear"] == 1.0
    assert 0.0 < params["compressor"]["threshold_linear"] < 1.0
    assert params["compressor"]["ratio"] > 1.0
    assert 0.0 < params["compressor"]["makeup_gain_linear"] <= 4.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_compressor_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "std::clamp",
        "clip",
        "SND_PCM",
        "snd_pcm",
        "resample",
        "set_channels",
        "applyLimiter",
        "applyNoiseGate",
        "applyNormalize",
        "low_pass",
        "high_pass",
        "denoise",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_config_validation_fails_closed() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = source.split("void FaCompressorNode::loadParameters")[1].split(
        "void FaCompressorNode::configureBackend"
    )[0]

    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "config_.threshold_linear <= 0.0" in load_parameters
    assert "config_.threshold_linear >= 1.0" in load_parameters
    assert "compressor.threshold_linear must be finite and in (0.0, 1.0)" in load_parameters
    assert "config_.ratio <= 1.0" in load_parameters
    assert "compressor.ratio must be finite and > 1.0" in load_parameters
    assert "config_.makeup_gain_linear <= 0.0" in load_parameters
    assert "config_.makeup_gain_linear > 4.0" in load_parameters
    assert "compressor.makeup_gain_linear must be finite and in (0.0, 4.0]" in load_parameters
    assert "fa_compressor requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_compressor requires expected.bit_depth=32" in load_parameters
    assert "fa_compressor requires expected.layout=interleaved" in load_parameters
    assert "config_.threshold_linear <= 0.0" in backend_source
    assert "config_.ratio <= 1.0" in backend_source
    assert "config_.makeup_gain_linear > 4.0" in backend_source


def test_runtime_frame_validation_drops_invalid_frames() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaCompressorNode::validateFrame")[1].split(
        "bool FaCompressorNode::applyCompressor"
    )[0]
    handle_frame = source.split("void FaCompressorNode::handleFrame")[1].split(
        "bool FaCompressorNode::validateFrame"
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


def test_compressor_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_compressor = source.split("bool FaCompressorNode::applyCompressor")[1].split(
        "void FaCompressorNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_compressor
    assert "out.stream_id = config_.output_topic;" in apply_compressor
    assert ".rms" not in apply_compressor
    assert ".peak" not in apply_compressor
    assert ".vad" not in apply_compressor


def test_static_compression_curve_is_backend_owned() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalStaticCurveBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_compressor = node_source.split("bool FaCompressorNode::applyCompressor")[1].split(
        "void FaCompressorNode::publishDiagnostics"
    )[0]

    assert "class InternalStaticCurveBackend" in header
    assert "struct ProcessResult" in header
    assert "uint64_t samples_compressed" in header
    assert "const double amplitude = std::abs(static_cast<double>(sample));" in process
    assert "if (amplitude > config_.threshold_linear)" in process
    assert "config_.threshold_linear + ((amplitude - config_.threshold_linear) / config_.ratio)" in process
    assert "std::signbit(sample) ? -compressed_abs : compressed_abs" in process
    assert "const double output_sample = signed_sample * config_.makeup_gain_linear;" in process
    assert "backend_->process(in.data, out.data)" in apply_compressor
    assert "samples_compressed_.fetch_add(result.samples_compressed);" in apply_compressor


def test_compressor_drops_non_finite_or_out_of_range_samples_instead_of_clamping() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalStaticCurveBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(output_sample)" in process
    assert "output_sample < kNormalizedMin || output_sample > kNormalizedMax" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeInput, 0};" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeOutput, 0};" in process
    assert "std::clamp" not in process
    assert "backends::processStatusMessage(result.status)" in node_source


def test_compressor_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "enum class ProcessStatus" in header
    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "return ProcessResult{ProcessStatus::kMisalignedInput, 0};" in backend_source
    assert "return ProcessResult{ProcessStatus::kNonFiniteInput, 0};" in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_diagnostics_include_parameters_and_counters() -> None:
    source = read_node_source()
    publish_diagnostics = source.split("void FaCompressorNode::publishDiagnostics")[1].split(
        "}  // namespace fa_compressor"
    )[0]

    assert 'status.name = "fa_compressor";' in publish_diagnostics
    assert '"threshold_linear"' in publish_diagnostics
    assert '"ratio"' in publish_diagnostics
    assert '"makeup_gain_linear"' in publish_diagnostics
    assert '"frames_in"' in publish_diagnostics
    assert '"frames_out"' in publish_diagnostics
    assert '"frames_dropped"' in publish_diagnostics
    assert '"samples_compressed"' in publish_diagnostics
    assert '"output_topic"' in publish_diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_static_curve.md",
        "config/default.yaml",
        "launch/fa_compressor.launch.py",
        "include/fa_compressor/fa_compressor_node.hpp",
        "include/fa_compressor/backends/internal_static_curve.hpp",
        "src/fa_compressor_node.cpp",
        "src/backends/internal_static_curve.cpp",
        "test/cpp/test_internal_static_curve_backend.cpp",
        "test/unit/test_fa_compressor_audio_frame_contract.py",
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

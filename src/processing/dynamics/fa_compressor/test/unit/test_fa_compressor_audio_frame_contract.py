from pathlib import Path

import yaml


def _package_root() -> Path:
    return Path(__file__).parents[2]


def _source_text() -> str:
    return (_package_root() / "src" / "fa_compressor_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((_package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
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
    source = _source_text()

    forbidden = (
        "std::clamp",
        "clip",
        "SND_PCM",
        "snd_pcm",
        "resample",
        "set_channels",
        "convert",
        "applyLimiter",
        "applyNoiseGate",
        "applyNormalize",
        "low_pass",
        "high_pass",
        "denoise",
    )
    for token in forbidden:
        assert token not in source


def test_startup_config_validation_fails_closed() -> None:
    source = _source_text()
    load_parameters = source.split("void FaCompressorNode::loadParameters")[1].split(
        "void FaCompressorNode::setupInterfaces"
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


def test_runtime_frame_validation_drops_invalid_frames() -> None:
    source = _source_text()
    validate_frame = source.split("bool FaCompressorNode::validateFrame")[1].split(
        "bool FaCompressorNode::applyCompressor"
    )[0]

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
    source = _source_text()
    apply_compressor = source.split("bool FaCompressorNode::applyCompressor")[1].split(
        "void FaCompressorNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_compressor
    assert "out.stream_id = config_.output_topic;" in apply_compressor
    assert ".rms" not in apply_compressor
    assert ".peak" not in apply_compressor
    assert ".vad" not in apply_compressor


def test_static_compression_curve_is_explicit_and_per_sample() -> None:
    source = _source_text()
    apply_compressor = source.split("bool FaCompressorNode::applyCompressor")[1].split(
        "void FaCompressorNode::publishDiagnostics"
    )[0]

    assert "const double amplitude = std::abs(static_cast<double>(sample));" in apply_compressor
    assert "if (amplitude > threshold)" in apply_compressor
    assert "compressed_abs = threshold + ((amplitude - threshold) / ratio);" in apply_compressor
    assert "std::signbit(sample) ? -compressed_abs : compressed_abs" in apply_compressor
    assert "const double output = signed_sample * makeup_gain;" in apply_compressor
    assert "samples_compressed_.fetch_add(compressed_in_frame);" in apply_compressor


def test_compressor_drops_non_finite_or_out_of_range_samples_instead_of_clamping() -> None:
    source = _source_text()
    apply_compressor = source.split("bool FaCompressorNode::applyCompressor")[1].split(
        "void FaCompressorNode::publishDiagnostics"
    )[0]

    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in apply_compressor
    assert "output < kMinNormalizedSample" in apply_compressor
    assert "output > kMaxNormalizedSample" in apply_compressor
    assert "Dropping frame because compressor output is outside normalized FLOAT32LE range" in apply_compressor
    assert "return false;" in apply_compressor
    assert "std::clamp" not in apply_compressor
    assert "kMaxNormalizedSample)" in apply_compressor


def test_diagnostics_include_parameters_and_counters() -> None:
    source = _source_text()
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
        "src/fa_compressor_node.cpp",
        "test/unit/test_fa_compressor_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (_package_root() / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    cmake_text = (_package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (_package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

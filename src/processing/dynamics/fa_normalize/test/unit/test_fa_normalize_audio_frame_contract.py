from pathlib import Path

import yaml


def test_default_config_requires_float32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_normalize"]["ros__parameters"]

    assert params["input_topic"] == "audio/noise_gated/mic"
    assert params["output_topic"] == "audio/normalized/mic"
    assert params["normalize"]["target_peak_linear"] == 0.9
    assert params["normalize"]["silence_threshold_linear"] == 0.0001
    assert 0.0 < params["normalize"]["target_peak_linear"] <= 1.0
    assert 0.0 <= params["normalize"]["silence_threshold_linear"]
    assert params["normalize"]["silence_threshold_linear"] < params["normalize"]["target_peak_linear"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_normalize_does_not_hide_other_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "std::clamp",
        "compress",
        "limiter",
        "limit",
        "gate.",
        "filter.",
        "denoise",
        "lufs",
        "loudness",
        "legacy",
        "compat",
    )
    for token in forbidden:
        assert token not in source


def test_startup_rejects_invalid_config_without_fallback() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaNormalizeNode::loadParameters")[1].split(
        "void FaNormalizeNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter<double>("normalize.target_peak_linear", config_.target_peak_linear);' in load_parameters
    assert (
        'this->declare_parameter<double>("normalize.silence_threshold_linear", '
        "config_.silence_threshold_linear);"
    ) in load_parameters
    assert "!isFinite(config_.target_peak_linear)" in load_parameters
    assert "config_.target_peak_linear <= 0.0" in load_parameters
    assert "config_.target_peak_linear > 1.0" in load_parameters
    assert "!isFinite(config_.silence_threshold_linear)" in load_parameters
    assert "config_.silence_threshold_linear < 0.0" in load_parameters
    assert "config_.silence_threshold_linear >= config_.target_peak_linear" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "requires expected.encoding=FLOAT32LE" in load_parameters
    assert "requires expected.bit_depth=32" in load_parameters
    assert "requires expected.layout=interleaved" in load_parameters


def test_normalize_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaNormalizeNode::validateFrame")[1].split(
        "bool FaNormalizeNode::applyNormalize"
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


def test_normalize_preserves_frame_identity_and_updates_stream_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")
    apply_normalize = source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_normalize
    assert "out.stream_id = config_.output_topic;" in apply_normalize
    assert "out.data.resize(in.data.size());" in apply_normalize
    assert ".rms" not in apply_normalize
    assert ".vad" not in apply_normalize
    assert ".epoch" not in apply_normalize


def test_peak_normalize_algorithm_uses_frame_peak_and_target_gain() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")
    apply_normalize = source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "float peak = 0.0F;" in apply_normalize
    assert "peak = std::max(peak, std::abs(sample));" in apply_normalize
    assert "const double gain = config_.target_peak_linear / static_cast<double>(peak);" in apply_normalize
    assert "const double normalized = static_cast<double>(samples[i]) * gain;" in apply_normalize
    assert "frames_normalized_.fetch_add(1);" in apply_normalize
    assert "last_gain_.store(gain);" in apply_normalize
    assert "std::clamp" not in apply_normalize


def test_silence_pass_through_changes_only_stream_identity_and_does_not_amplify() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")
    apply_normalize = source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "peak < static_cast<float>(config_.silence_threshold_linear)" in apply_normalize
    assert "out.data = in.data;" in apply_normalize
    assert "frames_silence_passthrough_.fetch_add(1);" in apply_normalize
    assert "last_gain_.store(1.0);" in apply_normalize


def test_normalize_drops_invalid_samples_and_outputs_instead_of_clamping() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")
    apply_normalize = source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "!std::isfinite(sample)" in apply_normalize
    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in apply_normalize
    assert "!isFinite(normalized)" in apply_normalize
    assert "normalized < kMinNormalizedSample" in apply_normalize
    assert "normalized > kMaxNormalizedSample" in apply_normalize
    assert "!std::isfinite(out_sample)" in apply_normalize
    assert "return false;" in apply_normalize
    assert "std::clamp" not in apply_normalize


def test_diagnostics_publish_config_last_gain_and_counters() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")
    diagnostics = source.split("void FaNormalizeNode::publishDiagnostics")[1].split(
        "}  // namespace fa_normalize"
    )[0]

    assert 'status.name = "fa_normalize";' in diagnostics
    assert '"target_peak_linear"' in diagnostics
    assert '"silence_threshold_linear"' in diagnostics
    assert '"last_gain"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"frames_silence_passthrough"' in diagnostics
    assert '"frames_normalized"' in diagnostics
    assert '"output_topic"' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_peak_normalize.md",
        "config/default.yaml",
        "launch/fa_normalize.launch.py",
        "include/fa_normalize/fa_normalize_node.hpp",
        "src/fa_normalize_node.cpp",
        "test/unit/test_fa_normalize_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

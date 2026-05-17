from pathlib import Path

import yaml


def test_default_config_requires_float32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_noise_gate"]["ros__parameters"]

    assert params["input_topic"] == "audio/dc_offset_removed/mic"
    assert params["output_topic"] == "audio/noise_gated/mic"
    assert params["gate"]["threshold_linear"] == 0.02
    assert params["gate"]["closed_gain_linear"] == 0.0
    assert 0.0 <= params["gate"]["threshold_linear"] <= 1.0
    assert 0.0 <= params["gate"]["closed_gain_linear"] <= 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"


def test_noise_gate_does_not_hide_other_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_noise_gate_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "normalize(",
        "std::clamp",
        "threshold.linear",
        "filter.",
        "denoise",
        "compress",
        "limiter",
        "limit",
    )
    for token in forbidden:
        assert token not in source


def test_startup_rejects_invalid_config_without_fallback() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_noise_gate_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaNoiseGateNode::loadParameters")[1].split(
        "void FaNoiseGateNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter<double>("gate.threshold_linear", config_.threshold_linear);' in load_parameters
    assert 'this->declare_parameter<double>("gate.closed_gain_linear", config_.closed_gain_linear);' in load_parameters
    assert "!isFinite(config_.threshold_linear)" in load_parameters
    assert "config_.threshold_linear < 0.0" in load_parameters
    assert "config_.threshold_linear > 1.0" in load_parameters
    assert "!isFinite(config_.closed_gain_linear)" in load_parameters
    assert "config_.closed_gain_linear < 0.0" in load_parameters
    assert "config_.closed_gain_linear > 1.0" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "requires expected.encoding=FLOAT32LE" in load_parameters
    assert "requires expected.bit_depth=32" in load_parameters
    assert "requires expected.layout=interleaved" in load_parameters


def test_noise_gate_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_noise_gate_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaNoiseGateNode::validateFrame")[1].split(
        "bool FaNoiseGateNode::applyNoiseGate"
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


def test_noise_gate_preserves_frame_identity_and_updates_stream_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_noise_gate_node.cpp").read_text(encoding="utf-8")
    apply_gate = source.split("bool FaNoiseGateNode::applyNoiseGate")[1].split(
        "void FaNoiseGateNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_gate
    assert "out.stream_id = config_.output_topic;" in apply_gate
    assert "out.data.resize(in.data.size());" in apply_gate
    assert ".rms" not in apply_gate
    assert ".peak" not in apply_gate
    assert ".vad" not in apply_gate


def test_noise_gate_algorithm_uses_threshold_and_closed_gain_only() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_noise_gate_node.cpp").read_text(encoding="utf-8")
    apply_gate = source.split("bool FaNoiseGateNode::applyNoiseGate")[1].split(
        "void FaNoiseGateNode::publishDiagnostics"
    )[0]

    assert "const float threshold = static_cast<float>(config_.threshold_linear);" in apply_gate
    assert "const double closed_gain = config_.closed_gain_linear;" in apply_gate
    assert "if (std::abs(sample) < threshold)" in apply_gate
    assert "output = static_cast<double>(sample) * closed_gain;" in apply_gate
    assert "++gated_in_frame;" in apply_gate
    assert "samples_gated_.fetch_add(gated_in_frame);" in apply_gate
    assert "else" not in apply_gate
    assert "std::clamp" not in apply_gate


def test_noise_gate_drops_invalid_samples_instead_of_clamping_or_normalizing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_noise_gate_node.cpp").read_text(encoding="utf-8")
    apply_gate = source.split("bool FaNoiseGateNode::applyNoiseGate")[1].split(
        "void FaNoiseGateNode::publishDiagnostics"
    )[0]

    assert "!std::isfinite(sample)" in apply_gate
    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in apply_gate
    assert "!isFinite(output) || output < kMinNormalizedSample || output > kMaxNormalizedSample" in apply_gate
    assert "!std::isfinite(out_sample)" in apply_gate
    assert "return false;" in apply_gate
    assert "std::clamp" not in apply_gate
    assert "normalize(" not in apply_gate


def test_diagnostics_publish_config_and_counters() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_noise_gate_node.cpp").read_text(encoding="utf-8")
    diagnostics = source.split("void FaNoiseGateNode::publishDiagnostics")[1].split(
        "}  // namespace fa_noise_gate"
    )[0]

    assert 'status.name = "fa_noise_gate";' in diagnostics
    assert '"gate_threshold_linear"' in diagnostics
    assert '"gate_closed_gain_linear"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"samples_gated"' in diagnostics
    assert '"output_topic"' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_threshold_gate.md",
        "config/default.yaml",
        "launch/fa_noise_gate.launch.py",
        "include/fa_noise_gate/fa_noise_gate_node.hpp",
        "src/fa_noise_gate_node.cpp",
        "test/unit/test_fa_noise_gate_audio_frame_contract.py",
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

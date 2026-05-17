from pathlib import Path

import yaml


def test_default_config_requires_float32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_expander"]["ros__parameters"]

    assert params["input_topic"] == "audio/noise_gated/mic"
    assert params["output_topic"] == "audio/expanded/mic"
    assert params["expander"]["threshold_linear"] == 0.05
    assert params["expander"]["ratio"] == 2.0
    assert 0.0 < params["expander"]["threshold_linear"] < 1.0
    assert params["expander"]["ratio"] > 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_expander_does_not_hide_other_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_expander_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "std::clamp",
        "closed_gain",
        "target_peak",
        "silence_threshold",
        "lufs",
        "loudness",
        "legacy",
        "compat",
    )
    for token in forbidden:
        assert token not in source


def test_startup_rejects_invalid_config_without_fallback() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_expander_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaExpanderNode::loadParameters")[1].split(
        "void FaExpanderNode::setupInterfaces"
    )[0]

    assert (
        'this->declare_parameter<double>("expander.threshold_linear", '
        "config_.threshold_linear);"
    ) in load_parameters
    assert 'this->declare_parameter<double>("expander.ratio", config_.ratio);' in load_parameters
    assert "!isFinite(config_.threshold_linear)" in load_parameters
    assert "config_.threshold_linear <= 0.0" in load_parameters
    assert "config_.threshold_linear >= 1.0" in load_parameters
    assert "!isFinite(config_.ratio)" in load_parameters
    assert "config_.ratio <= 1.0" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "requires expected.encoding=FLOAT32LE" in load_parameters
    assert "requires expected.bit_depth=32" in load_parameters
    assert "requires expected.layout=interleaved" in load_parameters


def test_expander_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_expander_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaExpanderNode::validateFrame")[1].split(
        "bool FaExpanderNode::applyExpansion"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.channels == 0U" in validate_frame
    assert "static_cast<size_t>(msg.channels) * sizeof(float)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_expander_preserves_frame_identity_and_updates_stream_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_expander_node.cpp").read_text(encoding="utf-8")
    apply_expansion = source.split("bool FaExpanderNode::applyExpansion")[1].split(
        "void FaExpanderNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_expansion
    assert "out.stream_id = config_.output_topic;" in apply_expansion
    assert "out.data.resize(in.data.size());" in apply_expansion
    assert ".rms" not in apply_expansion
    assert ".peak" not in apply_expansion
    assert ".vad" not in apply_expansion
    assert ".epoch" not in apply_expansion


def test_static_expansion_curve_uses_threshold_ratio_and_sample_sign() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_expander_node.cpp").read_text(encoding="utf-8")
    apply_expansion = source.split("bool FaExpanderNode::applyExpansion")[1].split(
        "void FaExpanderNode::publishDiagnostics"
    )[0]

    assert "const double threshold = config_.threshold_linear;" in apply_expansion
    assert "const double ratio = config_.ratio;" in apply_expansion
    assert "const double magnitude = std::abs(static_cast<double>(sample));" in apply_expansion
    assert "if (magnitude < threshold)" in apply_expansion
    assert "threshold * std::pow(magnitude / threshold, ratio)" in apply_expansion
    assert "std::copysign(expanded_abs, static_cast<double>(sample))" in apply_expansion
    assert "++expanded_in_frame;" in apply_expansion
    assert "samples_expanded_.fetch_add(expanded_in_frame);" in apply_expansion


def test_expander_drops_invalid_samples_and_outputs_instead_of_clamping_or_zeroing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_expander_node.cpp").read_text(encoding="utf-8")
    apply_expansion = source.split("bool FaExpanderNode::applyExpansion")[1].split(
        "void FaExpanderNode::publishDiagnostics"
    )[0]

    assert "!std::isfinite(sample)" in apply_expansion
    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in apply_expansion
    assert "!isFinite(expanded)" in apply_expansion
    assert "expanded < kMinNormalizedSample" in apply_expansion
    assert "expanded > kMaxNormalizedSample" in apply_expansion
    assert "!std::isfinite(out_sample)" in apply_expansion
    assert "return false;" in apply_expansion
    assert "std::clamp" not in apply_expansion
    assert "out_sample = 0.0" not in apply_expansion
    assert "expanded = 0.0" not in apply_expansion


def test_diagnostics_publish_config_and_counters() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_expander_node.cpp").read_text(encoding="utf-8")
    diagnostics = source.split("void FaExpanderNode::publishDiagnostics")[1].split(
        "}  // namespace fa_expander"
    )[0]

    assert 'status.name = "fa_expander";' in diagnostics
    assert '"expander_threshold_linear"' in diagnostics
    assert '"expander_ratio"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"samples_expanded"' in diagnostics
    assert '"output_topic"' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_static_expander.md",
        "config/default.yaml",
        "launch/fa_expander.launch.py",
        "include/fa_expander/fa_expander_node.hpp",
        "src/fa_expander_node.cpp",
        "test/unit/test_fa_expander_audio_frame_contract.py",
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

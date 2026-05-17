from pathlib import Path

import yaml


def test_default_config_requires_float32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_limiter"]["ros__parameters"]

    assert params["input_topic"] == "audio/gain/mic"
    assert params["output_topic"] == "audio/limit/mic"
    assert params["threshold"]["linear"] == 1.0
    assert 0.0 < params["threshold"]["linear"] <= 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"


def test_limiter_does_not_hide_other_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_limiter_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "gain.linear",
        "gain_",
    )
    for token in forbidden:
        assert token not in source


def test_limiter_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_limiter_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaLimiterNode::validateFrame")[1].split(
        "bool FaLimiterNode::applyLimiter"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame


def test_limiter_preserves_source_identity_and_updates_stream_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_limiter_node.cpp").read_text(encoding="utf-8")
    apply_limiter = source.split("bool FaLimiterNode::applyLimiter")[1].split(
        "void FaLimiterNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_limiter
    assert "out.stream_id = config_.output_topic;" in apply_limiter
    assert ".rms" not in apply_limiter
    assert ".peak" not in apply_limiter
    assert ".vad" not in apply_limiter


def test_limiter_clamps_samples_to_explicit_threshold() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_limiter_node.cpp").read_text(encoding="utf-8")
    apply_limiter = source.split("bool FaLimiterNode::applyLimiter")[1].split(
        "void FaLimiterNode::publishDiagnostics"
    )[0]

    assert "const float threshold = static_cast<float>(config_.threshold_linear);" in apply_limiter
    assert "if (sample > threshold)" in apply_limiter
    assert "out_sample = threshold;" in apply_limiter
    assert "else if (sample < -threshold)" in apply_limiter
    assert "out_sample = -threshold;" in apply_limiter
    assert "samples_limited_.fetch_add(limited_in_frame);" in apply_limiter
    assert "std::clamp" not in apply_limiter


def test_threshold_parameter_is_required_and_range_checked() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_limiter_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaLimiterNode::loadParameters")[1].split(
        "void FaLimiterNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter<double>("threshold.linear", config_.threshold_linear);' in load_parameters
    assert "config_.threshold_linear <= 0.0" in load_parameters
    assert "config_.threshold_linear > 1.0" in load_parameters
    assert "threshold.linear must be finite and in (0.0, 1.0]" in load_parameters


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_limiter.md",
        "config/default.yaml",
        "launch/fa_limiter.launch.py",
        "include/fa_limiter/fa_limiter_node.hpp",
        "src/fa_limiter_node.cpp",
        "test/unit",
        "test/integration",
        "test/launch",
        "test/fixtures",
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

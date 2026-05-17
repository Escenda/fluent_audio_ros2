from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_window_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_window"]["ros__parameters"]

    assert params["input_topic"] == "audio/buffered/mic"
    assert params["output_topic"] == "audio/windowed/mic"
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


def test_window_does_not_hide_other_processing_or_io_responsibilities() -> None:
    source = read_source()

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
    for token in forbidden:
        assert token not in source


def test_window_parameters_are_required_and_range_checked() -> None:
    source = read_source()
    load_parameters = source.split("void FaWindowNode::loadParameters")[1].split(
        "void FaWindowNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter("window.type", config_.window_type);' in load_parameters
    assert 'this->declare_parameter<int>("window.expected_frames", config_.expected_frames);' in load_parameters
    assert (
        'this->declare_parameter<bool>("window.strict_frame_count", '
        "config_.strict_frame_count);"
    ) in load_parameters
    assert "config_.window_type != kWindowHann && config_.window_type != kWindowHamming" in load_parameters
    assert "window.type must be one of hann, hamming" in load_parameters
    assert "config_.expected_frames <= 1" in load_parameters
    assert "window.expected_frames must be > 1" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters


def test_window_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaWindowNode::validateFrame")[1].split(
        "bool FaWindowNode::applyWindow"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_window_strict_and_dynamic_frame_count_handling_is_explicit() -> None:
    source = read_source()
    validate_frame = source.split("bool FaWindowNode::validateFrame")[1].split(
        "bool FaWindowNode::applyWindow"
    )[0]
    apply_window = source.split("bool FaWindowNode::applyWindow")[1].split(
        "std::vector<double> FaWindowNode::computeCoefficients"
    )[0]

    assert "const size_t frame_count = msg.data.size() / bytesPerFrame();" in validate_frame
    assert "config_.strict_frame_count &&" in validate_frame
    assert "frame_count != static_cast<size_t>(config_.expected_frames)" in validate_frame
    assert "!config_.strict_frame_count && frame_count <= 1U" in validate_frame
    assert "computeCoefficients(frame_count)" in apply_window


def test_window_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_source()
    apply_window = source.split("bool FaWindowNode::applyWindow")[1].split(
        "std::vector<double> FaWindowNode::computeCoefficients"
    )[0]

    assert "out = in;" in apply_window
    assert "out.stream_id = config_.output_topic;" in apply_window
    assert ".rms" not in apply_window
    assert ".peak" not in apply_window
    assert ".vad" not in apply_window


def test_window_uses_hann_and_hamming_coefficient_formulas() -> None:
    source = read_source()
    coefficient_at = source.split("double FaWindowNode::coefficientAt")[1].split(
        "size_t FaWindowNode::bytesPerFrame"
    )[0]

    assert "(2.0 * kPi * static_cast<double>(frame_index))" in coefficient_at
    assert "static_cast<double>(frame_count - 1U)" in coefficient_at
    assert "config_.window_type == kWindowHann" in coefficient_at
    assert "return 0.5 * (1.0 - std::cos(phase));" in coefficient_at
    assert "return 0.54 - (0.46 * std::cos(phase));" in coefficient_at


def test_window_drops_invalid_samples_instead_of_clamping_or_normalizing() -> None:
    source = read_source()
    apply_window = source.split("bool FaWindowNode::applyWindow")[1].split(
        "std::vector<double> FaWindowNode::computeCoefficients"
    )[0]

    assert "!std::isfinite(sample)" in apply_window
    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in apply_window
    assert "const double windowed = static_cast<double>(sample) * coefficients[frame_index];" in apply_window
    assert "!std::isfinite(windowed)" in apply_window
    assert "windowed < kMinNormalizedSample" in apply_window
    assert "windowed > kMaxNormalizedSample" in apply_window
    assert "!std::isfinite(out_sample)" in apply_window
    assert "return false;" in apply_window
    assert "std::clamp" not in apply_window


def test_diagnostics_include_type_frame_policy_last_frame_count_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaWindowNode::publishDiagnostics")[1].split(
        "}  // namespace fa_window"
    )[0]

    assert 'status.name = "fa_window";' in diagnostics
    assert 'pushKeyValue(status, "window_type", config_.window_type);' in diagnostics
    assert 'pushKeyValue(status, "expected_frames", std::to_string(config_.expected_frames));' in diagnostics
    assert 'pushKeyValue(status, "strict_frame_count", config_.strict_frame_count ? "true" : "false");' in diagnostics
    assert 'pushKeyValue(status, "last_frame_count", std::to_string(last_frame_count_.load()));' in diagnostics
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
        "src/fa_window_node.cpp",
        "test/unit/test_fa_window_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

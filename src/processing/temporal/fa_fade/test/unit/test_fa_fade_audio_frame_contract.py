from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_fade_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_fade"]["ros__parameters"]

    assert params["input_topic"] == "audio/buffered/mic"
    assert params["output_topic"] == "audio/faded/mic"
    assert params["fade"]["mode"] == "fade_in"
    assert params["fade"]["duration_frames"] == 16000
    assert params["fade"]["initial_position_frames"] == 0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_fade_does_not_hide_other_processing_or_io_responsibilities() -> None:
    source = read_source()

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
    )
    for token in forbidden:
        assert token not in source


def test_fade_parameters_are_required_and_range_checked() -> None:
    source = read_source()
    load_parameters = source.split("void FaFadeNode::loadParameters")[1].split(
        "void FaFadeNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter("fade.mode", config_.mode);' in load_parameters
    assert 'this->declare_parameter<int>("fade.duration_frames", config_.duration_frames);' in load_parameters
    assert (
        'this->declare_parameter<int>("fade.initial_position_frames", '
        "config_.initial_position_frames);"
    ) in load_parameters
    assert "config_.mode != kFadeIn && config_.mode != kFadeOut" in load_parameters
    assert "fade.mode must be one of fade_in, fade_out" in load_parameters
    assert "config_.duration_frames <= 0" in load_parameters
    assert "fade.duration_frames must be > 0" in load_parameters
    assert "config_.initial_position_frames < 0" in load_parameters
    assert "fade.initial_position_frames must be >= 0" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters


def test_fade_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaFadeNode::validateFrame")[1].split(
        "bool FaFadeNode::applyFade"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_fade_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_source()
    apply_fade = source.split("bool FaFadeNode::applyFade")[1].split(
        "double FaFadeNode::gainAtPosition"
    )[0]

    assert "out = in;" in apply_fade
    assert "out.stream_id = config_.output_topic;" in apply_fade
    assert ".rms" not in apply_fade
    assert ".peak" not in apply_fade
    assert ".vad" not in apply_fade


def test_fade_uses_linear_position_counter_across_accepted_frames() -> None:
    header = (package_root() / "include" / "fa_fade" / "fa_fade_node.hpp").read_text(
        encoding="utf-8"
    )
    source = read_source()
    constructor = source.split("FaFadeNode::FaFadeNode()")[1].split(
        "void FaFadeNode::loadParameters"
    )[0]
    apply_fade = source.split("bool FaFadeNode::applyFade")[1].split(
        "double FaFadeNode::gainAtPosition"
    )[0]
    gain_at_position = source.split("double FaFadeNode::gainAtPosition")[1].split(
        "size_t FaFadeNode::bytesPerFrame"
    )[0]

    assert "uint64_t position_frames_{0};" in header
    assert "position_frames_ = static_cast<uint64_t>(config_.initial_position_frames);" in constructor
    assert "const size_t frame_count = in.data.size() / bytesPerFrame();" in apply_fade
    assert (
        "position_frames_ + static_cast<uint64_t>(i / "
        "static_cast<size_t>(config_.expected_channels))"
    ) in apply_fade
    assert "position_frames_ += static_cast<uint64_t>(frame_count);" in apply_fade
    assert "static_cast<double>(position_frames) / static_cast<double>(config_.duration_frames)" in gain_at_position
    assert "return std::min(1.0, progress);" in gain_at_position
    assert "return std::max(0.0, 1.0 - progress);" in gain_at_position


def test_fade_drops_invalid_samples_instead_of_clamping_or_normalizing() -> None:
    source = read_source()
    apply_fade = source.split("bool FaFadeNode::applyFade")[1].split(
        "double FaFadeNode::gainAtPosition"
    )[0]

    assert "!std::isfinite(sample)" in apply_fade
    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in apply_fade
    assert "const double faded = static_cast<double>(sample) * gain;" in apply_fade
    assert "!std::isfinite(faded)" in apply_fade
    assert "faded < kMinNormalizedSample || faded > kMaxNormalizedSample" in apply_fade
    assert "!std::isfinite(out_sample)" in apply_fade
    assert "return false;" in apply_fade
    assert "std::clamp" not in apply_fade


def test_diagnostics_include_mode_duration_position_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaFadeNode::publishDiagnostics")[1].split(
        "}  // namespace fa_fade"
    )[0]

    assert 'status.name = "fa_fade";' in diagnostics
    assert 'pushKeyValue(status, "mode", config_.mode);' in diagnostics
    assert 'pushKeyValue(status, "duration_frames", std::to_string(config_.duration_frames));' in diagnostics
    assert 'pushKeyValue(status, "current_position_frames", std::to_string(position_frames_));' in diagnostics
    assert 'pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_linear_fade.md",
        "config/default.yaml",
        "launch/fa_fade.launch.py",
        "include/fa_fade/fa_fade_node.hpp",
        "src/fa_fade_node.cpp",
        "test/unit/test_fa_fade_audio_frame_contract.py",
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

from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_hum_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_hum"]["ros__parameters"]

    assert params["input_topic"] == "audio/dc_offset_removed/mic"
    assert params["output_topic"] == "audio/hum_removed/mic"
    assert params["hum"]["frequency_hz"] == 60.0
    assert params["hum"]["harmonics"] == 4
    assert params["hum"]["q"] == 30.0
    assert 0.0 < params["hum"]["frequency_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["hum"]["harmonics"] >= 1
    assert params["hum"]["q"] > 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_hum_removal_does_not_hide_other_processing_or_io_responsibilities() -> None:
    source = read_source()

    forbidden = (
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "gain.linear",
        "threshold.linear",
        "std::clamp",
    )
    for token in forbidden:
        assert token not in source


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = read_source()
    load_parameters = source.split("void FaHumNode::loadParameters")[1].split(
        "void FaHumNode::configureCascade"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "!isFinite(config_.frequency_hz)" in load_parameters
    assert "config_.frequency_hz <= 0.0" in load_parameters
    assert "config_.frequency_hz >= nyquist_hz" in load_parameters
    assert "config_.harmonics < 1" in load_parameters
    assert "!isFinite(config_.q)" in load_parameters
    assert "config_.q <= 0.0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "throw std::runtime_error" in load_parameters


def test_hum_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaHumNode::validateFrame")[1].split(
        "bool FaHumNode::applyHumRemoval"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "msg.source_id != active_source_id_" not in validate_frame


def test_source_change_resets_filter_state_instead_of_rejecting_frame() -> None:
    source = read_source()
    handle_frame = source.split("void FaHumNode::handleFrame")[1].split(
        "void FaHumNode::resetFilterStateForSource"
    )[0]
    reset_source = source.split("void FaHumNode::resetFilterStateForSource")[1].split(
        "bool FaHumNode::validateFrame"
    )[0]

    assert "active_source_id_ != msg->source_id" in handle_frame
    assert "resetFilterStateForSource(msg->source_id);" in handle_frame
    assert "active_source_id_ = source_id;" in reset_source
    assert "channel_states_.assign(" in reset_source
    assert "source_resets_.fetch_add(1);" in reset_source


def test_hum_preserves_metadata_and_updates_stream_identity_only() -> None:
    source = read_source()
    apply_hum = source.split("bool FaHumNode::applyHumRemoval")[1].split(
        "void FaHumNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_hum
    assert "out.stream_id = config_.output_topic;" in apply_hum
    assert "out.data.resize(in.data.size());" in apply_hum
    assert "out.encoding =" not in apply_hum
    assert "out.bit_depth =" not in apply_hum
    assert "out.sample_rate =" not in apply_hum
    assert "out.channels =" not in apply_hum
    assert "out.layout =" not in apply_hum
    assert ".rms" not in apply_hum
    assert ".peak" not in apply_hum
    assert ".vad" not in apply_hum


def test_hum_uses_notch_biquad_cascade_for_harmonics_below_nyquist() -> None:
    header = (package_root() / "include" / "fa_hum" / "fa_hum_node.hpp").read_text(encoding="utf-8")
    source = read_source()
    configure_cascade = source.split("void FaHumNode::configureCascade")[1].split(
        "void FaHumNode::setupInterfaces"
    )[0]
    apply_hum = source.split("bool FaHumNode::applyHumRemoval")[1].split(
        "void FaHumNode::publishDiagnostics"
    )[0]

    assert "struct BiquadCoefficients" in header
    assert "struct BiquadState" in header
    assert "using ChannelCascadeState = std::vector<BiquadState>;" in header
    assert "std::vector<BiquadCoefficients> cascade_coefficients_" in header
    assert "std::vector<ChannelCascadeState> channel_states_" in header
    assert "for (int harmonic = 1; harmonic <= config_.harmonics; ++harmonic)" in configure_cascade
    assert "const double center_hz = config_.frequency_hz * static_cast<double>(harmonic);" in configure_cascade
    assert "if (center_hz >= nyquist_hz)" in configure_cascade
    assert "break;" in configure_cascade
    assert "const double alpha = std::sin(omega) / (2.0 * config_.q);" in configure_cascade
    assert "const double a0 = 1.0 + alpha;" in configure_cascade
    assert "coefficients.b0 = 1.0 / a0;" in configure_cascade
    assert "coefficients.b1 = (-2.0 * cos_omega) / a0;" in configure_cascade
    assert "coefficients.b2 = 1.0 / a0;" in configure_cascade
    assert "coefficients.a1 = (-2.0 * cos_omega) / a0;" in configure_cascade
    assert "coefficients.a2 = (1.0 - alpha) / a0;" in configure_cascade
    assert "cascade_coefficients_.empty()" in configure_cascade
    assert "std::vector<ChannelCascadeState> next_states = channel_states_;" in apply_hum
    assert "for (size_t stage = 0; stage < cascade_coefficients_.size(); ++stage)" in apply_hum
    assert "BiquadState & state = channel_state.at(stage);" in apply_hum
    assert "channel_states_ = next_states;" in apply_hum


def test_hum_drops_non_finite_and_out_of_range_samples_without_clamping() -> None:
    source = read_source()
    apply_hum = source.split("bool FaHumNode::applyHumRemoval")[1].split(
        "void FaHumNode::publishDiagnostics"
    )[0]

    assert "!std::isfinite(sample) || !isNormalized(static_cast<double>(sample))" in apply_hum
    assert "!isFinite(stage_output)" in apply_hum
    assert "!isNormalized(filtered)" in apply_hum
    assert "!std::isfinite(out_sample) || !isNormalized(static_cast<double>(out_sample))" in apply_hum
    assert "outside normalized FLOAT32LE range [-1, 1]" in apply_hum
    assert "std::clamp" not in apply_hum
    assert "return false;" in apply_hum


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_notch_cascade.md",
        "config/default.yaml",
        "launch/fa_hum.launch.py",
        "include/fa_hum/fa_hum_node.hpp",
        "src/fa_hum_node.cpp",
        "test/unit/test_fa_hum_audio_frame_contract.py",
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
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

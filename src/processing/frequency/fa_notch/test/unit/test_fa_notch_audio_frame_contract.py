from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_notch_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_notch"]["ros__parameters"]

    assert params["input_topic"] == "audio/high_pass/mic"
    assert params["output_topic"] == "audio/notch/mic"
    assert params["filter"]["center_hz"] == 60.0
    assert params["filter"]["q"] == 30.0
    assert 0.0 < params["filter"]["center_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["filter"]["q"] > 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"


def test_notch_does_not_hide_other_processing_or_io_responsibilities() -> None:
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


def test_notch_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaNotchNode::validateFrame")[1].split(
        "bool FaNotchNode::applyNotch"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.source_id != active_source_id_" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame


def test_notch_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_source()
    apply_notch = source.split("bool FaNotchNode::applyNotch")[1].split(
        "void FaNotchNode::publishDiagnostics"
    )[0]

    assert "active_source_id_ = in.source_id;" in apply_notch
    assert "out = in;" in apply_notch
    assert "out.stream_id = config_.output_topic;" in apply_notch
    assert ".rms" not in apply_notch
    assert ".peak" not in apply_notch
    assert ".vad" not in apply_notch


def test_notch_uses_second_order_biquad_per_channel_state() -> None:
    header = (package_root() / "include" / "fa_notch" / "fa_notch_node.hpp").read_text(
        encoding="utf-8"
    )
    source = read_source()
    configure_filter = source.split("void FaNotchNode::configureFilter")[1].split(
        "void FaNotchNode::setupInterfaces"
    )[0]
    apply_notch = source.split("bool FaNotchNode::applyNotch")[1].split(
        "void FaNotchNode::publishDiagnostics"
    )[0]

    assert "struct BiquadCoefficients" in header
    assert "struct ChannelFilterState" in header
    assert "double previous_input_1" in header
    assert "double previous_input_2" in header
    assert "double previous_output_1" in header
    assert "double previous_output_2" in header
    assert "std::vector<ChannelFilterState> channel_states_" in header
    assert "const double alpha = std::sin(omega) / (2.0 * config_.q);" in configure_filter
    assert "const double a0 = 1.0 + alpha;" in configure_filter
    assert "coefficients_.b0 = 1.0 / a0;" in configure_filter
    assert "coefficients_.b1 = (-2.0 * cos_omega) / a0;" in configure_filter
    assert "coefficients_.b2 = 1.0 / a0;" in configure_filter
    assert "coefficients_.a1 = (-2.0 * cos_omega) / a0;" in configure_filter
    assert "coefficients_.a2 = (1.0 - alpha) / a0;" in configure_filter
    assert "std::vector<ChannelFilterState> next_states = channel_states_;" in apply_notch
    assert "ChannelFilterState & state = next_states.at(i % channel_count);" in apply_notch
    assert "coefficients_.b0 * input +" in apply_notch
    assert "coefficients_.b1 * state.previous_input_1 +" in apply_notch
    assert "coefficients_.b2 * state.previous_input_2 -" in apply_notch
    assert "coefficients_.a1 * state.previous_output_1 -" in apply_notch
    assert "coefficients_.a2 * state.previous_output_2;" in apply_notch
    assert "channel_states_ = next_states;" in apply_notch


def test_filter_parameters_are_required_and_range_checked() -> None:
    source = read_source()
    load_parameters = source.split("void FaNotchNode::loadParameters")[1].split(
        "void FaNotchNode::configureFilter"
    )[0]

    assert 'this->declare_parameter<double>("filter.center_hz", config_.center_hz);' in load_parameters
    assert 'this->declare_parameter<double>("filter.q", config_.q);' in load_parameters
    assert "const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;" in load_parameters
    assert "!isFinite(config_.center_hz)" in load_parameters
    assert "config_.center_hz <= 0.0" in load_parameters
    assert "config_.center_hz >= nyquist_hz" in load_parameters
    assert "!isFinite(config_.q)" in load_parameters
    assert "config_.q <= 0.0" in load_parameters
    assert "filter.center_hz must be finite, > 0.0, and < expected.sample_rate / 2.0" in load_parameters
    assert "filter.q must be finite and > 0.0" in load_parameters


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_notch.md",
        "config/default.yaml",
        "launch/fa_notch.launch.py",
        "include/fa_notch/fa_notch_node.hpp",
        "src/fa_notch_node.cpp",
        "test/unit",
        "test/integration",
        "test/launch",
        "test/fixtures",
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

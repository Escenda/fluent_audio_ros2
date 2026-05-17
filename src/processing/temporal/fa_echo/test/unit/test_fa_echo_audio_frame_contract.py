from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_echo_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_echo"]["ros__parameters"]

    assert params["input_topic"] == "audio/buffered/mic"
    assert params["output_topic"] == "audio/echo/mic"
    assert params["echo"]["delay_ms"] == 250.0
    assert params["echo"]["feedback_gain"] == 0.35
    assert params["echo"]["wet_gain"] == 0.4
    assert params["echo"]["dry_gain"] == 0.8
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_echo_does_not_hide_other_processing_or_io_responsibilities() -> None:
    source = read_source()

    forbidden = (
        "std::clamp",
        "clip",
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "cutoff_hz",
        "center_hz",
        "denoise",
        "reverb",
    )
    for token in forbidden:
        assert token not in source


def test_echo_parameters_are_required_and_range_checked() -> None:
    source = read_source()
    load_parameters = source.split("void FaEchoNode::loadParameters")[1].split(
        "void FaEchoNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter<double>("echo.delay_ms", config_.delay_ms);' in load_parameters
    assert 'this->declare_parameter<double>("echo.feedback_gain", config_.feedback_gain);' in load_parameters
    assert 'this->declare_parameter<double>("echo.wet_gain", config_.wet_gain);' in load_parameters
    assert 'this->declare_parameter<double>("echo.dry_gain", config_.dry_gain);' in load_parameters
    assert "!isFinite(config_.delay_ms) || config_.delay_ms <= 0.0" in load_parameters
    assert "!isFinite(config_.feedback_gain) || std::abs(config_.feedback_gain) >= 1.0" in load_parameters
    assert "!isFinite(config_.wet_gain)" in load_parameters
    assert "!isFinite(config_.dry_gain)" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "delay_samples_ == 0" in load_parameters


def test_echo_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaEchoNode::validateFrame")[1].split(
        "bool FaEchoNode::validateSamples"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_echo_drops_invalid_samples_before_state_mutation() -> None:
    source = read_source()
    handle_frame = source.split("void FaEchoNode::handleFrame")[1].split(
        "bool FaEchoNode::validateFrame"
    )[0]
    validate_samples = source.split("bool FaEchoNode::validateSamples")[1].split(
        "void FaEchoNode::resetDelayState"
    )[0]

    assert "if (!validateSamples(*msg))" in handle_frame
    assert "if (!applyEcho(*msg, out))" in handle_frame
    assert handle_frame.index("if (!validateSamples(*msg))") < handle_frame.index(
        "if (!applyEcho(*msg, out))"
    )
    assert "!isNormalizedSample(sample)" in validate_samples
    assert "return false;" in validate_samples
    assert "std::clamp" not in validate_samples


def test_echo_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_source()
    apply_echo = source.split("bool FaEchoNode::applyEcho")[1].split(
        "size_t FaEchoNode::bytesPerFrame"
    )[0]

    assert "out = in;" in apply_echo
    assert "out.stream_id = config_.output_topic;" in apply_echo
    assert ".rms" not in apply_echo
    assert ".peak" not in apply_echo
    assert ".vad" not in apply_echo


def test_echo_uses_per_channel_ring_buffers_and_feedback_recurrence() -> None:
    header = (package_root() / "include" / "fa_echo" / "fa_echo_node.hpp").read_text(
        encoding="utf-8"
    )
    source = read_source()
    apply_echo = source.split("bool FaEchoNode::applyEcho")[1].split(
        "size_t FaEchoNode::bytesPerFrame"
    )[0]

    assert "std::vector<std::vector<float>> delay_buffers_{};" in header
    assert "std::vector<size_t> delay_positions_{};" in header
    assert "std::vector<std::vector<float>> next_buffers = delay_buffers_;" in apply_echo
    assert "std::vector<size_t> next_positions = delay_positions_;" in apply_echo
    assert "const size_t delay_index = next_positions[channel_index];" in apply_echo
    assert "const float delayed_sample = next_buffers[channel_index][delay_index];" in apply_echo
    assert "config_.dry_gain * static_cast<double>(input_sample)" in apply_echo
    assert "config_.wet_gain * static_cast<double>(delayed_sample)" in apply_echo
    assert "static_cast<double>(input_sample) +" in apply_echo
    assert "config_.feedback_gain * static_cast<double>(delayed_sample)" in apply_echo
    assert "next_buffers[channel_index][delay_index] = next_state_float;" in apply_echo
    assert "next_positions[channel_index] = (delay_index + 1U) % delay_samples_;" in apply_echo


def test_echo_rejects_invalid_output_or_state_without_clipping() -> None:
    source = read_source()
    apply_echo = source.split("bool FaEchoNode::applyEcho")[1].split(
        "size_t FaEchoNode::bytesPerFrame"
    )[0]

    assert "bool isNormalizedSample(float value)" in source
    assert "value >= kMinNormalizedSample && value <= kMaxNormalizedSample" in source
    assert "!isFinite(output_sample) || !isFinite(next_state)" in apply_echo
    assert "!isNormalizedSample(output_float) || !isNormalizedSample(next_state_float)" in apply_echo
    assert "return false;" in apply_echo
    assert "std::clamp" not in apply_echo


def test_echo_resets_state_when_source_id_changes() -> None:
    source = read_source()
    reset_state = source.split("void FaEchoNode::resetDelayState")[1].split(
        "bool FaEchoNode::applyEcho"
    )[0]
    apply_echo = source.split("bool FaEchoNode::applyEcho")[1].split(
        "size_t FaEchoNode::bytesPerFrame"
    )[0]

    assert "std::vector<float>(delay_samples_, kSilenceSample)" in reset_state
    assert "positions.assign(static_cast<size_t>(config_.expected_channels), 0U);" in reset_state
    assert "const bool needs_initialization = current_source_id_.empty();" in apply_echo
    assert "const bool source_changed = !current_source_id_.empty() && in.source_id != current_source_id_;" in apply_echo
    assert "resetDelayState(next_buffers, next_positions);" in apply_echo
    assert "Dropping frame because echo delay state does not match expected channel count" in apply_echo
    assert "source_resets_.fetch_add(1);" in apply_echo
    assert "current_source_id_ = in.source_id;" in apply_echo
    assert "delay_buffers_ = next_buffers;" in apply_echo
    assert "delay_positions_ = next_positions;" in apply_echo


def test_diagnostics_include_echo_source_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaEchoNode::publishDiagnostics")[1].split(
        "}  // namespace fa_echo"
    )[0]

    assert 'status.name = "fa_echo";' in diagnostics
    assert 'pushKeyValue(status, "delay_ms", std::to_string(config_.delay_ms));' in diagnostics
    assert 'pushKeyValue(status, "delay_samples", std::to_string(delay_samples_));' in diagnostics
    assert 'pushKeyValue(status, "feedback_gain", std::to_string(config_.feedback_gain));' in diagnostics
    assert 'pushKeyValue(status, "wet_gain", std::to_string(config_.wet_gain));' in diagnostics
    assert 'pushKeyValue(status, "dry_gain", std::to_string(config_.dry_gain));' in diagnostics
    assert 'pushKeyValue(status, "current_source_id", current_source_id_);' in diagnostics
    assert 'pushKeyValue(status, "messages_in", std::to_string(messages_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "messages_out", std::to_string(messages_out_.load()));' in diagnostics
    assert (
        'pushKeyValue(status, "messages_dropped", std::to_string(messages_dropped_.load()));'
        in diagnostics
    )
    assert 'pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_feedback_echo.md",
        "config/default.yaml",
        "launch/fa_echo.launch.py",
        "include/fa_echo/fa_echo_node.hpp",
        "src/fa_echo_node.cpp",
        "test/unit/test_fa_echo_audio_frame_contract.py",
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

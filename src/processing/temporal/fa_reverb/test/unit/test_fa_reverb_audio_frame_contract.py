from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_reverb_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_reverb"]["ros__parameters"]

    assert params["input_topic"] == "audio/echo/mic"
    assert params["output_topic"] == "audio/reverb/mic"
    assert params["reverb"]["room_size"] == 0.72
    assert params["reverb"]["damping"] == 0.35
    assert params["reverb"]["wet_gain"] == 0.32
    assert params["reverb"]["dry_gain"] == 0.68
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_reverb_does_not_hide_other_processing_or_io_responsibilities() -> None:
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
        "alsa",
    )
    for token in forbidden:
        assert token not in source


def test_reverb_parameters_are_declared_without_runtime_defaults() -> None:
    source = read_source()
    load_parameters = source.split("void FaReverbNode::loadParameters")[1].split(
        "void FaReverbNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<double>("reverb.room_size");' in load_parameters
    assert 'this->declare_parameter<double>("reverb.damping");' in load_parameters
    assert 'this->declare_parameter<double>("reverb.wet_gain");' in load_parameters
    assert 'this->declare_parameter<double>("reverb.dry_gain");' in load_parameters
    assert 'this->declare_parameter<int>("expected.sample_rate");' in load_parameters
    assert 'this->declare_parameter<int>("expected.channels");' in load_parameters
    assert 'this->declare_parameter<std::string>("expected.encoding");' in load_parameters
    assert 'this->declare_parameter<int>("expected.bit_depth");' in load_parameters
    assert 'this->declare_parameter<std::string>("expected.layout");' in load_parameters
    assert 'this->declare_parameter<int>("qos.depth");' in load_parameters
    assert 'this->declare_parameter<bool>("qos.reliable");' in load_parameters
    assert 'this->declare_parameter<int>("diagnostics.publish_period_ms");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredDouble(*this, \"reverb.room_size\")" in load_parameters
    assert "readRequiredBool(*this, \"qos.reliable\")" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line


def test_reverb_parameters_are_range_checked() -> None:
    source = read_source()
    load_parameters = source.split("void FaReverbNode::loadParameters")[1].split(
        "void FaReverbNode::setupInterfaces"
    )[0]

    assert "!isFinite(config_.room_size) || config_.room_size < 0.0 || config_.room_size > 1.0" in load_parameters
    assert "!isFinite(config_.damping) || config_.damping < 0.0 || config_.damping > 1.0" in load_parameters
    assert "!isFinite(config_.wet_gain) || config_.wet_gain < 0.0 || config_.wet_gain > 1.0" in load_parameters
    assert "!isFinite(config_.dry_gain) || config_.dry_gain < 0.0 || config_.dry_gain > 1.0" in load_parameters
    assert "(config_.wet_gain + config_.dry_gain) > 1.0" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_sample_rate > kMaxExpectedSampleRate" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_channels > kMaxExpectedChannels" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters


def test_reverb_validates_float32_interleaved_frame_contract() -> None:
    source = read_source()
    validate_frame = source.split("bool FaReverbNode::validateFrame")[1].split(
        "bool FaReverbNode::validateSamples"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_reverb_drops_invalid_samples_before_state_mutation() -> None:
    source = read_source()
    handle_frame = source.split("void FaReverbNode::handleFrame")[1].split(
        "bool FaReverbNode::validateFrame"
    )[0]
    validate_samples = source.split("bool FaReverbNode::validateSamples")[1].split(
        "void FaReverbNode::resetReverbState"
    )[0]

    assert "if (!validateSamples(*msg))" in handle_frame
    assert "if (!applyReverb(*msg, out))" in handle_frame
    assert handle_frame.index("if (!validateSamples(*msg))") < handle_frame.index(
        "if (!applyReverb(*msg, out))"
    )
    assert "!isNormalizedSample(sample)" in validate_samples
    assert "return false;" in validate_samples
    assert "std::clamp" not in validate_samples


def test_reverb_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_source()
    apply_reverb = source.split("bool FaReverbNode::applyReverb")[1].split(
        "size_t FaReverbNode::bytesPerFrame"
    )[0]

    assert "out = in;" in apply_reverb
    assert "out.stream_id = config_.output_topic;" in apply_reverb
    assert ".rms" not in apply_reverb
    assert ".peak" not in apply_reverb
    assert ".vad" not in apply_reverb


def test_reverb_uses_per_channel_multi_tap_feedback_delay_state() -> None:
    header = (package_root() / "include" / "fa_reverb" / "fa_reverb_node.hpp").read_text(
        encoding="utf-8"
    )
    source = read_source()
    apply_reverb = source.split("bool FaReverbNode::applyReverb")[1].split(
        "size_t FaReverbNode::bytesPerFrame"
    )[0]

    assert "std::vector<std::vector<DelayLineState>> delay_lines_{};" in header
    assert "std::vector<size_t> delay_samples_{};" in header
    assert "std::vector<std::vector<DelayLineState>> next_state = delay_lines_;" in apply_reverb
    assert "resetReverbState(next_state);" in apply_reverb
    assert "for (DelayLineState & line : next_state[channel_index])" in apply_reverb
    assert "const float delayed_sample = line.buffer[delay_index];" in apply_reverb
    assert "wet_sum += static_cast<double>(delayed_sample);" in apply_reverb
    assert "effective_feedback_gain_ * filtered_sample" in apply_reverb
    assert "line.buffer[delay_index] = next_state_float;" in apply_reverb
    assert "line.position = (delay_index + 1U) % line.buffer.size();" in apply_reverb
    assert "const double wet_sample = wet_sum / static_cast<double>(delay_samples_.size());" in apply_reverb


def test_reverb_rejects_invalid_output_or_state_without_clipping() -> None:
    source = read_source()
    apply_reverb = source.split("bool FaReverbNode::applyReverb")[1].split(
        "size_t FaReverbNode::bytesPerFrame"
    )[0]

    assert "bool isNormalizedSample(float value)" in source
    assert "value >= kMinNormalizedSample && value <= kMaxNormalizedSample" in source
    assert "!isFinite(filtered_sample) || !isFinite(next_feedback_state)" in apply_reverb
    assert "!isNormalizedSample(filtered_float) || !isNormalizedSample(next_state_float)" in apply_reverb
    assert "!isFinite(output_sample)" in apply_reverb
    assert "!isNormalizedSample(output_float)" in apply_reverb
    assert "return false;" in apply_reverb
    assert "std::clamp" not in apply_reverb
    assert "delay_lines_ = next_state;" in apply_reverb


def test_reverb_resets_state_when_source_id_changes() -> None:
    source = read_source()
    reset_state = source.split("void FaReverbNode::resetReverbState")[1].split(
        "bool FaReverbNode::validateReverbState"
    )[0]
    apply_reverb = source.split("bool FaReverbNode::applyReverb")[1].split(
        "size_t FaReverbNode::bytesPerFrame"
    )[0]

    assert "state.assign(" in reset_state
    assert "channel_state[line_index].buffer.assign(delay_samples_[line_index], kSilenceSample);" in reset_state
    assert "channel_state[line_index].position = 0U;" in reset_state
    assert "channel_state[line_index].filter_state = kSilenceSample;" in reset_state
    assert "const bool needs_initialization = current_source_id_.empty();" in apply_reverb
    assert "const bool source_changed = !current_source_id_.empty() && in.source_id != current_source_id_;" in apply_reverb
    assert "source_resets_.fetch_add(1);" in apply_reverb
    assert "current_source_id_ = in.source_id;" in apply_reverb
    assert "delay_lines_ = next_state;" in apply_reverb


def test_diagnostics_include_reverb_source_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaReverbNode::publishDiagnostics")[1].split(
        "}  // namespace fa_reverb"
    )[0]

    assert 'status.name = "fa_reverb";' in diagnostics
    assert 'pushKeyValue(status, "room_size", std::to_string(config_.room_size));' in diagnostics
    assert 'pushKeyValue(status, "damping", std::to_string(config_.damping));' in diagnostics
    assert 'pushKeyValue(status, "wet_gain", std::to_string(config_.wet_gain));' in diagnostics
    assert 'pushKeyValue(status, "dry_gain", std::to_string(config_.dry_gain));' in diagnostics
    assert 'pushKeyValue(status, "effective_feedback_gain", std::to_string(effective_feedback_gain_));' in diagnostics
    assert 'pushKeyValue(status, "delay_lines", std::to_string(delay_samples_.size()));' in diagnostics
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
        "docs/backends/internal_feedback_delay.md",
        "config/default.yaml",
        "launch/fa_reverb.launch.py",
        "include/fa_reverb/fa_reverb_node.hpp",
        "src/fa_reverb_node.cpp",
        "test/unit/test_fa_reverb_audio_frame_contract.py",
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

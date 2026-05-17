from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_delay_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_delay"]["ros__parameters"]

    assert params["input_topic"] == "audio/buffered/mic"
    assert params["output_topic"] == "audio/delayed/mic"
    assert params["delay"]["ms"] == 250.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_delay_does_not_hide_other_processing_or_io_responsibilities() -> None:
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
        "reverb",
        "echo",
    )
    for token in forbidden:
        assert token not in source


def test_delay_parameters_are_required_and_range_checked() -> None:
    source = read_source()
    load_parameters = source.split("void FaDelayNode::loadParameters")[1].split(
        "void FaDelayNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter<double>("delay.ms", config_.delay_ms);' in load_parameters
    assert "!std::isfinite(config_.delay_ms) || config_.delay_ms <= 0.0" in load_parameters
    assert "delay.ms must be > 0 and finite" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "delay_samples_ == 0" in load_parameters
    assert "delay.ms must convert to at least 1 sample" in load_parameters


def test_delay_converts_ms_to_whole_samples_from_expected_sample_rate() -> None:
    source = read_source()
    load_parameters = source.split("void FaDelayNode::loadParameters")[1].split(
        "void FaDelayNode::setupInterfaces"
    )[0]

    assert (
        "config_.delay_ms * static_cast<double>(config_.expected_sample_rate) / 1000.0"
        in load_parameters
    )
    assert "delay_samples_ = static_cast<size_t>(std::llround(raw_delay_samples));" in load_parameters


def test_delay_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaDelayNode::validateFrame")[1].split(
        "bool FaDelayNode::validateSamples"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_delay_drops_invalid_samples_before_state_mutation() -> None:
    source = read_source()
    handle_frame = source.split("void FaDelayNode::handleFrame")[1].split(
        "bool FaDelayNode::validateFrame"
    )[0]
    validate_samples = source.split("bool FaDelayNode::validateSamples")[1].split(
        "void FaDelayNode::ensureDelayState"
    )[0]

    assert "if (!validateSamples(*msg))" in handle_frame
    assert "ensureDelayState(msg->source_id);" in handle_frame
    assert handle_frame.index("if (!validateSamples(*msg))") < handle_frame.index(
        "ensureDelayState(msg->source_id);"
    )
    assert "!std::isfinite(sample)" in validate_samples
    assert "sample < kMinNormalizedSample" in validate_samples
    assert "sample > kMaxNormalizedSample" in validate_samples
    assert "return false;" in validate_samples
    assert "std::clamp" not in validate_samples


def test_delay_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_source()
    apply_delay = source.split("bool FaDelayNode::applyDelay")[1].split(
        "size_t FaDelayNode::bytesPerFrame"
    )[0]

    assert "out = in;" in apply_delay
    assert "out.stream_id = config_.output_topic;" in apply_delay
    assert ".rms" not in apply_delay
    assert ".peak" not in apply_delay
    assert ".vad" not in apply_delay


def test_delay_uses_per_channel_buffers_initialized_with_silence() -> None:
    header = (package_root() / "include" / "fa_delay" / "fa_delay_node.hpp").read_text(
        encoding="utf-8"
    )
    source = read_source()
    reset_state = source.split("void FaDelayNode::resetDelayState")[1].split(
        "bool FaDelayNode::applyDelay"
    )[0]
    apply_delay = source.split("bool FaDelayNode::applyDelay")[1].split(
        "size_t FaDelayNode::bytesPerFrame"
    )[0]

    assert "std::vector<std::deque<float>> delay_buffers_{};" in header
    assert "delay_buffers_.assign(" in reset_state
    assert "static_cast<size_t>(config_.expected_channels)" in reset_state
    assert "std::deque<float>(delay_samples_, kSilenceSample)" in reset_state
    assert "const float delayed_sample = delay_buffers_[channel_index].front();" in apply_delay
    assert "delay_buffers_[channel_index].pop_front();" in apply_delay
    assert "delay_buffers_[channel_index].push_back(input_sample);" in apply_delay
    assert "writeFloatSample(out.data, sample_index, delayed_sample);" in apply_delay


def test_delay_resets_state_when_source_id_changes() -> None:
    source = read_source()
    ensure_state = source.split("void FaDelayNode::ensureDelayState")[1].split(
        "void FaDelayNode::resetDelayState"
    )[0]

    assert "source_id == current_source_id_" in ensure_state
    assert "source_id != current_source_id_" in ensure_state
    assert "source_resets_.fetch_add(1);" in ensure_state
    assert "resetDelayState(source_id);" in ensure_state


def test_diagnostics_include_delay_source_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaDelayNode::publishDiagnostics")[1].split(
        "}  // namespace fa_delay"
    )[0]

    assert 'status.name = "fa_delay";' in diagnostics
    assert 'pushKeyValue(status, "delay_ms", std::to_string(config_.delay_ms));' in diagnostics
    assert 'pushKeyValue(status, "delay_samples", std::to_string(delay_samples_));' in diagnostics
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
        "docs/backends/internal_sample_delay.md",
        "config/default.yaml",
        "launch/fa_delay.launch.py",
        "include/fa_delay/fa_delay_node.hpp",
        "src/fa_delay_node.cpp",
        "test/unit/test_fa_delay_audio_frame_contract.py",
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

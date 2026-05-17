from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_low_pass_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_low_pass"]["ros__parameters"]

    assert params["input_topic"] == "audio/sample_format/mic"
    assert params["output_topic"] == "audio/low_pass/mic"
    assert params["filter"]["cutoff_hz"] == 3400.0
    assert 0.0 < params["filter"]["cutoff_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_low_pass_does_not_hide_other_processing_or_io_responsibilities() -> None:
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
        "limiter",
        "denoise",
    )
    for token in forbidden:
        assert token not in source


def test_low_pass_validates_startup_config_fail_closed() -> None:
    source = read_source()
    load_parameters = source.split("void FaLowPassNode::loadParameters")[1].split(
        "void FaLowPassNode::configureFilterState"
    )[0]

    assert 'this->declare_parameter<double>("filter.cutoff_hz", config_.cutoff_hz);' in load_parameters
    assert 'throw std::runtime_error("input_topic is required");' in load_parameters
    assert 'throw std::runtime_error("output_topic is required");' in load_parameters
    assert "const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;" in load_parameters
    assert "!isFinite(config_.cutoff_hz)" in load_parameters
    assert "config_.cutoff_hz <= 0.0" in load_parameters
    assert "config_.cutoff_hz >= nyquist_hz" in load_parameters
    assert "filter.cutoff_hz must be finite, > 0.0, and < expected.sample_rate / 2.0" in load_parameters
    assert "fa_low_pass requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_low_pass requires expected.bit_depth=32" in load_parameters
    assert "fa_low_pass requires expected.layout=interleaved" in load_parameters


def test_low_pass_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaLowPassNode::validateFrame")[1].split(
        "bool FaLowPassNode::applyLowPass"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.source_id != active_source_id_" not in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame


def test_low_pass_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_source()
    apply_low_pass = source.split("bool FaLowPassNode::applyLowPass")[1].split(
        "void FaLowPassNode::publishDiagnostics"
    )[0]

    assert "active_source_id_ = in.source_id;" in apply_low_pass
    assert "out = in;" in apply_low_pass
    assert "out.stream_id = config_.output_topic;" in apply_low_pass
    assert ".rms" not in apply_low_pass
    assert ".peak" not in apply_low_pass
    assert ".vad" not in apply_low_pass


def test_low_pass_uses_first_order_coefficient_and_recurrence_per_channel() -> None:
    header = (package_root() / "include" / "fa_low_pass" / "fa_low_pass_node.hpp").read_text(
        encoding="utf-8"
    )
    source = read_source()
    apply_low_pass = source.split("bool FaLowPassNode::applyLowPass")[1].split(
        "void FaLowPassNode::publishDiagnostics"
    )[0]

    assert "struct ChannelFilterState" in header
    assert "float previous_output" in header
    assert "bool initialized" in header
    assert "std::vector<ChannelFilterState> channel_states_" in header
    assert "filter_alpha_ = sample_interval_sec / (rc_sec + sample_interval_sec);" in source
    assert "std::vector<ChannelFilterState> next_states = channel_states_;" in apply_low_pass
    assert "ChannelFilterState & state = next_states.at(i % channel_count);" in apply_low_pass
    assert "out_sample = sample;" in apply_low_pass
    assert "static_cast<double>(state.previous_output) +" in apply_low_pass
    assert "filter_alpha_ * (static_cast<double>(sample) - static_cast<double>(state.previous_output))" in apply_low_pass
    assert "state.previous_output = out_sample;" in apply_low_pass
    assert "state.initialized = true;" in apply_low_pass
    assert "channel_states_ = next_states;" in apply_low_pass


def test_low_pass_rejects_non_finite_or_out_of_range_samples_without_clamping() -> None:
    source = read_source()
    apply_low_pass = source.split("bool FaLowPassNode::applyLowPass")[1].split(
        "void FaLowPassNode::publishDiagnostics"
    )[0]

    assert "bool isNormalizedSample(float value)" in source
    assert "value >= kNormalizedMin && value <= kNormalizedMax" in source
    assert "!isNormalizedSample(sample)" in apply_low_pass
    assert "!isNormalizedSample(out_sample)" in apply_low_pass
    assert "return false;" in apply_low_pass
    assert "std::clamp" not in apply_low_pass


def test_low_pass_resets_state_on_source_change() -> None:
    source = read_source()
    apply_low_pass = source.split("bool FaLowPassNode::applyLowPass")[1].split(
        "void FaLowPassNode::publishDiagnostics"
    )[0]

    assert "if (active_source_id_.empty() || in.source_id != active_source_id_)" in apply_low_pass
    assert "next_states.assign(static_cast<size_t>(config_.expected_channels), ChannelFilterState{});" in apply_low_pass
    assert "source_resets_.fetch_add(1);" in apply_low_pass


def test_low_pass_diagnostics_include_filter_state_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaLowPassNode::publishDiagnostics")[1]

    assert 'status.name = "fa_low_pass";' in diagnostics
    assert 'pushKeyValue(status, "filter_cutoff_hz", std::to_string(config_.cutoff_hz));' in diagnostics
    assert 'pushKeyValue(status, "filter_alpha", std::to_string(filter_alpha_));' in diagnostics
    assert 'pushKeyValue(status, "state_source_id", active_source_id_);' in diagnostics
    assert 'pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_first_order_low_pass.md",
        "config/default.yaml",
        "launch/fa_low_pass.launch.py",
        "include/fa_low_pass/fa_low_pass_node.hpp",
        "src/fa_low_pass_node.cpp",
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

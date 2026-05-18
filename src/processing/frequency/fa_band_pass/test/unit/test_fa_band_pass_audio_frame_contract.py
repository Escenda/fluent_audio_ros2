from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_band_pass_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_band_pass_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_band_pass"]["ros__parameters"]

    assert params["input_topic"] == "fa_band_pass/input"
    assert params["output_topic"] == "fa_band_pass/output"
    assert params["input_stream_id"] == "audio/sample_format/mic"
    assert params["output"]["stream_id"] == "audio/band_pass/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["filter"]["low_cut_hz"] == 80.0
    assert params["filter"]["high_cut_hz"] == 3400.0
    assert 0.0 < params["filter"]["low_cut_hz"] < params["filter"]["high_cut_hz"]
    assert params["filter"]["high_cut_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is True


def test_band_pass_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [
        read_source(),
        (package_root() / "src" / "backends" / "internal_first_order_band_pass.cpp").read_text(
            encoding="utf-8"
        ),
    ]

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
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_band_pass_validates_startup_config_fail_closed() -> None:
    source = read_source()
    load_parameters = source.split("void FaBandPassNode::loadParameters")[1].split(
        "void FaBandPassNode::configureBackend"
    )[0]

    assert 'readRequiredString(*this, "input_topic")' in load_parameters
    assert 'readRequiredString(*this, "output_topic")' in load_parameters
    assert 'readRequiredString(*this, "input_stream_id")' in load_parameters
    assert 'readRequiredString(*this, "output.stream_id")' in load_parameters
    assert 'readRequiredDouble(*this, "filter.low_cut_hz")' in load_parameters
    assert 'readRequiredDouble(*this, "filter.high_cut_hz")' in load_parameters
    assert 'readRequiredInt(*this, "expected.sample_rate")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    assert 'readRequiredInt(*this, "diagnostics.qos.depth")' in load_parameters
    assert 'readRequiredBool(*this, "diagnostics.qos.reliable")' in load_parameters
    assert "diagnostics.qos.depth must be > 0" in load_parameters
    assert "rclcpp::SystemDefaultsQoS()" not in source
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert 'throw std::runtime_error("input_topic is required");' in load_parameters
    assert 'throw std::runtime_error("output_topic is required");' in load_parameters
    assert "input_stream_id is required" in load_parameters
    assert "output.stream_id is required" in load_parameters
    assert "resolve_topic_name(config_.input_topic)" in load_parameters
    assert "resolve_topic_name(config_.output_topic)" in load_parameters
    assert "input_stream_id must be distinct from ROS topics" in load_parameters
    assert "output.stream_id must be distinct from ROS topics" in load_parameters
    assert "input_stream_id and output.stream_id must be distinct" in load_parameters
    assert "const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;" in load_parameters
    assert "!isFinite(config_.low_cut_hz)" in load_parameters
    assert "config_.low_cut_hz <= 0.0" in load_parameters
    assert "!isFinite(config_.high_cut_hz)" in load_parameters
    assert "config_.high_cut_hz <= config_.low_cut_hz" in load_parameters
    assert "config_.high_cut_hz >= nyquist_hz" in load_parameters
    assert "filter.low_cut_hz must be finite and > 0.0" in load_parameters
    assert "filter.high_cut_hz must be finite, > filter.low_cut_hz, and < expected.sample_rate / 2.0" in load_parameters
    assert "fa_band_pass requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_band_pass requires expected.bit_depth=32" in load_parameters
    assert "fa_band_pass requires expected.layout=interleaved" in load_parameters


def test_band_pass_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaBandPassNode::validateFrame")[1].split(
        "bool FaBandPassNode::applyBandPass"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.source_id != active_source_id_" in validate_frame
    assert "msg.epoch <= *last_epoch_" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame


def test_band_pass_drops_null_frame_before_dereference() -> None:
    source = read_source()
    handle_frame = source.split("void FaBandPassNode::handleFrame")[1].split(
        "bool FaBandPassNode::validateFrame"
    )[0]

    assert "if (!msg)" in handle_frame
    assert '"Dropping null AudioFrame pointer"' in handle_frame
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert handle_frame.index("if (!msg)") < handle_frame.index("validateFrame(*msg)")


def test_band_pass_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_source()
    apply_band_pass = source.split("bool FaBandPassNode::applyBandPass")[1].split(
        "void FaBandPassNode::publishDiagnostics"
    )[0]

    assert "active_source_id_ = in.source_id;" in apply_band_pass
    assert "out = in;" in apply_band_pass
    assert "out.stream_id = config_.output_stream_id;" in apply_band_pass
    assert ".rms" not in apply_band_pass
    assert ".peak" not in apply_band_pass
    assert ".vad" not in apply_band_pass


def test_band_pass_binds_source_only_after_backend_accepts_frame() -> None:
    source = read_source()
    apply_band_pass = source.split("bool FaBandPassNode::applyBandPass")[1].split(
        "void FaBandPassNode::publishDiagnostics"
    )[0]

    assert "const bool should_bind_source = active_source_id_.empty();" in apply_band_pass
    assert (
        "const backends::ProcessStatus status = backend_->process(in.data, out.data, should_reset_state);"
        in apply_band_pass
    )
    assert "if (status != backends::ProcessStatus::kOk)" in apply_band_pass
    assert apply_band_pass.index("backend_->process") < apply_band_pass.index(
        "if (status != backends::ProcessStatus::kOk)"
    )
    assert apply_band_pass.index("if (status != backends::ProcessStatus::kOk)") < apply_band_pass.index(
        "active_source_id_ = in.source_id;"
    )


def test_band_pass_resets_filter_state_on_forward_epoch_gap_only_after_backend_accepts() -> None:
    header = (package_root() / "include" / "fa_band_pass" / "fa_band_pass_node.hpp").read_text(
        encoding="utf-8"
    )
    apply_band_pass = read_source().split("bool FaBandPassNode::applyBandPass")[1].split(
        "void FaBandPassNode::publishDiagnostics"
    )[0]

    assert "#include <optional>" in header
    assert "std::optional<uint32_t> last_epoch_" in header
    assert "std::atomic<uint64_t> state_resets_" in header
    assert "const bool should_reset_state =" in apply_band_pass
    assert "last_epoch_.has_value() && in.epoch != (*last_epoch_ + 1U)" in apply_band_pass
    assert "backend_->process(in.data, out.data, should_reset_state)" in apply_band_pass
    assert "state_resets_.fetch_add(1);" in apply_band_pass
    assert "last_epoch_ = in.epoch;" in apply_band_pass
    assert apply_band_pass.index("if (status != backends::ProcessStatus::kOk)") < apply_band_pass.index(
        "state_resets_.fetch_add(1);"
    )
    assert apply_band_pass.index("if (status != backends::ProcessStatus::kOk)") < apply_band_pass.index(
        "last_epoch_ = in.epoch;"
    )


def test_band_pass_uses_first_order_coefficients_and_recurrence_per_channel() -> None:
    header = (
        package_root() / "include" / "fa_band_pass" / "backends" / "internal_first_order_band_pass.hpp"
    ).read_text(encoding="utf-8")
    source = (package_root() / "src" / "backends" / "internal_first_order_band_pass.cpp").read_text(
        encoding="utf-8"
    )
    process = source.split("ProcessStatus InternalFirstOrderBandPassBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "struct ChannelFilterState" in header
    assert "float previous_hp_input" in header
    assert "float previous_hp_output" in header
    assert "float previous_lp_output" in header
    assert "bool initialized" in header
    assert "std::vector<ChannelFilterState> channel_states_" in header
    assert "hp_alpha_ = rc_hp / (rc_hp + dt);" in source
    assert "lp_alpha_ = dt / (rc_lp + dt);" in source
    assert "reset_state ?" in process
    assert "channel_states_;" in process
    assert "ChannelFilterState & state =" in process
    assert "next_channel_states.at(i % static_cast<size_t>(config_.channels));" in process
    assert "hp_alpha_ * (static_cast<double>(state.previous_hp_output) +" in process
    assert "static_cast<double>(sample) - static_cast<double>(state.previous_hp_input)" in process
    assert "static_cast<double>(state.previous_lp_output) +" in process
    assert "lp_alpha_ * (static_cast<double>(hp_sample) - static_cast<double>(state.previous_lp_output))" in process
    assert "state.previous_hp_input = sample;" in process
    assert "state.previous_hp_output = hp_sample;" in process
    assert "state.previous_lp_output = out_sample;" in process
    assert "state.initialized = true;" in process
    assert "channel_states_ = std::move(next_channel_states);" in process


def test_band_pass_rejects_non_finite_or_out_of_range_samples_without_clamping() -> None:
    source = (package_root() / "src" / "backends" / "internal_first_order_band_pass.cpp").read_text(
        encoding="utf-8"
    )
    header = (
        package_root() / "include" / "fa_band_pass" / "backends" / "internal_first_order_band_pass.hpp"
    ).read_text(encoding="utf-8")

    assert "bool isNormalizedSample(float value)" in source
    assert "value >= kNormalizedMin && value <= kNormalizedMax" in source
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "return ProcessStatus::kOutOfRangeInput;" in source
    assert "return ProcessStatus::kOutOfRangeOutput;" in source
    assert "std::clamp" not in source


def test_band_pass_diagnostics_include_filter_state_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaBandPassNode::publishDiagnostics")[1]

    assert 'status.name = "fa_band_pass";' in diagnostics
    assert 'pushKeyValue(status, "input_topic", config_.input_topic);' in diagnostics
    assert 'pushKeyValue(status, "output_topic", config_.output_topic);' in diagnostics
    assert 'pushKeyValue(status, "input_stream_id", config_.input_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "output_stream_id", config_.output_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "filter_low_cut_hz", std::to_string(config_.low_cut_hz));' in diagnostics
    assert 'pushKeyValue(status, "filter_high_cut_hz", std::to_string(config_.high_cut_hz));' in diagnostics
    assert 'pushKeyValue(status, "hp_alpha", std::to_string(backend_->highPassAlpha()));' in diagnostics
    assert 'pushKeyValue(status, "lp_alpha", std::to_string(backend_->lowPassAlpha()));' in diagnostics
    assert 'pushKeyValue(status, "state_source_id", active_source_id_);' in diagnostics
    assert 'pushKeyValue(status, "state_resets", std::to_string(state_resets_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_first_order_band_pass.md",
        "config/default.yaml",
        "launch/fa_band_pass.launch.py",
        "include/fa_band_pass/fa_band_pass_node.hpp",
        "include/fa_band_pass/backends/internal_first_order_band_pass.hpp",
        "src/fa_band_pass_node.cpp",
        "src/backends/internal_first_order_band_pass.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_first_order_band_pass_backend.cpp",
        "test/cpp/test_band_pass_graph.cpp",
        "test/unit/test_fa_band_pass_audio_frame_contract.py",
        "test/unit",
        "test/integration",
        "test/launch",
        "test/launch/test_fa_band_pass_launch_contract.py",
        "test/fixtures",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "fa_band_pass_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

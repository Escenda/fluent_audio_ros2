from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_low_pass_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_low_pass"]["ros__parameters"]

    assert params["input_topic"] == "audio/resample16k/mic"
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
    sources = [
        read_source(),
        (package_root() / "src" / "backends" / "internal_first_order_low_pass.cpp").read_text(
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


def test_low_pass_validates_startup_config_fail_closed() -> None:
    source = read_source()
    load_parameters = source.split("void FaLowPassNode::loadParameters")[1].split(
        "void FaLowPassNode::configureBackend"
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
    assert "msg.source_id != active_source_id_" in validate_frame
    assert "msg.epoch <= *last_epoch_" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame


def test_low_pass_drops_null_frame_before_dereference() -> None:
    source = read_source()
    handle_frame = source.split("void FaLowPassNode::handleFrame")[1].split(
        "bool FaLowPassNode::validateFrame"
    )[0]

    assert "if (!msg)" in handle_frame
    assert '"Dropping null AudioFrame pointer"' in handle_frame
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert handle_frame.index("if (!msg)") < handle_frame.index("validateFrame(*msg)")


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


def test_low_pass_binds_source_only_after_backend_accepts_frame() -> None:
    source = read_source()
    apply_low_pass = source.split("bool FaLowPassNode::applyLowPass")[1].split(
        "void FaLowPassNode::publishDiagnostics"
    )[0]

    assert "const bool should_bind_source = active_source_id_.empty();" in apply_low_pass
    assert (
        "const backends::ProcessStatus status = backend_->process(in.data, out.data, should_reset_state);"
        in apply_low_pass
    )
    assert "if (status != backends::ProcessStatus::kOk)" in apply_low_pass
    assert apply_low_pass.index("backend_->process") < apply_low_pass.index(
        "if (status != backends::ProcessStatus::kOk)"
    )
    assert apply_low_pass.index("if (status != backends::ProcessStatus::kOk)") < apply_low_pass.index(
        "active_source_id_ = in.source_id;"
    )


def test_low_pass_resets_filter_state_on_forward_epoch_gap_only_after_backend_accepts() -> None:
    header = (package_root() / "include" / "fa_low_pass" / "fa_low_pass_node.hpp").read_text(
        encoding="utf-8"
    )
    apply_low_pass = read_source().split("bool FaLowPassNode::applyLowPass")[1].split(
        "void FaLowPassNode::publishDiagnostics"
    )[0]

    assert "#include <optional>" in header
    assert "std::optional<uint32_t> last_epoch_" in header
    assert "std::atomic<uint64_t> state_resets_" in header
    assert "const bool should_reset_state =" in apply_low_pass
    assert "last_epoch_.has_value() && in.epoch != (*last_epoch_ + 1U)" in apply_low_pass
    assert "backend_->process(in.data, out.data, should_reset_state)" in apply_low_pass
    assert "state_resets_.fetch_add(1);" in apply_low_pass
    assert "last_epoch_ = in.epoch;" in apply_low_pass
    assert apply_low_pass.index("if (status != backends::ProcessStatus::kOk)") < apply_low_pass.index(
        "state_resets_.fetch_add(1);"
    )
    assert apply_low_pass.index("if (status != backends::ProcessStatus::kOk)") < apply_low_pass.index(
        "last_epoch_ = in.epoch;"
    )


def test_low_pass_uses_first_order_coefficient_and_recurrence_per_channel() -> None:
    header = (
        package_root() / "include" / "fa_low_pass" / "backends" / "internal_first_order_low_pass.hpp"
    ).read_text(encoding="utf-8")
    source = (package_root() / "src" / "backends" / "internal_first_order_low_pass.cpp").read_text(
        encoding="utf-8"
    )
    process = source.split("ProcessStatus InternalFirstOrderLowPassBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "struct ChannelFilterState" in header
    assert "float previous_output" in header
    assert "bool initialized" in header
    assert "std::vector<ChannelFilterState> channel_states_" in header
    assert "alpha_ = sample_interval_sec / (rc_sec + sample_interval_sec);" in source
    assert "reset_state ?" in process
    assert "channel_states_;" in process
    assert "ChannelFilterState & state =" in process
    assert "next_channel_states.at(i % static_cast<size_t>(config_.channels));" in process
    assert "out_sample = sample;" in process
    assert "static_cast<double>(state.previous_output) +" in process
    assert "alpha_ * (static_cast<double>(sample) - static_cast<double>(state.previous_output))" in process
    assert "state.previous_output = out_sample;" in process
    assert "state.initialized = true;" in process
    assert "channel_states_ = std::move(next_channel_states);" in process


def test_low_pass_rejects_non_finite_or_out_of_range_samples_without_clamping() -> None:
    source = (package_root() / "src" / "backends" / "internal_first_order_low_pass.cpp").read_text(
        encoding="utf-8"
    )
    header = (
        package_root() / "include" / "fa_low_pass" / "backends" / "internal_first_order_low_pass.hpp"
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


def test_low_pass_diagnostics_include_filter_state_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaLowPassNode::publishDiagnostics")[1]

    assert 'status.name = "fa_low_pass";' in diagnostics
    assert 'pushKeyValue(status, "filter_cutoff_hz", std::to_string(config_.cutoff_hz));' in diagnostics
    assert 'pushKeyValue(status, "filter_alpha", std::to_string(backend_->alpha()));' in diagnostics
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
        "docs/backends/internal_first_order_low_pass.md",
        "config/default.yaml",
        "launch/fa_low_pass.launch.py",
        "include/fa_low_pass/fa_low_pass_node.hpp",
        "include/fa_low_pass/backends/internal_first_order_low_pass.hpp",
        "src/fa_low_pass_node.cpp",
        "src/backends/internal_first_order_low_pass.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_first_order_low_pass_backend.cpp",
        "test/cpp/test_low_pass_graph.cpp",
        "test/unit/test_fa_low_pass_audio_frame_contract.py",
        "test/unit",
        "test/integration",
        "test/launch",
        "test/launch/test_fa_low_pass_launch_contract.py",
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
    assert "fa_low_pass_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

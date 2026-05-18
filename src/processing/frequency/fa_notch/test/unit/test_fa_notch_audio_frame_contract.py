from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_notch_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_notch"]["ros__parameters"]

    assert params["input_topic"] == "fa_notch/input"
    assert params["output_topic"] == "fa_notch/output"
    assert params["input_stream_id"] == "audio/high_pass/mic"
    assert params["output"]["stream_id"] == "audio/notch/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["filter"]["center_hz"] == 60.0
    assert params["filter"]["q"] == 30.0
    assert 0.0 < params["filter"]["center_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["filter"]["q"] > 0.0
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


def test_notch_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [
        read_source(),
        (package_root() / "src" / "backends" / "internal_notch.cpp").read_text(
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


def test_notch_validates_startup_config_fail_closed() -> None:
    source = read_source()
    load_parameters = source.split("void FaNotchNode::loadParameters")[1].split(
        "void FaNotchNode::configureBackend"
    )[0]

    assert 'readRequiredString(*this, "input_topic")' in load_parameters
    assert 'readRequiredString(*this, "output_topic")' in load_parameters
    assert 'readRequiredString(*this, "input_stream_id")' in load_parameters
    assert 'readRequiredString(*this, "output.stream_id")' in load_parameters
    assert 'readRequiredDouble(*this, "filter.center_hz")' in load_parameters
    assert 'readRequiredDouble(*this, "filter.q")' in load_parameters
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
    assert "!isFinite(config_.center_hz)" in load_parameters
    assert "config_.center_hz <= 0.0" in load_parameters
    assert "config_.center_hz >= nyquist_hz" in load_parameters
    assert "!isFinite(config_.q)" in load_parameters
    assert "config_.q <= 0.0" in load_parameters
    assert "filter.center_hz must be finite, > 0.0, and < expected.sample_rate / 2.0" in load_parameters
    assert "filter.q must be finite and > 0.0" in load_parameters
    assert "fa_notch requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_notch requires expected.bit_depth=32" in load_parameters
    assert "fa_notch requires expected.layout=interleaved" in load_parameters


def test_notch_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaNotchNode::validateFrame")[1].split(
        "bool FaNotchNode::applyNotch"
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


def test_notch_drops_null_frame_before_dereference() -> None:
    source = read_source()
    handle_frame = source.split("void FaNotchNode::handleFrame")[1].split(
        "bool FaNotchNode::validateFrame"
    )[0]

    assert "if (!msg)" in handle_frame
    assert '"Dropping null AudioFrame pointer"' in handle_frame
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert handle_frame.index("if (!msg)") < handle_frame.index("validateFrame(*msg)")


def test_notch_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_source()
    apply_notch = source.split("bool FaNotchNode::applyNotch")[1].split(
        "void FaNotchNode::publishDiagnostics"
    )[0]

    assert "active_source_id_ = in.source_id;" in apply_notch
    assert "out = in;" in apply_notch
    assert "out.stream_id = config_.output_stream_id;" in apply_notch
    assert ".rms" not in apply_notch
    assert ".peak" not in apply_notch
    assert ".vad" not in apply_notch


def test_notch_binds_source_only_after_backend_accepts_frame() -> None:
    source = read_source()
    apply_notch = source.split("bool FaNotchNode::applyNotch")[1].split(
        "void FaNotchNode::publishDiagnostics"
    )[0]

    assert "const bool should_bind_source = active_source_id_.empty();" in apply_notch
    assert (
        "const backends::ProcessStatus status = backend_->process(in.data, out.data, should_reset_state);"
        in apply_notch
    )
    assert "if (status != backends::ProcessStatus::kOk)" in apply_notch
    assert apply_notch.index("backend_->process") < apply_notch.index(
        "if (status != backends::ProcessStatus::kOk)"
    )
    assert apply_notch.index("if (status != backends::ProcessStatus::kOk)") < apply_notch.index(
        "active_source_id_ = in.source_id;"
    )


def test_notch_resets_filter_state_on_forward_epoch_gap_only_after_backend_accepts() -> None:
    header = (package_root() / "include" / "fa_notch" / "fa_notch_node.hpp").read_text(
        encoding="utf-8"
    )
    apply_notch = read_source().split("bool FaNotchNode::applyNotch")[1].split(
        "void FaNotchNode::publishDiagnostics"
    )[0]

    assert "#include <optional>" in header
    assert "std::optional<uint32_t> last_epoch_" in header
    assert "std::atomic<uint64_t> state_resets_" in header
    assert "const bool should_reset_state =" in apply_notch
    assert "last_epoch_.has_value() && in.epoch != (*last_epoch_ + 1U)" in apply_notch
    assert "backend_->process(in.data, out.data, should_reset_state)" in apply_notch
    assert "state_resets_.fetch_add(1);" in apply_notch
    assert "last_epoch_ = in.epoch;" in apply_notch
    assert apply_notch.index("if (status != backends::ProcessStatus::kOk)") < apply_notch.index(
        "state_resets_.fetch_add(1);"
    )
    assert apply_notch.index("if (status != backends::ProcessStatus::kOk)") < apply_notch.index(
        "last_epoch_ = in.epoch;"
    )


def test_notch_uses_second_order_biquad_per_channel_state() -> None:
    header = (
        package_root() / "include" / "fa_notch" / "backends" / "internal_notch.hpp"
    ).read_text(encoding="utf-8")
    source = (package_root() / "src" / "backends" / "internal_notch.cpp").read_text(
        encoding="utf-8"
    )
    constructor = source.split("InternalNotchBackend::InternalNotchBackend")[1].split(
        "double InternalNotchBackend::centerHz"
    )[0]
    process = source.split("ProcessStatus InternalNotchBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "struct BiquadCoefficients" in header
    assert "struct ChannelFilterState" in header
    assert "double previous_input_1" in header
    assert "double previous_input_2" in header
    assert "double previous_output_1" in header
    assert "double previous_output_2" in header
    assert "std::vector<ChannelFilterState> channel_states_" in header
    assert "const double alpha = std::sin(omega) / (2.0 * config_.q);" in constructor
    assert "const double a0 = 1.0 + alpha;" in constructor
    assert "coefficients_.b0 = 1.0 / a0;" in constructor
    assert "coefficients_.b1 = (-2.0 * cos_omega) / a0;" in constructor
    assert "coefficients_.b2 = 1.0 / a0;" in constructor
    assert "coefficients_.a1 = (-2.0 * cos_omega) / a0;" in constructor
    assert "coefficients_.a2 = (1.0 - alpha) / a0;" in constructor
    assert "reset_state ?" in process
    assert "channel_states_;" in process
    assert "ChannelFilterState & state =" in process
    assert "next_channel_states.at(i % static_cast<size_t>(config_.channels));" in process
    assert "coefficients_.b0 * input_sample +" in process
    assert "coefficients_.b1 * state.previous_input_1 +" in process
    assert "coefficients_.b2 * state.previous_input_2 -" in process
    assert "coefficients_.a1 * state.previous_output_1 -" in process
    assert "coefficients_.a2 * state.previous_output_2;" in process
    assert "channel_states_ = std::move(next_channel_states);" in process


def test_notch_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    backend_header = (
        package_root() / "include" / "fa_notch" / "backends" / "internal_notch.hpp"
    ).read_text(encoding="utf-8")
    backend_source = (package_root() / "src" / "backends" / "internal_notch.cpp").read_text(
        encoding="utf-8"
    )
    node_source = read_source()

    assert "enum class ProcessStatus" in backend_header
    assert "kEmptyInput" in backend_header
    assert "kMisalignedInput" in backend_header
    assert "kNonFiniteInput" in backend_header
    assert "kNonFiniteOutput" in backend_header
    assert "processStatusMessage(ProcessStatus status)" in backend_header
    assert "return ProcessStatus::kMisalignedInput;" in backend_source
    assert "std::numeric_limits<float>::max()" in backend_source
    assert "backends::processStatusMessage(status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in backend_header
        assert token not in backend_source


def test_notch_diagnostics_include_filter_state_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaNotchNode::publishDiagnostics")[1]

    assert 'status.name = "fa_notch";' in diagnostics
    assert 'pushKeyValue(status, "input_topic", config_.input_topic);' in diagnostics
    assert 'pushKeyValue(status, "output_topic", config_.output_topic);' in diagnostics
    assert 'pushKeyValue(status, "input_stream_id", config_.input_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "output_stream_id", config_.output_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "filter_center_hz", std::to_string(config_.center_hz));' in diagnostics
    assert 'pushKeyValue(status, "filter_q", std::to_string(config_.q));' in diagnostics
    assert 'pushKeyValue(status, "coefficient_b0", std::to_string(coefficients.b0));' in diagnostics
    assert 'pushKeyValue(status, "coefficient_b1", std::to_string(coefficients.b1));' in diagnostics
    assert 'pushKeyValue(status, "coefficient_b2", std::to_string(coefficients.b2));' in diagnostics
    assert 'pushKeyValue(status, "coefficient_a1", std::to_string(coefficients.a1));' in diagnostics
    assert 'pushKeyValue(status, "coefficient_a2", std::to_string(coefficients.a2));' in diagnostics
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
        "docs/backends/internal_notch.md",
        "config/default.yaml",
        "launch/fa_notch.launch.py",
        "include/fa_notch/fa_notch_node.hpp",
        "include/fa_notch/backends/internal_notch.hpp",
        "src/fa_notch_node.cpp",
        "src/backends/internal_notch.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_notch_backend.cpp",
        "test/cpp/test_notch_graph.cpp",
        "test/unit/test_fa_notch_audio_frame_contract.py",
        "test/unit",
        "test/integration",
        "test/launch",
        "test/launch/test_fa_notch_launch_contract.py",
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
    assert "fa_notch_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

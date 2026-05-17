from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def source_text() -> str:
    return (package_root() / "src" / "fa_clock_drift_node.cpp").read_text(encoding="utf-8")


def header_text() -> str:
    return (
        package_root() / "include" / "fa_clock_drift" / "fa_clock_drift_node.hpp"
    ).read_text(encoding="utf-8")


def test_default_config_declares_required_clock_drift_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_clock_drift"]["ros__parameters"]

    assert params["input_topic"] == "audio/sample_format/mic"
    assert params["output_topic"] == "audio/clock_drift_corrected/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["drift"]["ema_alpha"] == 0.1
    assert params["drift"]["max_correction_ms_per_frame"] == 2.0
    assert params["drift"]["reset_threshold_ms"] == 50.0
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "CMakeLists.txt",
        "package.xml",
        "README.md",
        "config/default.yaml",
        "launch/fa_clock_drift.launch.py",
        "include/fa_clock_drift/fa_clock_drift_node.hpp",
        "src/fa_clock_drift_node.cpp",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/sample_clock_timeline.md",
        "test/unit/test_fa_clock_drift_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_node_identity_matches_contract() -> None:
    source = source_text()
    header = header_text()
    launch = (package_root() / "launch" / "fa_clock_drift.launch.py").read_text(encoding="utf-8")

    assert "namespace fa_clock_drift" in header
    assert "class FaClockDriftNode : public rclcpp::Node" in header
    assert ': rclcpp::Node("fa_clock_drift")' in source
    assert "fa_clock_drift::FaClockDriftNode" in source
    assert 'executable="fa_clock_drift_node"' in launch
    assert 'default_value="fa_clock_drift"' in launch


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = source_text()
    load_parameters = source.split("void FaClockDriftNode::loadParameters")[1].split(
        "void FaClockDriftNode::setupInterfaces"
    )[0]

    assert 'declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'declare_parameter<int>("expected.sample_rate");' in load_parameters
    assert 'declare_parameter<double>("drift.ema_alpha");' in load_parameters
    assert 'declare_parameter<double>("drift.max_correction_ms_per_frame");' in load_parameters
    assert 'declare_parameter<double>("drift.reset_threshold_ms");' in load_parameters
    assert 'declare_parameter<bool>("qos.reliable");' in load_parameters
    assert 'declare_parameter<double>("drift.ema_alpha", config_.drift_ema_alpha)' not in load_parameters
    assert 'declare_parameter<bool>("qos.reliable", config_.qos_reliable)' not in load_parameters
    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "expected.sample_rate must be > 0" in load_parameters
    assert "expected.channels must be > 0" in load_parameters
    assert "expected.encoding is required" in load_parameters
    assert "expected.bit_depth must be > 0" in load_parameters
    assert "expected.bit_depth must be byte-aligned" in load_parameters
    assert "expected.layout is required" in load_parameters
    assert "!std::isfinite(config_.drift_ema_alpha)" in load_parameters
    assert "config_.drift_ema_alpha <= 0.0" in load_parameters
    assert "config_.drift_ema_alpha > 1.0" in load_parameters
    assert "!std::isfinite(config_.drift_max_correction_ms_per_frame)" in load_parameters
    assert "config_.drift_max_correction_ms_per_frame < 0.0" in load_parameters
    assert "!std::isfinite(config_.drift_reset_threshold_ms)" in load_parameters
    assert "config_.drift_reset_threshold_ms <= 0.0" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters


def test_runtime_validation_rejects_invalid_frames_before_state_mutation() -> None:
    source = source_text()
    validate_frame = source.split("bool FaClockDriftNode::validateFrame")[1].split(
        "bool FaClockDriftNode::correctFrame"
    )[0]
    handle_frame = source.split("void FaClockDriftNode::handleFrame")[1].split(
        "bool FaClockDriftNode::validateFrame"
    )[0]
    correct_frame = source.split("bool FaClockDriftNode::correctFrame")[1].split(
        "bool FaClockDriftNode::publishBaselineFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerSampleFrame()) != 0U" in validate_frame
    assert "if (!validateFrame(*msg))" in handle_frame
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index(
        "if (!correctFrame(*msg, out))"
    )
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert "hasDifferentStreamIdentity(in)" in correct_frame
    assert "previous_output_timestamp_ns_.has_value()" in correct_frame


def test_source_and_format_change_resets_timeline_and_publishes_baseline() -> None:
    source = source_text()
    has_different = source.split("bool FaClockDriftNode::hasDifferentStreamIdentity")[1].split(
        "void FaClockDriftNode::activateStream"
    )[0]
    correct_frame = source.split("bool FaClockDriftNode::correctFrame")[1].split(
        "bool FaClockDriftNode::publishBaselineFrame"
    )[0]
    baseline = source.split("bool FaClockDriftNode::publishBaselineFrame")[1].split(
        "bool FaClockDriftNode::hasDifferentStreamIdentity"
    )[0]

    assert "msg.source_id != active_stream_->source_id" in has_different
    assert "msg.stream_id != active_stream_->stream_id" in has_different
    assert "msg.sample_rate != active_stream_->sample_rate" in has_different
    assert "msg.channels != active_stream_->channels" in has_different
    assert "msg.encoding != active_stream_->encoding" in has_different
    assert "msg.bit_depth != active_stream_->bit_depth" in has_different
    assert "msg.layout != active_stream_->layout" in has_different
    assert "resetTimeline();" in correct_frame
    assert "timeline_resets_.fetch_add(1);" in correct_frame
    assert "return publishBaselineFrame(in, out, 0.0L, current_frame_duration_ns);" in correct_frame
    assert "activateStream(in, input_timestamp_ns);" in baseline
    assert "previous_frame_duration_ns_ = current_frame_duration_ns;" in baseline
    assert "drift_estimate_ns_ = 0.0L;" in baseline


def test_drift_estimate_and_limit_logic_are_explicit() -> None:
    source = source_text()
    header = header_text()
    correct_frame = source.split("bool FaClockDriftNode::correctFrame")[1].split(
        "bool FaClockDriftNode::publishBaselineFrame"
    )[0]
    bound_correction = source.split("long double FaClockDriftNode::boundCorrectionNanoseconds")[1].split(
        "size_t FaClockDriftNode::bytesPerSampleFrame"
    )[0]

    assert "std::optional<long double> previous_output_timestamp_ns_" in header
    assert "std::optional<long double> previous_frame_duration_ns_" in header
    assert "long double drift_estimate_ns_" in header
    assert "frameDurationNanoseconds(in, current_frame_duration_ns)" in correct_frame
    assert "previous_frame_duration_ns_.has_value()" in correct_frame
    assert "const long double expected_timestamp_ns =" in correct_frame
    assert "*previous_output_timestamp_ns_ + *previous_frame_duration_ns_" in correct_frame
    assert "previous_frame_duration_ns_ = current_frame_duration_ns;" in correct_frame
    assert "const long double observed_drift_ns = input_timestamp_ns - expected_timestamp_ns;" in correct_frame
    assert "((1.0L - alpha) * drift_estimate_ns_) + (alpha * observed_drift_ns)" in correct_frame
    assert "boundCorrectionNanoseconds(next_estimate_ns, *previous_frame_duration_ns_)" in correct_frame
    assert "correction_limited_frames_.fetch_add(1);" in correct_frame
    assert "const long double output_timestamp_ns = expected_timestamp_ns + bounded_correction_ns;" in correct_frame
    assert "if (correction_ns > max_correction_ns)" in bound_correction
    assert "long double negative_limit_ns = max_correction_ns;" in bound_correction
    assert "previous_frame_duration_ns <= 1.0L" in bound_correction
    assert "negative_limit_ns = 0.0L;" in bound_correction
    assert "negative_limit_ns > previous_frame_duration_ns - 1.0L" in bound_correction
    assert "negative_limit_ns = previous_frame_duration_ns - 1.0L;" in bound_correction
    assert "if (correction_ns < -negative_limit_ns)" in bound_correction
    assert "std::clamp" not in source


def test_python_clock_drift_model_uses_previous_duration_and_never_regresses() -> None:
    previous_output_ns = 1_000_000_000.0
    previous_duration_ns = 1_000_000.0
    current_duration_ns = 3_000_000.0
    observed_input_ns = 999_000_000.0
    max_correction_ns = 2_000_000.0

    expected_ns = previous_output_ns + previous_duration_ns
    observed_drift_ns = observed_input_ns - expected_ns
    negative_limit_ns = min(max_correction_ns, previous_duration_ns - 1.0)
    bounded_correction_ns = max(observed_drift_ns, -negative_limit_ns)
    output_ns = expected_ns + bounded_correction_ns

    assert expected_ns == previous_output_ns + previous_duration_ns
    assert expected_ns != previous_output_ns + current_duration_ns
    assert output_ns > previous_output_ns
    assert output_ns == previous_output_ns + 1.0


def test_reset_threshold_starts_new_baseline() -> None:
    source = source_text()
    correct_frame = source.split("bool FaClockDriftNode::correctFrame")[1].split(
        "bool FaClockDriftNode::publishBaselineFrame"
    )[0]

    assert "config_.drift_reset_threshold_ms" in correct_frame
    assert "std::fabs(observed_drift_ns) > reset_threshold_ns" in correct_frame
    assert "timeline_resets_.fetch_add(1);" in correct_frame
    assert "return publishBaselineFrame(in, out, observed_drift_ns, current_frame_duration_ns);" in correct_frame


def test_timestamp_fail_closed_behavior_resets_state() -> None:
    source = source_text()
    build_stamp = source.split("bool FaClockDriftNode::buildStamp")[1].split(
        "long double FaClockDriftNode::boundCorrectionNanoseconds"
    )[0]
    stamp_to_ns = source.split("bool FaClockDriftNode::stampToNanoseconds")[1].split(
        "bool FaClockDriftNode::buildStamp"
    )[0]
    correct_frame = source.split("bool FaClockDriftNode::correctFrame")[1].split(
        "bool FaClockDriftNode::publishBaselineFrame"
    )[0]

    assert "stamp.nanosec >= static_cast<uint32_t>(kNanosecondsPerSecond)" in stamp_to_ns
    assert "std::isfinite(timestamp_ns)" in stamp_to_ns
    assert "!std::isfinite(timestamp_ns)" in build_stamp
    assert "timestamp_ns < 0.0L" in build_stamp
    assert "timestamp_ns > kMaxBuiltinTimeNanoseconds" in build_stamp
    assert "static_cast<long double>(rounded_ns) > kMaxBuiltinTimeNanoseconds" in build_stamp
    assert "if (!stampToNanoseconds(in.header.stamp, input_timestamp_ns))" in correct_frame
    assert "if (!buildStamp(output_timestamp_ns, output_stamp))" in correct_frame
    assert "resetTimeline();" in correct_frame
    assert "return false;" in correct_frame


def test_output_preserves_payload_epoch_and_updates_stamp_stream_id_only() -> None:
    source = source_text()
    correct_frame = source.split("bool FaClockDriftNode::correctFrame")[1].split(
        "bool FaClockDriftNode::publishBaselineFrame"
    )[0]
    baseline = source.split("bool FaClockDriftNode::publishBaselineFrame")[1].split(
        "bool FaClockDriftNode::hasDifferentStreamIdentity"
    )[0]

    assert "out = in;" in correct_frame
    assert "out.header.stamp = output_stamp;" in correct_frame
    assert "out.stream_id = config_.output_topic;" in correct_frame
    assert "out.data" not in correct_frame
    assert "out.epoch" not in correct_frame
    assert "out.source_id" not in correct_frame
    assert "out = in;" in baseline
    assert "out.header.stamp = output_stamp;" in baseline
    assert "out.stream_id = config_.output_topic;" in baseline
    assert "out.data" not in baseline
    assert "out.epoch" not in baseline
    assert "out.source_id" not in baseline


def test_processing_node_has_no_device_io_sample_editing_or_legacy_aliases() -> None:
    source = source_text()
    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "normalize(",
        "std::clamp",
        "padding",
        "legacy",
        "deprecated",
        "alias",
        "reinterpret_cast",
        "std::memcpy",
    )
    for token in forbidden:
        assert token not in source


def test_diagnostics_publish_config_counters_and_drift_state() -> None:
    source = source_text()
    header = header_text()
    diagnostics = source.split("void FaClockDriftNode::publishDiagnostics")[1].split(
        "}  // namespace fa_clock_drift"
    )[0]

    assert "std::atomic<uint64_t> frames_in_" in header
    assert "std::atomic<uint64_t> frames_out_" in header
    assert "std::atomic<uint64_t> frames_dropped_" in header
    assert "std::atomic<uint64_t> timeline_resets_" in header
    assert "std::atomic<uint64_t> correction_limited_frames_" in header
    assert '"input_topic"' in diagnostics
    assert '"output_topic"' in diagnostics
    assert '"expected_sample_rate"' in diagnostics
    assert '"expected_channels"' in diagnostics
    assert '"expected_encoding"' in diagnostics
    assert '"expected_bit_depth"' in diagnostics
    assert '"expected_layout"' in diagnostics
    assert '"drift_ema_alpha"' in diagnostics
    assert '"drift_max_correction_ms_per_frame"' in diagnostics
    assert '"drift_reset_threshold_ms"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"timeline_resets"' in diagnostics
    assert '"drift_estimate_ms"' in diagnostics
    assert '"last_observed_drift_ms"' in diagnostics
    assert '"correction_limited_frames"' in diagnostics


def test_colcon_runs_pytest_contracts_and_lint_dependencies() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "find_package(ament_lint_auto REQUIRED)" in cmake_text
    assert "find_package(builtin_interfaces REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "ament_lint_auto_find_test_dependencies()" in cmake_text
    assert "<depend>builtin_interfaces</depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

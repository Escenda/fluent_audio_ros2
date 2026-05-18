from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_aec_linear_node.cpp").read_text(
        encoding="utf-8"
    )


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "baseline_linear.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_drops_reference_failures() -> None:
    config_path = package_root() / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    params = config["fa_aec_linear"]["ros__parameters"]

    assert params["reference_failure_policy"] == "drop"
    assert params["expected_channels"] == 1
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["mic_stream_id"] == "audio/mic/resample16k"
    assert params["ref_stream_id"] == "audio/ref/resample16k"
    assert params["output"]["stream_id"] == "audio/aec_linear/output"
    assert "-1" not in config_text


def test_aec_linear_outputs_audio_frame_without_analysis_metadata() -> None:
    source = read_node_source()

    assert "FrameValidationStatus::kMissingSourceId" in source
    assert "FrameValidationStatus::kStreamIdMismatch" in source
    assert "FrameValidationStatus::kInvalidTimestamp" in source
    assert "fa_aec_linear received a null reference AudioFrame pointer" in source
    assert "fa_aec_linear received a null mic AudioFrame pointer" in source
    assert "fa_aec_linear publisher is not initialized" in source
    assert "validateFrame(*msg, config_.ref_stream_id)" in source
    assert "validateFrame(*msg, config_.mic_stream_id)" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "out_msg.source_id = msg->source_id;" in source
    assert "out_msg.stream_id = config_.output_stream_id;" in source
    assert "out_msg.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaAecLinearNode::loadParameters")[1].split(
        "if (config_.mic_topic.empty())"
    )[0]

    required_reads = (
        'readRequiredBool(*this, "enabled")',
        'readRequiredString(*this, "mic_topic")',
        'readRequiredString(*this, "ref_topic")',
        'readRequiredString(*this, "output_topic")',
        'readRequiredString(*this, "mic_stream_id")',
        'readRequiredString(*this, "ref_stream_id")',
        'readRequiredString(*this, "output.stream_id")',
        'readRequiredInt(*this, "expected_sample_rate")',
        'readRequiredInt(*this, "expected_channels")',
        'readRequiredString(*this, "expected.encoding")',
        'readRequiredInt(*this, "expected.bit_depth")',
        'readRequiredInt(*this, "ref_timeout_ms")',
        'readRequiredString(*this, "reference_failure_policy")',
        'readRequiredDouble(*this, "cancel_gain")',
        'readRequiredInt(*this, "qos.depth")',
        'readRequiredInt(*this, "diagnostics.qos.depth")',
        'readRequiredBool(*this, "qos.reliable")',
        'readRequiredBool(*this, "diagnostics.qos.reliable")',
    )
    for read in required_reads:
      assert read in load_parameters

    assert "readRequiredInt(" in load_parameters
    assert '"diagnostics.publish_period_ms"' in load_parameters
    assert "this->get_parameter(" not in load_parameters
    assert "SystemDefaultsQoS" not in source
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert "config_." not in line


def test_aec_linear_has_ros_free_baseline_backend() -> None:
    backend_header = (
        package_root() / "include" / "fa_aec_linear" / "backends" / "baseline_linear.hpp"
    ).read_text(encoding="utf-8")
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "BaselineLinearBackend" in backend_header
    assert "ProcessStatus" in backend_header
    assert "ProcessResult" in backend_header
    assert "fa_interfaces/msg/audio_frame" not in backend_header
    assert "rclcpp" not in backend_header
    assert "AudioFrame" not in backend_source
    assert "backend_->process(msg->data, ref->data, out_bytes)" in node_source
    assert "baseline_linear backend rejected input or output" in node_source


def test_aec_linear_rejects_ambiguous_format_and_hidden_clamp() -> None:
    package = package_root()
    source = read_node_source()
    backend = read_backend_source()
    header = (package / "include" / "fa_aec_linear" / "fa_aec_linear_node.hpp").read_text(
        encoding="utf-8"
    )
    spec = (package / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    test_plan = (package / "docs" / "テスト設計.md").read_text(encoding="utf-8")

    assert "isSupportedAudioFormatPair" in backend
    assert "PCM16LE" in backend
    assert "FLOAT32LE" in backend
    assert "PCM32LE" not in source + backend
    assert "msg.encoding != config_.expected_encoding" in source
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in source
    assert "ref->encoding != msg->encoding" in source
    assert "ProcessStatus::kSampleCountMismatch" in backend
    assert "expected encoding/bit_depth must be PCM16LE/16 or FLOAT32LE/32" in source + backend
    assert "ProcessStatus::kOutOfRangeOutput" in backend
    assert "Add an explicit dynamics/limiter node if range control is required" in source
    assert "std::clamp" not in source + backend
    assert "decodeToFloat" not in header
    assert "encodeFromFloat" not in header
    assert "hidden clamp" in spec
    assert "PCM32LE/32" in algorithm
    assert "output range overflow は clamp せず drop" in test_plan


def test_aec_linear_rejects_channel_wildcards_and_documents_stream_binding() -> None:
    package = package_root()
    source = read_node_source()
    spec = (package / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    test_plan = (package / "docs" / "テスト設計.md").read_text(encoding="utf-8")

    assert "config_.expected_channels <= 0" in source
    assert "config_.expected_channels > 0" not in source
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in source
    assert "channel 検査の無効化は禁止" in spec
    assert "mic frame の `stream_id` は `mic_stream_id`" in spec
    assert "reference frame の `stream_id` は `ref_stream_id`" in spec
    assert "channel 検査の wildcard はない" in algorithm
    assert "stream_id` が `mic_stream_id`" in test_plan
    assert "stream_id` が `ref_stream_id`" in test_plan


def test_aec_linear_uses_header_stamp_reference_contract() -> None:
    source = read_node_source()
    spec = (package_root() / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package_root() / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )

    assert "hasValidStamp(msg.header.stamp)" in source
    assert "last_ref_stamp_ = rclcpp::Time(msg->header.stamp, RCL_ROS_TIME);" in source
    assert "const rclcpp::Time mic_stamp(msg->header.stamp, RCL_ROS_TIME);" in source
    assert "mic_stamp - ref_stamp" in source
    assert "ref_skew_ms >= 0" in source
    assert "ref_skew_ms <= config_.ref_timeout_ms" in source
    assert "header.stamp" in spec
    assert "reference は mic と同時刻または過去" in algorithm


def test_aec_linear_rejects_resolved_topic_feedback_loops() -> None:
    source = read_node_source()

    assert "resolve_topic_name(config_.mic_topic)" in source
    assert "resolve_topic_name(config_.ref_topic)" in source
    assert "resolve_topic_name(config_.output_topic)" in source
    assert "config_.resolved_mic_topic == config_.resolved_ref_topic" in source
    assert "config_.resolved_mic_topic == config_.resolved_output_topic" in source
    assert "config_.resolved_ref_topic == config_.resolved_output_topic" in source
    assert "sameIdentityString(config_.mic_topic, config_.ref_topic)" in source
    assert "mic_stream_id, ref_stream_id, and output.stream_id must be distinct" in source
    assert (
        "mic_stream_id, ref_stream_id, and output.stream_id must be distinct from ROS topics"
        in source
    )


def test_main_shutdown_is_guarded_by_rclcpp_ok_and_manifest_declares_launch_deps() -> None:
    package = package_root()
    main_source = (package / "src" / "main.cpp").read_text(encoding="utf-8")
    package_xml = (package / "package.xml").read_text(encoding="utf-8")
    cmake_text = (package / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "if (rclcpp::ok()) {\n      rclcpp::shutdown();" in main_source
    assert "}\n  if (rclcpp::ok()) {\n    rclcpp::shutdown();" in main_source
    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "test/cpp/test_baseline_linear_backend.cpp" in cmake_text

from pathlib import Path

import yaml


def test_default_config_declares_required_latency_compensation_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_latency_compensation"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"
    assert params["output_topic"] == "audio/latency_compensated/frame"
    assert params["input_stream_id"] == "audio/preprocessed/mono16k"
    assert params["output"]["stream_id"] == "audio/latency_compensated/mono16k"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["compensation"]["offset_ms"] == 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_latency_compensation_does_not_hide_io_or_audio_sample_editing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_latency_compensation_node.cpp").read_text(
        encoding="utf-8"
    )

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "gain.linear",
        "threshold.linear",
        "filter.",
        "noise",
        "denoise",
        "normalize(",
        "std::clamp",
        "memcpy",
        "reinterpret_cast",
    )
    for token in forbidden:
        assert token not in source


def test_node_identity_matches_contract() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_latency_compensation_node.cpp").read_text(
        encoding="utf-8"
    )
    header = (
        package_root
        / "include"
        / "fa_latency_compensation"
        / "fa_latency_compensation_node.hpp"
    ).read_text(encoding="utf-8")
    main_source = (package_root / "src" / "main.cpp").read_text(encoding="utf-8")
    launch = (package_root / "launch" / "fa_latency_compensation.launch.py").read_text(
        encoding="utf-8"
    )

    assert "namespace fa_latency_compensation" in header
    assert "class FaLatencyCompensationNode : public rclcpp::Node" in header
    assert 'rclcpp::Node("fa_latency_compensation", options)' in source
    assert "explicit FaLatencyCompensationNode(" in header
    assert "fa_latency_compensation::FaLatencyCompensationNode" in main_source
    assert 'executable="fa_latency_compensation_node"' in launch
    assert "default_value" not in launch
    assert "FindPackageShare" not in launch


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_latency_compensation_node.cpp").read_text(
        encoding="utf-8"
    )
    load_parameters = source.split("void FaLatencyCompensationNode::loadParameters")[1].split(
        "void FaLatencyCompensationNode::setupInterfaces"
    )[0]

    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"input_stream_id is required\")" in load_parameters
    assert "throw std::runtime_error(\"output.stream_id is required\")" in load_parameters
    assert 'declare_parameter<double>("compensation.offset_ms");' in load_parameters
    assert 'declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'declare_parameter<bool>("qos.reliable");' in load_parameters
    assert 'declare_parameter<double>("compensation.offset_ms", config_.offset_ms)' not in load_parameters
    assert 'declare_parameter<bool>("qos.reliable", config_.qos_reliable)' not in load_parameters
    required_reads = (
        'readRequiredString(*this, "input_topic")',
        'readRequiredString(*this, "output_topic")',
        'readRequiredString(*this, "input_stream_id")',
        'readRequiredString(*this, "output.stream_id")',
        'readRequiredDouble(*this, "compensation.offset_ms")',
        'readRequiredInt(*this, "expected.sample_rate")',
        'readRequiredInt(*this, "expected.channels")',
        'readRequiredString(*this, "expected.encoding")',
        'readRequiredInt(*this, "expected.bit_depth")',
        'readRequiredString(*this, "expected.layout")',
        'readRequiredInt(*this, "qos.depth")',
        'readRequiredBool(*this, "qos.reliable")',
        'readRequiredInt(*this, "diagnostics.qos.depth")',
        'readRequiredBool(*this, "diagnostics.qos.reliable")',
    )
    for read in required_reads:
        assert read in load_parameters
    assert "readRequiredInt(" in load_parameters
    assert '"diagnostics.publish_period_ms"' in load_parameters
    assert "sameIdentityString(config_.input_stream_id, config_.input_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, config_.output_topic)" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, resolved_input_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, resolved_output_topic)" in load_parameters
    assert "this->get_parameter(" not in load_parameters
    assert "SystemDefaultsQoS" not in source
    assert "std::isfinite(config_.offset_ms)" in load_parameters
    assert "compensation.offset_ms must be finite" in load_parameters
    assert "compensation.offset_ms exceeds int64 nanosecond range" in load_parameters
    assert "expected.sample_rate must be > 0" in load_parameters
    assert "expected.channels must be > 0" in load_parameters
    assert "expected.encoding is required" in load_parameters
    assert "expected.bit_depth must be > 0" in load_parameters
    assert "expected.bit_depth must be byte-aligned" in load_parameters
    assert "expected.layout is required" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters
    assert "diagnostics.qos.depth must be > 0" in load_parameters
    assert "input_stream_id and output.stream_id must be distinct" in load_parameters


def test_runtime_validates_audio_frame_contract_before_compensation() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_latency_compensation_node.cpp").read_text(
        encoding="utf-8"
    )
    validate_frame = source.split("bool FaLatencyCompensationNode::validateFrame")[1].split(
        "bool FaLatencyCompensationNode::compensateFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_compensation_updates_only_stamp_and_stream_identity_after_copy() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_latency_compensation_node.cpp").read_text(
        encoding="utf-8"
    )
    compensate_frame = source.split("bool FaLatencyCompensationNode::compensateFrame")[1].split(
        "size_t FaLatencyCompensationNode::bytesPerFrame"
    )[0]

    assert "const rclcpp::Time input_stamp(in.header.stamp);" in compensate_frame
    assert "std::llround(config_.offset_ms * kNanosecondsPerMillisecond)" in compensate_frame
    assert "input_stamp.nanoseconds() > std::numeric_limits<int64_t>::max() - offset_ns" in compensate_frame
    assert "input_stamp.nanoseconds() < std::numeric_limits<int64_t>::min() - offset_ns" in compensate_frame
    assert "const int64_t adjusted_ns = input_stamp.nanoseconds() + offset_ns;" in compensate_frame
    assert "if (adjusted_ns < 0)" in compensate_frame
    assert "if (adjusted_ns > kMaxBuiltinTimeNanoseconds)" in compensate_frame
    assert "exceeds builtin_interfaces/Time range" in compensate_frame
    assert "return false;" in compensate_frame
    assert "out = in;" in compensate_frame
    assert "out.header.stamp = nanosecondsToStamp(adjusted_ns);" in compensate_frame
    assert "out.stream_id = config_.output_stream_id;" in compensate_frame
    assert "out.data" not in compensate_frame


def test_negative_timestamp_drop_and_diagnostics_counters_are_explicit() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_latency_compensation_node.cpp").read_text(
        encoding="utf-8"
    )
    header = (
        package_root
        / "include"
        / "fa_latency_compensation"
        / "fa_latency_compensation_node.hpp"
    ).read_text(encoding="utf-8")
    handle_frame = source.split("void FaLatencyCompensationNode::handleFrame")[1].split(
        "bool FaLatencyCompensationNode::validateFrame"
    )[0]
    diagnostics = source.split("void FaLatencyCompensationNode::publishDiagnostics")[1].split(
        "}  // namespace fa_latency_compensation"
    )[0]

    assert "std::atomic<uint64_t> negative_timestamp_drops_" in header
    assert "std::atomic<uint64_t> timestamp_overflow_drops_" in header
    assert "if (!compensateFrame(*msg, out))" in handle_frame
    assert "audio_pub_->publish(out);" in handle_frame
    assert "frames_in" in diagnostics
    assert "frames_out" in diagnostics
    assert "frames_dropped" in diagnostics
    assert 'pushKeyValue(status, "input_stream_id", config_.input_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "output_stream_id", config_.output_stream_id);' in diagnostics
    assert "negative_timestamp_drops" in diagnostics
    assert "timestamp_overflow_drops" in diagnostics


def test_package_layout_matches_required_streaming_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "CMakeLists.txt",
        "package.xml",
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/no_runtime_backend.md",
        "config/default.yaml",
        "launch/fa_latency_compensation.launch.py",
        "include/fa_latency_compensation/fa_latency_compensation_node.hpp",
        "src/fa_latency_compensation_node.cpp",
        "src/main.cpp",
        "test/unit/test_fa_latency_compensation_audio_frame_contract.py",
        "test/cpp/test_fa_latency_compensation_node_contract.cpp",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root / relative_path).exists()


def test_colcon_runs_pytest_contracts_and_lint_dependencies() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_lint_auto REQUIRED)" in cmake_text
    assert "add_library(fa_latency_compensation_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_node_contract_test" in cmake_text
    assert "ament_lint_auto_find_test_dependencies()" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

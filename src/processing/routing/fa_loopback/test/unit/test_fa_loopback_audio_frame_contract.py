from pathlib import Path

import yaml


def test_default_config_defines_explicit_loopback_routing_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    launch_text = (package_root / "launch" / "fa_loopback.launch.py").read_text(encoding="utf-8")
    params = config["fa_loopback"]["ros__parameters"]

    assert params["input_topic"] == "fa_loopback/input"
    assert params["output_topic"] == "fa_loopback/output"
    assert params["input_stream_id"] == "audio/output/frame"
    assert params["output"]["stream_id"] == "audio/loopback/frame"
    assert params["input_topic"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["input_stream_id"] not in (params["input_topic"], params["output_topic"])
    assert params["output"]["stream_id"] not in (params["input_topic"], params["output_topic"])
    assert "loopback" not in params
    assert params["expected"]["sample_rate"] == 48000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is True
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
    assert '"node_name"' in launch_text
    assert '"config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_loopback_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaLoopbackNode::loadParameters")[1].split(
        "void FaLoopbackNode::setupInterfaces"
    )[0]

    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"input_stream_id is required\")" in load_parameters
    assert "throw std::runtime_error(\"output.stream_id is required\")" in load_parameters
    assert "input_topic and output_topic must differ after resolution" in load_parameters
    assert "input_stream_id and output.stream_id must be distinct" in load_parameters
    assert "input_stream_id must not match raw or resolved topic names" in load_parameters
    assert "output.stream_id must not match raw or resolved topic names" in load_parameters
    assert "expected.sample_rate must be in (0, 384000]" in load_parameters
    assert "expected.channels must be in (0, 64]" in load_parameters
    assert "expected.encoding is required" in load_parameters
    assert "expected.bit_depth must be > 0" in load_parameters
    assert "expected.bit_depth must be byte-aligned" in load_parameters
    assert (
        "expected.encoding/expected.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32"
        in load_parameters
    )
    assert "fa_loopback requires expected.layout=interleaved" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters
    assert 'declare_parameter<std::string>("input_topic")' in load_parameters
    assert 'declare_parameter<std::string>("output_topic")' in load_parameters
    assert 'declare_parameter<std::string>("input_stream_id")' in load_parameters
    assert 'declare_parameter<std::string>("output.stream_id")' in load_parameters
    forbidden_declare = 'declare_parameter("input_topic", '
    forbidden_declare += "config_"
    assert forbidden_declare not in load_parameters
    assert ("loopback." + "require_distinct_topics") not in load_parameters


def test_runtime_frame_validation_drops_invalid_frames_before_publish() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_loopback_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaLoopbackNode::validateFrame")[1].split(
        "void FaLoopbackNode::publishLoopback"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "AudioFrame layout mismatch" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "AudioFrame encoding mismatch" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "AudioFrame format mismatch" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "AudioFrame data size is invalid for configured interleaved samples" in validate_frame
    assert "return false;" in validate_frame


def test_loopback_copies_frame_and_only_changes_stream_id() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_loopback_node.cpp").read_text(encoding="utf-8")
    publish_loopback = source.split("void FaLoopbackNode::publishLoopback")[1].split(
        "size_t FaLoopbackNode::bytesPerFrame"
    )[0]

    assert "fa_interfaces::msg::AudioFrame out = msg;" in publish_loopback
    assert "out.stream_id = config_.output_stream_id;" in publish_loopback
    assert "audio_pub_->publish(out);" in publish_loopback
    assert "out.data" not in publish_loopback
    assert "out.sample_rate" not in publish_loopback
    assert "out.channels" not in publish_loopback
    assert "out.encoding" not in publish_loopback
    assert "out.bit_depth" not in publish_loopback
    assert "out.layout" not in publish_loopback
    assert "out.epoch" not in publish_loopback


def test_loopback_does_not_implement_audio_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_loopback_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "decode",
        "encodeFloat",
        "dbToLinear",
        "input_gains",
        "std::" + "clamp",
        "std::lround",
        "std::memcpy",
        "resample",
        "sample_format",
        "channel_convert",
        "normalize",
        "gain",
    )
    for token in forbidden:
        assert token not in source


def test_loopback_contract_forbidden_runtime_patterns_are_absent() -> None:
    package_root = Path(__file__).parents[2]
    relative_paths = (
        "include/fa_loopback/fa_loopback_node.hpp",
        "src/fa_loopback_node.cpp",
        "src/main.cpp",
        "launch/fa_loopback.launch.py",
    )
    forbidden_qos = "QoS("
    forbidden_qos += "std::max"
    forbidden_declare = 'declare_parameter("input_topic", '
    forbidden_declare += "config_"
    forbidden_tokens = (
        "SystemDefaults" + "QoS",
        "dict[str, " + "An" + "y]",
        "except " + "ImportError",
        "std::max(",
        forbidden_qos,
        "std::" + "clamp",
        forbidden_declare,
        "loopback." + "require_distinct_topics",
    )
    forbidden_whole_words = ("An" + "y", "ob" + "ject")

    for relative_path in relative_paths:
        text = (package_root / relative_path).read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text
        for token in forbidden_whole_words:
            assert f" {token} " not in text
            assert f"<{token}>" not in text
            assert f": {token}" not in text


def test_diagnostics_report_config_and_counters() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_loopback_node.cpp").read_text(encoding="utf-8")
    diagnostics = source.split("void FaLoopbackNode::publishDiagnostics")[1].split(
        "}  // namespace fa_loopback"
    )[0]
    header = (package_root / "include" / "fa_loopback" / "fa_loopback_node.hpp").read_text(
        encoding="utf-8"
    )

    assert "std::atomic<uint64_t> frames_in_" in header
    assert "std::atomic<uint64_t> frames_out_" in header
    assert "std::atomic<uint64_t> frames_dropped_" in header
    assert "input_topic" in diagnostics
    assert "output_topic" in diagnostics
    assert "input_stream_id" in diagnostics
    assert "output_stream_id" in diagnostics
    assert "expected.sample_rate" in diagnostics
    assert "expected.channels" in diagnostics
    assert "expected.encoding" in diagnostics
    assert "expected.bit_depth" in diagnostics
    assert "expected.layout" in diagnostics
    assert "qos.depth" in diagnostics
    assert "qos.reliable" in diagnostics
    assert "diagnostics.qos.depth" in diagnostics
    assert "diagnostics.qos.reliable" in diagnostics
    assert "frames_in" in diagnostics
    assert "frames_out" in diagnostics
    assert "frames_dropped" in diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/no_runtime_backend.md",
        "config/default.yaml",
        "launch/fa_loopback.launch.py",
        "include/fa_loopback/fa_loopback_node.hpp",
        "src/fa_loopback_node.cpp",
        "src/main.cpp",
        "test/unit/test_fa_loopback_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml

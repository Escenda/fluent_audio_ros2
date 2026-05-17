from pathlib import Path

import yaml


def test_default_config_defines_explicit_patchbay_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    launch_text = (package_root / "launch" / "fa_patchbay.launch.py").read_text(encoding="utf-8")
    params = config["fa_patchbay"]["ros__parameters"]

    assert params["input_topics"] == ["audio/input/frame"]
    assert params["output_topics"] == ["audio/output/frame"]
    assert len(params["input_topics"]) == len(params["output_topics"])
    route_pairs = list(zip(params["input_topics"], params["output_topics"]))
    assert len(route_pairs) == len(set(route_pairs))
    for input_topic, output_topic in route_pairs:
        assert input_topic
        assert output_topic
        assert input_topic != output_topic
    assert params["expected"]["sample_rate"] == 48000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is True
    assert params["diagnostics"]["publish_period_ms"] == 1000
    assert 'default_value="fa_patchbay"' in launch_text


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_patchbay_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaPatchbayNode::loadParameters")[1].split(
        "void FaPatchbayNode::setupInterfaces"
    )[0]

    assert "throw std::runtime_error(\"input_topics must contain at least one topic\")" in load_parameters
    assert "throw std::runtime_error(\"output_topics must contain at least one topic\")" in load_parameters
    assert (
        "throw std::runtime_error(\"input_topics and output_topics must have the same length\")"
        in load_parameters
    )
    assert "throw std::runtime_error(\"input_topics must not contain empty topic\")" in load_parameters
    assert "throw std::runtime_error(\"output_topics must not contain empty topic\")" in load_parameters
    assert "throw std::runtime_error(\"route output topic must not equal its input topic\")" in load_parameters
    assert "throw std::runtime_error(\"route pairs must be unique\")" in load_parameters
    assert "expected.sample_rate must be > 0" in load_parameters
    assert "expected.channels must be > 0" in load_parameters
    assert "expected.encoding is required" in load_parameters
    assert "expected.bit_depth must be > 0" in load_parameters
    assert "expected.bit_depth must be byte-aligned" in load_parameters
    assert (
        "expected.encoding/expected.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32"
        in load_parameters
    )
    assert "fa_patchbay requires expected.layout=interleaved" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters


def test_runtime_frame_validation_drops_invalid_frames_before_routing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_patchbay_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaPatchbayNode::validateFrame")[1].split(
        "void FaPatchbayNode::publishCopies"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_patchbay_copies_frame_and_only_changes_stream_id() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_patchbay_node.cpp").read_text(encoding="utf-8")
    publish_copies = source.split("void FaPatchbayNode::publishCopies")[1].split(
        "size_t FaPatchbayNode::bytesPerFrame"
    )[0]

    assert "fa_interfaces::msg::AudioFrame out = msg;" in publish_copies
    assert "out.stream_id = route.output_topic;" in publish_copies
    assert "route.publisher->publish(out);" in publish_copies
    assert "out.data" not in publish_copies
    assert "out.sample_rate" not in publish_copies
    assert "out.channels" not in publish_copies
    assert "out.encoding" not in publish_copies
    assert "out.bit_depth" not in publish_copies
    assert "out.layout" not in publish_copies
    assert "std::memcpy" not in publish_copies


def test_subscriptions_capture_input_topic_with_ros_callback_shape() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_patchbay_node.cpp").read_text(encoding="utf-8")
    header = (package_root / "include" / "fa_patchbay" / "fa_patchbay_node.hpp").read_text(
        encoding="utf-8"
    )
    setup_interfaces = source.split("void FaPatchbayNode::setupInterfaces")[1].split(
        "void FaPatchbayNode::handleFrame"
    )[0]

    assert "[this, input_topic](const fa_interfaces::msg::AudioFrame::SharedPtr msg)" in setup_interfaces
    assert "this->handleFrame(input_topic, msg);" in setup_interfaces
    assert "std::bind(&FaPatchbayNode::handleFrame" not in setup_interfaces
    assert "void handleFrame(const std::string & input_topic" in header


def test_patchbay_does_not_implement_audio_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_patchbay_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "decode",
        "encodeFloat",
        "dbToLinear",
        "input_gains",
        "std::clamp",
        "std::lround",
        "std::memcpy",
        "resample",
        "sample_format",
        "channel_convert",
        "mixSamples",
        "gain",
    )
    for token in forbidden:
        assert token not in source


def test_diagnostics_report_routing_state_and_counters() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_patchbay_node.cpp").read_text(encoding="utf-8")
    diagnostics = source.split("void FaPatchbayNode::publishDiagnostics")[1].split(
        "}  // namespace fa_patchbay"
    )[0]
    header = (package_root / "include" / "fa_patchbay" / "fa_patchbay_node.hpp").read_text(
        encoding="utf-8"
    )

    assert "std::atomic<uint64_t> frames_in_" in header
    assert "std::atomic<uint64_t> frames_dropped_" in header
    assert "std::atomic<uint64_t> copies_out_" in header
    assert "route_count" in diagnostics
    assert "unique_input_count" in diagnostics
    assert "input_topic." in diagnostics
    assert "output_topic." in diagnostics
    assert "expected.sample_rate" in diagnostics
    assert "expected.channels" in diagnostics
    assert "expected.encoding" in diagnostics
    assert "expected.bit_depth" in diagnostics
    assert "expected.layout" in diagnostics
    assert "frames_in" in diagnostics
    assert "frames_dropped" in diagnostics
    assert "copies_out" in diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/no_runtime_backend.md",
        "config/default.yaml",
        "launch/fa_patchbay.launch.py",
        "include/fa_patchbay/fa_patchbay_node.hpp",
        "src/fa_patchbay_node.cpp",
        "test/unit/test_fa_patchbay_audio_frame_contract.py",
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

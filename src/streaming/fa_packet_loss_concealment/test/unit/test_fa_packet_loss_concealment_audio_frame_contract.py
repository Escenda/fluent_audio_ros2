from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def source_text() -> str:
    return (package_root() / "src" / "fa_packet_loss_concealment_node.cpp").read_text(
        encoding="utf-8"
    )


def header_text() -> str:
    return (
        package_root()
        / "include"
        / "fa_packet_loss_concealment"
        / "fa_packet_loss_concealment_node.hpp"
    ).read_text(encoding="utf-8")


def test_default_config_declares_explicit_plc_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_packet_loss_concealment_node"]["ros__parameters"]

    assert params["input_topic"] == "audio/stream/input"
    assert params["output_topic"] == "audio/stream/plc"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["plc"]["max_gap_frames"] == 3
    assert params["plc"]["attenuation_per_gap"] == 0.7
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_node_identity_and_package_names_are_explicit() -> None:
    source = source_text()
    header = header_text()
    launch = (package_root() / "launch" / "fa_packet_loss_concealment.launch.py").read_text(
        encoding="utf-8"
    )

    assert "namespace fa_packet_loss_concealment" in header
    assert "class FaPacketLossConcealmentNode : public rclcpp::Node" in header
    assert ': rclcpp::Node("fa_packet_loss_concealment_node")' in source
    assert 'executable="fa_packet_loss_concealment_node"' in launch
    assert 'default_value="fa_packet_loss_concealment_node"' in launch


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = source_text()
    load_parameters = source.split(
        "void FaPacketLossConcealmentNode::loadParameters"
    )[1].split("void FaPacketLossConcealmentNode::setupInterfaces")[0]

    required_declarations = (
        'declare_parameter<std::string>("input_topic");',
        'declare_parameter<std::string>("output_topic");',
        'declare_parameter<int>("expected.sample_rate");',
        'declare_parameter<int>("expected.channels");',
        'declare_parameter<std::string>("expected.encoding");',
        'declare_parameter<int>("expected.bit_depth");',
        'declare_parameter<std::string>("expected.layout");',
        'declare_parameter<int>("plc.max_gap_frames");',
        'declare_parameter<double>("plc.attenuation_per_gap");',
        'declare_parameter<int>("qos.depth");',
        'declare_parameter<bool>("qos.reliable");',
        'declare_parameter<int>("diagnostics.publish_period_ms");',
    )
    for declaration in required_declarations:
        assert declaration in load_parameters

    assert "input_topic is required" in load_parameters
    assert "output_topic is required" in load_parameters
    assert "expected.sample_rate must be > 0" in load_parameters
    assert "expected.channels must be > 0" in load_parameters
    assert "requires expected.encoding=FLOAT32LE" in load_parameters
    assert "requires expected.bit_depth=32" in load_parameters
    assert "requires expected.layout=interleaved" in load_parameters
    assert "plc.max_gap_frames must be >= 0" in load_parameters
    assert "!std::isfinite(config_.attenuation_per_gap)" in load_parameters
    assert "config_.attenuation_per_gap <= 0.0" in load_parameters
    assert "config_.attenuation_per_gap > 1.0" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters
    assert 'declare_parameter<bool>("qos.reliable", config_.qos_reliable)' not in load_parameters


def test_runtime_validation_rejects_invalid_frames_before_state_mutation() -> None:
    source = source_text()
    validate_frame = source.split("bool FaPacketLossConcealmentNode::validateFrame")[1].split(
        "bool FaPacketLossConcealmentNode::sourceChanged"
    )[0]
    handle_frame = source.split("void FaPacketLossConcealmentNode::handleFrame")[1].split(
        "bool FaPacketLossConcealmentNode::validateFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerSampleFrame()) != 0U" in validate_frame
    assert "std::isfinite(sample)" in validate_frame
    assert "sample < -1.0F || sample > 1.0F" in validate_frame
    assert "if (!validateFrame(*msg))" in handle_frame
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index("sourceChanged(*msg)")
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index("updatePreviousFrame(*msg);")
    assert "frames_dropped_.fetch_add(1);" in handle_frame


def test_plc_synthesizes_missing_epochs_with_repeat_attenuation() -> None:
    source = source_text()
    handle_frame = source.split("void FaPacketLossConcealmentNode::handleFrame")[1].split(
        "bool FaPacketLossConcealmentNode::validateFrame"
    )[0]
    publish_concealed = source.split(
        "bool FaPacketLossConcealmentNode::publishConcealedFrames"
    )[1].split("bool FaPacketLossConcealmentNode::buildConcealedFrame")[0]
    build_frame = source.split("bool FaPacketLossConcealmentNode::buildConcealedFrame")[1].split(
        "bool FaPacketLossConcealmentNode::timestampForGap"
    )[0]

    assert "const uint32_t missing_frame_count = msg->epoch - previous_frame_.epoch - 1U;" in handle_frame
    assert "publishConcealedFrames(missing_frame_count)" in handle_frame
    assert "for (uint32_t gap_index = 1U; gap_index <= missing_frame_count; ++gap_index)" in publish_concealed
    assert "audio_pub_->publish(out);" in publish_concealed
    assert "concealed_frames_.fetch_add(1);" in publish_concealed
    assert "out.stream_id = config_.output_topic;" in build_frame
    assert "out.epoch = previous_frame_.epoch + gap_index;" in build_frame
    assert "out.data.resize(previous_frame_.data.size());" in build_frame
    assert "std::pow(config_.attenuation_per_gap, static_cast<double>(gap_index))" in build_frame
    assert "readFloat32LeSample(previous_frame_.data, offset)" in build_frame
    assert "writeFloat32LeSample(out.data, offset" in build_frame


def test_gap_larger_than_max_resets_and_publishes_current_only() -> None:
    source = source_text()
    handle_frame = source.split("void FaPacketLossConcealmentNode::handleFrame")[1].split(
        "bool FaPacketLossConcealmentNode::validateFrame"
    )[0]
    oversized_gap_branch = handle_frame.split(
        "if (missing_frame_count > static_cast<uint32_t>(config_.max_gap_frames))"
    )[1].split("} else if (!publishConcealedFrames(missing_frame_count))")[0]

    assert "PLC gap of %u frames exceeds plc.max_gap_frames" in oversized_gap_branch
    assert "resetPreviousFrame();" in oversized_gap_branch
    assert "gap_resets_.fetch_add(1);" in oversized_gap_branch
    assert "publishConcealedFrames" not in oversized_gap_branch
    assert "publishCurrentFrame(*msg);" in handle_frame
    assert handle_frame.index("resetPreviousFrame();") < handle_frame.rindex("publishCurrentFrame(*msg);")
    assert handle_frame.index("publishCurrentFrame(*msg);") < handle_frame.index("updatePreviousFrame(*msg);")


def test_duplicate_and_regressing_epochs_drop_without_baseline_update() -> None:
    source = source_text()
    handle_frame = source.split("void FaPacketLossConcealmentNode::handleFrame")[1].split(
        "bool FaPacketLossConcealmentNode::validateFrame"
    )[0]
    duplicate_branch = handle_frame.split("if (msg->epoch <= previous_frame_.epoch)")[1].split(
        "const uint32_t missing_frame_count"
    )[0]

    assert "Dropping duplicate or regressing AudioFrame epoch" in duplicate_branch
    assert "duplicate_drops_.fetch_add(1);" in duplicate_branch
    assert "frames_dropped_.fetch_add(1);" in duplicate_branch
    assert "return;" in duplicate_branch
    assert "updatePreviousFrame" not in duplicate_branch
    assert "publishCurrentFrame" not in duplicate_branch


def test_diagnostics_publish_config_and_required_counters() -> None:
    source = source_text()
    header = header_text()
    diagnostics = source.split("void FaPacketLossConcealmentNode::publishDiagnostics")[1].split(
        "}  // namespace fa_packet_loss_concealment"
    )[0]

    assert "std::atomic<uint64_t> frames_in_" in header
    assert "std::atomic<uint64_t> frames_out_" in header
    assert "std::atomic<uint64_t> frames_dropped_" in header
    assert "std::atomic<uint64_t> concealed_frames_" in header
    assert "std::atomic<uint64_t> duplicate_drops_" in header
    assert "std::atomic<uint64_t> gap_resets_" in header
    for key in (
        '"backend"',
        '"input_topic"',
        '"output_topic"',
        '"expected_sample_rate"',
        '"expected_channels"',
        '"expected_encoding"',
        '"expected_bit_depth"',
        '"expected_layout"',
        '"plc_max_gap_frames"',
        '"plc_attenuation_per_gap"',
        '"qos_depth"',
        '"qos_reliable"',
        '"diagnostics_publish_period_ms"',
        '"frames_in"',
        '"frames_out"',
        '"frames_dropped"',
        '"concealed_frames"',
        '"duplicate_drops"',
        '"gap_resets"',
    ):
        assert key in diagnostics


def test_streaming_node_has_no_device_io_resampling_padding_or_legacy_aliases() -> None:
    source = source_text()
    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "std::clamp",
        "legacy",
        "deprecated",
        "alias",
        "try { import",
    )
    for token in forbidden:
        assert token not in source


def test_package_layout_matches_required_streaming_layout() -> None:
    required_paths = (
        "CMakeLists.txt",
        "package.xml",
        "README.md",
        "config/default.yaml",
        "launch/fa_packet_loss_concealment.launch.py",
        "include/fa_packet_loss_concealment/fa_packet_loss_concealment_node.hpp",
        "src/fa_packet_loss_concealment_node.cpp",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/repeat_attenuation_plc.md",
        "test/unit/test_fa_packet_loss_concealment_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_contracts_and_lint_auto() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "find_package(ament_lint_auto REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "ament_lint_auto_find_test_dependencies()" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

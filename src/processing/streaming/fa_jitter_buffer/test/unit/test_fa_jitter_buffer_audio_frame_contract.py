from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def source_text() -> str:
    return (package_root() / "src" / "fa_jitter_buffer_node.cpp").read_text(encoding="utf-8")


def header_text() -> str:
    return (
        package_root() / "include" / "fa_jitter_buffer" / "fa_jitter_buffer_node.hpp"
    ).read_text(encoding="utf-8")


def test_default_config_declares_required_jitter_buffer_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_jitter_buffer_node"]["ros__parameters"]

    assert params["input_topic"] == "audio/network/mic"
    assert params["output_topic"] == "audio/jitter_buffered/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["jitter"]["target_depth_frames"] == 2
    assert params["jitter"]["max_depth_frames"] == 8
    assert params["jitter"]["reset_on_epoch_regression"] is False
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "CMakeLists.txt",
        "package.xml",
        "README.md",
        "config/default.yaml",
        "launch/fa_jitter_buffer.launch.py",
        "include/fa_jitter_buffer/fa_jitter_buffer_node.hpp",
        "src/fa_jitter_buffer_node.cpp",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_jitter_buffer.md",
        "test/unit/test_fa_jitter_buffer_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_node_identity_matches_contract() -> None:
    source = source_text()
    header = header_text()
    launch = (package_root() / "launch" / "fa_jitter_buffer.launch.py").read_text(encoding="utf-8")

    assert "namespace fa_jitter_buffer" in header
    assert "class FaJitterBufferNode : public rclcpp::Node" in header
    assert ': rclcpp::Node("fa_jitter_buffer_node")' in source
    assert "fa_jitter_buffer::FaJitterBufferNode" in source
    assert 'executable="fa_jitter_buffer_node"' in launch
    assert 'default_value="fa_jitter_buffer_node"' in launch


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = source_text()
    load_parameters = source.split("void FaJitterBufferNode::loadParameters")[1].split(
        "void FaJitterBufferNode::setupInterfaces"
    )[0]

    assert 'declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'declare_parameter<int>("expected.sample_rate");' in load_parameters
    assert 'declare_parameter<int>("jitter.target_depth_frames");' in load_parameters
    assert 'declare_parameter<int>("jitter.max_depth_frames");' in load_parameters
    assert 'declare_parameter<bool>("jitter.reset_on_epoch_regression");' in load_parameters
    assert 'declare_parameter<bool>("qos.reliable");' in load_parameters
    assert 'declare_parameter<bool>("qos.reliable", config_.qos_reliable)' not in load_parameters
    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "expected.sample_rate must be > 0" in load_parameters
    assert "expected.channels must be > 0" in load_parameters
    assert "expected.encoding is required" in load_parameters
    assert "expected.bit_depth must be > 0" in load_parameters
    assert "expected.bit_depth must be byte-aligned" in load_parameters
    assert "FLOAT32LE requires expected.bit_depth=32" in load_parameters
    assert "expected.layout is required" in load_parameters
    assert "jitter.target_depth_frames must be >= 0" in load_parameters
    assert "jitter.max_depth_frames must be > 0" in load_parameters
    assert "jitter.max_depth_frames must be > jitter.target_depth_frames" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters


def test_runtime_validation_rejects_invalid_frames_before_buffer_mutation() -> None:
    source = source_text()
    validate_frame = source.split("bool FaJitterBufferNode::validateFrame")[1].split(
        "bool FaJitterBufferNode::validateFloat32InterleavedSamples"
    )[0]
    handle_frame = source.split("void FaJitterBufferNode::handleFrame")[1].split(
        "bool FaJitterBufferNode::validateFrame"
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
        "hasDifferentContract(*msg)"
    )
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index("insertFrame(*msg);")
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert "return;" in handle_frame


def test_float32_interleaved_samples_are_finite_and_normalized() -> None:
    source = source_text()
    validate_samples = source.split("bool FaJitterBufferNode::validateFloat32InterleavedSamples")[1].split(
        "bool FaJitterBufferNode::hasDifferentContract"
    )[0]

    assert "msg.encoding == kEncodingFloat32 && msg.layout == kInterleavedLayout" in source
    assert "readFloat32LeSample(data, offset)" in validate_samples
    assert "std::isfinite(sample)" in validate_samples
    assert "sample < -1.0F || sample > 1.0F" in validate_samples
    assert "return false;" in validate_samples


def test_epoch_buffer_uses_ordered_map_and_publishes_oldest_over_target_depth() -> None:
    source = source_text()
    header = header_text()
    publish_ready = source.split("void FaJitterBufferNode::publishReadyFrames")[1].split(
        "float FaJitterBufferNode::readFloat32LeSample"
    )[0]

    assert "std::map<uint32_t, fa_interfaces::msg::AudioFrame> buffered_frames_" in header
    assert "buffered_frames_.emplace(msg.epoch, msg)" in source
    assert "buffered_frames_.size() > static_cast<size_t>(config_.target_depth_frames)" in publish_ready
    assert "auto oldest = buffered_frames_.begin();" in publish_ready
    assert "fa_interfaces::msg::AudioFrame out = oldest->second;" in publish_ready
    assert "out.stream_id = config_.output_topic;" in publish_ready
    assert "audio_pub_->publish(out);" in publish_ready
    assert "last_published_epoch_ = oldest->first;" in publish_ready
    assert "buffered_frames_.erase(oldest);" in publish_ready


def test_max_depth_is_enforced_before_buffer_growth() -> None:
    source = source_text()
    header = header_text()
    handle_frame = source.split("void FaJitterBufferNode::handleFrame")[1].split(
        "bool FaJitterBufferNode::validateFrame"
    )[0]

    assert "std::atomic<uint64_t> max_depth_resets_" in header
    assert "buffered_frames_.size() >= static_cast<size_t>(config_.max_depth_frames)" in handle_frame
    assert "resetBuffer();" in handle_frame
    assert "max_depth_resets_.fetch_add(1);" in handle_frame
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert handle_frame.index("config_.max_depth_frames") < handle_frame.index("insertFrame(*msg);")


def test_duplicate_and_late_epoch_handling_are_explicit() -> None:
    source = source_text()
    duplicate = source.split("bool FaJitterBufferNode::isDuplicateEpoch")[1].split(
        "bool FaJitterBufferNode::isLateEpoch"
    )[0]
    late = source.split("bool FaJitterBufferNode::isLateEpoch")[1].split(
        "void FaJitterBufferNode::activateStream"
    )[0]
    handle_frame = source.split("void FaJitterBufferNode::handleFrame")[1].split(
        "bool FaJitterBufferNode::validateFrame"
    )[0]

    assert "buffered_frames_.find(msg.epoch) != buffered_frames_.end()" in duplicate
    assert "msg.epoch == *last_published_epoch_" in duplicate
    assert "msg.epoch < *last_published_epoch_" in late
    assert "isDuplicateEpoch(*msg)" in handle_frame
    assert "duplicate_drops_.fetch_add(1);" in handle_frame
    assert "isLateEpoch(*msg)" in handle_frame
    assert "!config_.reset_on_epoch_regression" in handle_frame
    assert "late_drops_.fetch_add(1);" in handle_frame
    assert "resetBuffer();" in handle_frame


def test_source_and_format_contract_change_resets_buffer_without_mixing_streams() -> None:
    source = source_text()
    has_different = source.split("bool FaJitterBufferNode::hasDifferentContract")[1].split(
        "bool FaJitterBufferNode::isDuplicateEpoch"
    )[0]
    handle_frame = source.split("void FaJitterBufferNode::handleFrame")[1].split(
        "bool FaJitterBufferNode::validateFrame"
    )[0]

    assert "msg.source_id != active_stream_->source_id" in has_different
    assert "msg.stream_id != active_stream_->stream_id" in has_different
    assert "msg.sample_rate != active_stream_->sample_rate" in has_different
    assert "msg.channels != active_stream_->channels" in has_different
    assert "msg.encoding != active_stream_->encoding" in has_different
    assert "msg.bit_depth != active_stream_->bit_depth" in has_different
    assert "msg.layout != active_stream_->layout" in has_different
    assert "hasDifferentContract(*msg)" in handle_frame
    assert "resets_.fetch_add(1);" in handle_frame
    assert "activateStream(*msg);" in handle_frame


def test_processing_node_has_no_device_io_audio_editing_or_legacy_aliases() -> None:
    source = source_text()
    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "normalize(",
        "std::clamp",
        "padding",
        "invent",
        "device",
        "legacy",
        "deprecated",
        "alias",
    )
    for token in forbidden:
        assert token not in source


def test_diagnostics_publish_config_state_and_required_counters() -> None:
    source = source_text()
    header = header_text()
    diagnostics = source.split("void FaJitterBufferNode::publishDiagnostics")[1].split(
        "}  // namespace fa_jitter_buffer"
    )[0]

    assert "std::atomic<uint64_t> frames_in_" in header
    assert "std::atomic<uint64_t> frames_out_" in header
    assert "std::atomic<uint64_t> frames_dropped_" in header
    assert "std::atomic<uint64_t> duplicate_drops_" in header
    assert "std::atomic<uint64_t> late_drops_" in header
    assert "std::atomic<uint64_t> max_depth_resets_" in header
    assert "std::atomic<uint64_t> resets_" in header
    assert '"input_topic"' in diagnostics
    assert '"output_topic"' in diagnostics
    assert '"expected_sample_rate"' in diagnostics
    assert '"expected_channels"' in diagnostics
    assert '"expected_encoding"' in diagnostics
    assert '"expected_bit_depth"' in diagnostics
    assert '"expected_layout"' in diagnostics
    assert '"target_depth_frames"' in diagnostics
    assert '"max_depth_frames"' in diagnostics
    assert '"reset_on_epoch_regression"' in diagnostics
    assert '"buffered_frames"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"duplicate_drops"' in diagnostics
    assert '"late_drops"' in diagnostics
    assert '"max_depth_resets"' in diagnostics
    assert '"resets"' in diagnostics


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

from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def source_text() -> str:
    return (package_root() / "src" / "fa_overlap_add_node.cpp").read_text(encoding="utf-8")


def header_text() -> str:
    return (
        package_root()
        / "include"
        / "fa_overlap_add"
        / "fa_overlap_add_node.hpp"
    ).read_text(encoding="utf-8")


def test_default_config_declares_required_overlap_add_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_overlap_add_node"]["ros__parameters"]

    assert params["input_topic"] == "audio/chunked_overlap/mic"
    assert params["output_topic"] == "audio/overlap_added/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["window"]["frame_samples"] == 512
    assert params["window"]["hop_samples"] == 256
    assert params["window"]["type"] == "hann"
    assert params["overlap"]["max_buffered_chunks"] == 4
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is True
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_ros2_node_name_and_executable_match_required_contract() -> None:
    source = source_text()
    header = header_text()
    main_source = (package_root() / "src" / "main.cpp").read_text(encoding="utf-8")
    launch = (package_root() / "launch" / "fa_overlap_add.launch.py").read_text(encoding="utf-8")
    cmake = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "class FaOverlapAddNode : public rclcpp::Node" in header
    assert 'rclcpp::Node("fa_overlap_add_node", options)' in source
    assert "explicit FaOverlapAddNode(const rclcpp::NodeOptions & options" in header
    assert "fa_overlap_add::FaOverlapAddNode" in main_source
    assert 'default_value="fa_overlap_add_node"' in launch
    assert 'executable="fa_overlap_add_node"' in launch
    assert "add_executable(fa_overlap_add_node" in cmake


def test_overlap_add_does_not_include_device_io_conversion_clamp_or_tail_padding() -> None:
    source = source_text()
    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "convertPcm",
        "std::clamp",
        "legacy",
        "deprecated",
        "alias",
        "tail padding",
        "zero padding",
    )
    for token in forbidden:
        assert token not in source


def test_startup_validation_fails_closed_for_invalid_config_without_code_defaults() -> None:
    source = source_text()
    load_parameters = source.split("void FaOverlapAddNode::loadParameters")[1].split(
        "void FaOverlapAddNode::buildSynthesisWindow"
    )[0]

    required_declarations = (
        'declare_parameter<std::string>("input_topic");',
        'declare_parameter<std::string>("output_topic");',
        'declare_parameter<int>("expected.sample_rate");',
        'declare_parameter<int>("expected.channels");',
        'declare_parameter<std::string>("expected.encoding");',
        'declare_parameter<int>("expected.bit_depth");',
        'declare_parameter<std::string>("expected.layout");',
        'declare_parameter<int>("window.frame_samples");',
        'declare_parameter<int>("window.hop_samples");',
        'declare_parameter<std::string>("window.type");',
        'declare_parameter<int>("overlap.max_buffered_chunks");',
        'declare_parameter<int>("qos.depth");',
        'declare_parameter<bool>("qos.reliable");',
        'declare_parameter<int>("diagnostics.publish_period_ms");',
    )
    for declaration in required_declarations:
        assert declaration in load_parameters

    assert 'declare_parameter<bool>("qos.reliable", config_.qos_reliable)' not in load_parameters
    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "expected.sample_rate must be > 0" in load_parameters
    assert "expected.channels must be > 0" in load_parameters
    assert "fa_overlap_add requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_overlap_add requires expected.bit_depth=32" in load_parameters
    assert "fa_overlap_add requires expected.layout=interleaved" in load_parameters
    assert "window.frame_samples must be > 0" in load_parameters
    assert "window.hop_samples must be > 0" in load_parameters
    assert "window.hop_samples must be <= window.frame_samples" in load_parameters
    assert "window.type must be rectangular or hann" in load_parameters
    assert "overlap.max_buffered_chunks must be > 0" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters


def test_runtime_validation_rejects_invalid_frames_before_state_mutation() -> None:
    source = source_text()
    validate_frame = source.split("bool FaOverlapAddNode::validateFrame")[1].split(
        "bool FaOverlapAddNode::hasInputEpochRegression"
    )[0]
    handle_frame = source.split("void FaOverlapAddNode::handleFrame")[1].split(
        "bool FaOverlapAddNode::validateFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.data.size() != chunkBytes()" in validate_frame
    assert "std::isfinite(sample)" in validate_frame
    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in validate_frame
    assert "if (!validateFrame(*msg))" in handle_frame
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index("requiresStreamReset(*msg)")
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index("accumulateChunk(*msg);")
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert "return;" in handle_frame


def test_window_handling_is_explicit_and_rejects_unknown_types() -> None:
    source = source_text()
    build_window = source.split("void FaOverlapAddNode::buildSynthesisWindow")[1].split(
        "void FaOverlapAddNode::setupInterfaces"
    )[0]
    load_parameters = source.split("void FaOverlapAddNode::loadParameters")[1].split(
        "void FaOverlapAddNode::buildSynthesisWindow"
    )[0]

    assert 'config_.window_type != kWindowRectangular && config_.window_type != kWindowHann' in load_parameters
    assert "window.type must be rectangular or hann" in load_parameters
    assert "synthesis_window_.assign(static_cast<size_t>(config_.frame_samples), 1.0);" in build_window
    assert "config_.window_type == kWindowRectangular" in build_window
    assert "0.5 - (0.5 * std::cos(centered_phase))" in build_window
    assert "window.type=hann produced a non-positive synthesis weight" in build_window


def test_overlap_add_accumulates_weighted_samples_and_publishes_only_safe_hops() -> None:
    source = source_text()
    accumulate = source.split("void FaOverlapAddNode::accumulateChunk")[1].split(
        "void FaOverlapAddNode::publishAvailableFrames"
    )[0]
    publish = source.split("void FaOverlapAddNode::publishAvailableFrames")[1].split(
        "bool FaOverlapAddNode::buildOutputFrame"
    )[0]
    build_output = source.split("bool FaOverlapAddNode::buildOutputFrame")[1].split(
        "bool FaOverlapAddNode::advanceNextOutputStamp"
    )[0]
    consume = source.split("void FaOverlapAddNode::consumePublishedHop")[1].split(
        "void FaOverlapAddNode::resetOverlapState"
    )[0]

    assert "const size_t chunk_start = next_chunk_start_sample_frames_;" in accumulate
    assert "sample_sums_.resize(required_samples, 0.0);" in accumulate
    assert "weight_sums_.resize(required_sample_frames, 0.0);" in accumulate
    assert "sample_sums_[output_sample_offset + channel] += static_cast<double>(sample) * weight;" in accumulate
    assert "weight_sums_[chunk_start + frame_index] += weight;" in accumulate
    assert "next_chunk_start_sample_frames_ += static_cast<size_t>(config_.hop_samples);" in accumulate
    assert "next_chunk_start_sample_frames_ >= hop_sample_frames" in publish
    assert "buildOutputFrame(out)" in publish
    assert "sample_sums_[sample_index] / weight" in build_output
    assert "out.data.resize(hop_sample_frames * channels * sizeof(float));" in build_output
    assert "audio_pub_->publish(out);" in publish
    assert "consumePublishedHop();" in publish
    assert "sample_sums_.erase(" in consume
    assert "weight_sums_.erase(" in consume
    assert "next_chunk_start_sample_frames_ -= hop_sample_frames;" in consume


def test_output_contract_drops_and_resets_instead_of_clamping_or_inventing_tail() -> None:
    source = source_text()
    publish = source.split("void FaOverlapAddNode::publishAvailableFrames")[1].split(
        "bool FaOverlapAddNode::buildOutputFrame"
    )[0]
    build_output = source.split("bool FaOverlapAddNode::buildOutputFrame")[1].split(
        "bool FaOverlapAddNode::advanceNextOutputStamp"
    )[0]

    assert "std::clamp" not in build_output
    assert "weight <= kMinNormalizationWeight" in build_output
    assert "!std::isfinite(normalized)" in build_output
    assert "normalized < static_cast<double>(kMinNormalizedSample)" in build_output
    assert "normalized > static_cast<double>(kMaxNormalizedSample)" in build_output
    assert "return false;" in build_output
    assert "resetOverlapState();" in publish
    assert "frames_dropped_.fetch_add(1);" in publish
    assert "resets_.fetch_add(1);" in publish
    assert "out.data.resize(hop_sample_frames * channels * sizeof(float));" in build_output
    assert "chunkBytes()" not in build_output


def test_duplicate_or_regressing_input_epoch_drops_without_replaying_stale_audio() -> None:
    source = source_text()
    header = header_text()
    handle_frame = source.split("void FaOverlapAddNode::handleFrame")[1].split(
        "bool FaOverlapAddNode::validateFrame"
    )[0]
    regression_check = source.split("bool FaOverlapAddNode::hasInputEpochRegression")[1].split(
        "bool FaOverlapAddNode::requiresStreamReset"
    )[0]

    assert "bool hasInputEpochRegression(const fa_interfaces::msg::AudioFrame & msg) const;" in header
    assert "std::atomic<uint64_t> epoch_regression_drops_" in header
    assert "hasInputEpochRegression(*msg)" in handle_frame
    assert "epoch_regression_drops_.fetch_add(1);" in handle_frame
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert handle_frame.index("hasInputEpochRegression(*msg)") < handle_frame.index(
        "requiresStreamReset(*msg)"
    )
    assert "msg.source_id != active_stream_->source_id || hasFormatChange(msg)" in regression_check
    assert "msg.epoch < next_expected_input_epoch_.value()" in regression_check


def test_source_format_and_future_epoch_gap_reset_state_without_resetting_output_epoch() -> None:
    source = source_text()
    header = header_text()
    requires_reset = source.split("bool FaOverlapAddNode::requiresStreamReset")[1].split(
        "bool FaOverlapAddNode::hasFormatChange"
    )[0]
    has_format_change = source.split("bool FaOverlapAddNode::hasFormatChange")[1].split(
        "void FaOverlapAddNode::activateStream"
    )[0]
    reset_state = source.split("void FaOverlapAddNode::resetOverlapState")[1].split(
        "float FaOverlapAddNode::readFloat32LeSample"
    )[0]
    build_output = source.split("bool FaOverlapAddNode::buildOutputFrame")[1].split(
        "bool FaOverlapAddNode::advanceNextOutputStamp"
    )[0]

    assert "std::optional<uint32_t> next_expected_input_epoch_" in header
    assert "msg.source_id != active_stream_->source_id" in requires_reset
    assert "hasFormatChange(msg)" in requires_reset
    assert "msg.epoch > next_expected_input_epoch_.value()" in requires_reset
    assert "msg.sample_rate != active_stream_->sample_rate" in has_format_change
    assert "msg.channels != active_stream_->channels" in has_format_change
    assert "msg.encoding != active_stream_->encoding" in has_format_change
    assert "msg.bit_depth != active_stream_->bit_depth" in has_format_change
    assert "msg.layout != active_stream_->layout" in has_format_change
    assert "next_expected_input_epoch_ = msg.epoch + 1U;" in source
    assert "next_output_epoch_ = 0" not in reset_state
    assert "out.source_id = active_stream_->source_id;" in build_output
    assert "out.stream_id = config_.output_topic;" in build_output
    assert "out.epoch = next_output_epoch_;" in build_output


def test_diagnostics_publish_config_counters_and_backend_identity() -> None:
    source = source_text()
    header = header_text()
    diagnostics = source.split("void FaOverlapAddNode::publishDiagnostics")[1].split(
        "}  // namespace fa_overlap_add"
    )[0]

    assert "std::atomic<uint64_t> frames_in_" in header
    assert "std::atomic<uint64_t> frames_out_" in header
    assert "std::atomic<uint64_t> frames_dropped_" in header
    assert "std::atomic<uint64_t> epoch_regression_drops_" in header
    assert "std::atomic<uint64_t> chunks_accumulated_" in header
    assert "std::atomic<uint64_t> resets_" in header
    assert "std::atomic<uint64_t> buffered_sample_frames_" in header
    for key in (
        "input_topic",
        "output_topic",
        "expected_sample_rate",
        "expected_channels",
        "expected_encoding",
        "expected_bit_depth",
        "expected_layout",
        "frame_samples",
        "hop_samples",
        "window_type",
        "max_buffered_chunks",
        "buffered_sample_frames",
        "frames_in",
        "frames_out",
        "frames_dropped",
        "epoch_regression_drops",
        "chunks_accumulated",
        "resets",
    ):
        assert key in diagnostics
    assert '"backend.name", "internal_overlap_add"' in diagnostics


def test_package_layout_matches_required_streaming_layout() -> None:
    required_paths = (
        "CMakeLists.txt",
        "package.xml",
        "README.md",
        "config/default.yaml",
        "launch/fa_overlap_add.launch.py",
        "include/fa_overlap_add/fa_overlap_add_node.hpp",
        "src/fa_overlap_add_node.cpp",
        "src/main.cpp",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_overlap_add.md",
        "test/unit/test_fa_overlap_add_audio_frame_contract.py",
        "test/cpp/test_fa_overlap_add_node_contract.cpp",
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
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_lint_auto REQUIRED)" in cmake_text
    assert "add_library(fa_overlap_add_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_node_contract_test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "ament_lint_auto_find_test_dependencies()" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

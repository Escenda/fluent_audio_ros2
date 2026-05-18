from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def source_text() -> str:
    return (package_root() / "src" / "fa_chunk_overlap_node.cpp").read_text(encoding="utf-8")


def header_text() -> str:
    return (
        package_root() / "include" / "fa_chunk_overlap" / "fa_chunk_overlap_node.hpp"
    ).read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_overlap_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_chunk_overlap"]["ros__parameters"]

    assert params["input_topic"] == "audio/float32le/mic"
    assert params["output_topic"] == "audio/chunked_overlap/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["window"]["frame_samples"] == 512
    assert params["window"]["hop_samples"] == 256
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is True
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_chunk_overlap_does_not_include_device_io_conversion_or_padding() -> None:
    source = source_text()
    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "convert",
        "std::clamp",
        "padding",
        "pad",
        "zero",
        "legacy",
    )
    for token in forbidden:
        assert token not in source


def test_node_identity_matches_contract() -> None:
    source = source_text()
    header = header_text()
    main_source = (package_root() / "src" / "main.cpp").read_text(encoding="utf-8")
    launch = (package_root() / "launch" / "fa_chunk_overlap.launch.py").read_text(
        encoding="utf-8"
    )

    assert "namespace fa_chunk_overlap" in header
    assert "class FaChunkOverlapNode : public rclcpp::Node" in header
    assert 'rclcpp::Node("fa_chunk_overlap", options)' in source
    assert "explicit FaChunkOverlapNode(const rclcpp::NodeOptions & options" in header
    assert "fa_chunk_overlap::FaChunkOverlapNode" in main_source
    assert 'executable="fa_chunk_overlap_node"' in launch
    assert "default_value" not in launch
    assert "FindPackageShare" not in launch


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = source_text()
    load_parameters = source.split("void FaChunkOverlapNode::loadParameters")[1].split(
        "void FaChunkOverlapNode::setupInterfaces"
    )[0]

    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "expected.sample_rate must be > 0" in load_parameters
    assert "expected.channels must be > 0" in load_parameters
    assert "fa_chunk_overlap requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_chunk_overlap requires expected.bit_depth=32" in load_parameters
    assert "fa_chunk_overlap requires expected.layout=interleaved" in load_parameters
    assert "window.frame_samples must be > 0" in load_parameters
    assert "window.hop_samples must be > 0" in load_parameters
    assert "window.hop_samples must be <= window.frame_samples" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters
    assert 'declare_parameter<int>("window.frame_samples");' in load_parameters
    assert 'declare_parameter<int>("window.hop_samples");' in load_parameters
    assert 'declare_parameter<bool>("qos.reliable");' in load_parameters
    assert 'declare_parameter<bool>("qos.reliable", config_.qos_reliable)' not in load_parameters


def test_runtime_validation_rejects_invalid_frames_before_buffer_mutation() -> None:
    source = source_text()
    validate_frame = source.split("bool FaChunkOverlapNode::validateFrame")[1].split(
        "bool FaChunkOverlapNode::hasDifferentSource"
    )[0]
    handle_frame = source.split("void FaChunkOverlapNode::handleFrame")[1].split(
        "bool FaChunkOverlapNode::validateFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerSampleFrame()) != 0" in validate_frame
    assert "std::isfinite(sample)" in validate_frame
    assert "sample < -1.0F || sample > 1.0F" in validate_frame
    assert "if (!validateFrame(*msg))" in handle_frame
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index("appendFrame(*msg);")
    assert "input_frames_dropped_.fetch_add(1);" in handle_frame
    assert "return;" in handle_frame


def test_source_change_resets_buffer_without_mixing_sources() -> None:
    source = source_text()
    handle_frame = source.split("void FaChunkOverlapNode::handleFrame")[1].split(
        "bool FaChunkOverlapNode::validateFrame"
    )[0]
    source_check = source.split("bool FaChunkOverlapNode::hasDifferentSource")[1].split(
        "void FaChunkOverlapNode::activateStream"
    )[0]

    assert "hasDifferentSource(*msg)" in handle_frame
    assert "resetActiveBuffer();" in handle_frame
    assert "source_resets_.fetch_add(1);" in handle_frame
    assert "activateStream(*msg);" in handle_frame
    assert "msg.source_id != active_stream_->source_id" in source_check


def test_overlap_algorithm_publishes_window_then_consumes_hop_only() -> None:
    source = source_text()
    publish_chunks = source.split("void FaChunkOverlapNode::publishAvailableChunks")[1].split(
        "void FaChunkOverlapNode::resetActiveBuffer"
    )[0]

    assert "const size_t bytes_per_window = windowBytes();" in publish_chunks
    assert "const size_t bytes_per_hop = hopBytes();" in publish_chunks
    assert "!buffered_segments_.empty()" in publish_chunks
    assert "buffer_.size() >= bytes_per_window" in publish_chunks
    assert "out.header = buffered_segments_.front().header;" in publish_chunks
    assert "out.stream_id = config_.output_topic;" in publish_chunks
    assert "out.epoch = next_output_epoch_;" in publish_chunks
    assert "next_output_epoch_ += 1U;" in publish_chunks
    assert "out.data.assign(" in publish_chunks
    assert "bytes_per_window" in publish_chunks
    assert "buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(bytes_per_hop));" in publish_chunks
    assert "if (!consumeBufferedBytes(bytes_per_hop))" in publish_chunks
    assert "bytes_per_window));" in publish_chunks


def test_partial_hop_consumption_advances_segment_timestamp() -> None:
    source = source_text()
    advance_header = source.split("bool FaChunkOverlapNode::advanceSegmentHeader")[1].split(
        "bool FaChunkOverlapNode::consumeBufferedBytes"
    )[0]
    consume_bytes = source.split("bool FaChunkOverlapNode::consumeBufferedBytes")[1].split(
        "float FaChunkOverlapNode::readFloat32LeSample"
    )[0]

    assert "consumed_bytes / bytes_per_sample_frame" in advance_header
    assert "config_.expected_sample_rate" in advance_header
    assert "stampToNanoseconds(segment.header.stamp)" in advance_header
    assert "advance_ns <= 0" in advance_header
    assert "segment.header.stamp = nanosecondsToStamp(advanced_ns);" in advance_header
    assert "advanceSegmentHeader(segment, remaining)" in consume_bytes
    assert consume_bytes.index("advanceSegmentHeader(segment, remaining)") < consume_bytes.index(
        "segment.byte_count -= remaining;"
    )


def test_diagnostics_publish_overlap_counters() -> None:
    source = source_text()
    header = (package_root() / "include" / "fa_chunk_overlap" / "fa_chunk_overlap_node.hpp").read_text(
        encoding="utf-8"
    )
    diagnostics = source.split("void FaChunkOverlapNode::publishDiagnostics")[1].split(
        "}  // namespace fa_chunk_overlap"
    )[0]

    assert "std::atomic<uint64_t> input_frames_dropped_" in header
    assert "std::atomic<uint64_t> source_resets_" in header
    assert "frame_samples" in diagnostics
    assert "hop_samples" in diagnostics
    assert "overlap_samples" in diagnostics
    assert "buffered_sample_frames" in diagnostics
    assert "input_frames_dropped" in diagnostics
    assert "chunks_out" in diagnostics
    assert "source_resets" in diagnostics


def test_package_layout_matches_required_streaming_layout() -> None:
    required_paths = (
        "CMakeLists.txt",
        "package.xml",
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_overlap_buffer.md",
        "config/default.yaml",
        "launch/fa_chunk_overlap.launch.py",
        "include/fa_chunk_overlap/fa_chunk_overlap_node.hpp",
        "src/fa_chunk_overlap_node.cpp",
        "src/main.cpp",
        "test/unit/test_fa_chunk_overlap_audio_frame_contract.py",
        "test/cpp/test_fa_chunk_overlap_node_contract.cpp",
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
    assert "add_library(fa_chunk_overlap_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_node_contract_test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "ament_lint_auto_find_test_dependencies()" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

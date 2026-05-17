from pathlib import Path

import yaml


def test_default_config_requires_float32_interleaved_fixed_chunk_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_frame_buffer"]["ros__parameters"]

    assert params["input_topic"] == "audio/noise_gated/mic"
    assert params["output_topic"] == "audio/buffered/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["buffering"]["frames_per_chunk"] == 512
    assert params["buffering"]["max_buffered_chunks"] == 4
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_frame_buffer_does_not_hide_io_or_other_processing_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_frame_buffer_node.cpp").read_text(encoding="utf-8")

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
        "isfinite",
    )
    for token in forbidden:
        assert token not in source


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_frame_buffer_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaFrameBufferNode::loadParameters")[1].split(
        "void FaFrameBufferNode::setupInterfaces"
    )[0]

    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "expected.sample_rate must be > 0" in load_parameters
    assert "expected.channels must be > 0" in load_parameters
    assert "fa_frame_buffer requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_frame_buffer requires expected.bit_depth=32" in load_parameters
    assert "fa_frame_buffer requires expected.layout=interleaved" in load_parameters
    assert "buffering.frames_per_chunk must be > 0" in load_parameters
    assert "buffering.max_buffered_chunks must be > 0" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters


def test_frame_buffer_validates_frame_contract_before_buffering() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_frame_buffer_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaFrameBufferNode::validateFrame")[1].split(
        "bool FaFrameBufferNode::isCompatibleWithBufferedStream"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_stream_compatibility_change_clears_buffer_before_append() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_frame_buffer_node.cpp").read_text(encoding="utf-8")
    handle_frame = source.split("void FaFrameBufferNode::handleFrame")[1].split(
        "bool FaFrameBufferNode::validateFrame"
    )[0]
    compatibility = source.split("bool FaFrameBufferNode::isCompatibleWithBufferedStream")[1].split(
        "void FaFrameBufferNode::appendFrame"
    )[0]

    assert "if (!isCompatibleWithBufferedStream(*msg))" in handle_frame
    assert "clearBufferedStream();" in handle_frame
    assert "buffer_resets_.fetch_add(1);" in handle_frame
    assert "const BufferedFrameIdentity & identity = buffered_segments_.front().identity;" in compatibility
    assert "msg.source_id == identity.source_id" in compatibility
    assert "msg.sample_rate == identity.sample_rate" in compatibility
    assert "msg.channels == identity.channels" in compatibility
    assert "msg.encoding == identity.encoding" in compatibility
    assert "msg.bit_depth == identity.bit_depth" in compatibility
    assert "msg.layout == identity.layout" in compatibility


def test_fixed_chunk_algorithm_preserves_first_contributing_frame_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_frame_buffer_node.cpp").read_text(encoding="utf-8")
    append_frame = source.split("void FaFrameBufferNode::appendFrame")[1].split(
        "void FaFrameBufferNode::publishAvailableChunks"
    )[0]
    publish_chunks = source.split("void FaFrameBufferNode::publishAvailableChunks")[1].split(
        "void FaFrameBufferNode::dropOldestChunkForOverflow"
    )[0]

    assert "buffered_segments_.push_back(BufferedSegment{identityFromFrame(msg), msg.data.size()});" in append_frame
    assert "buffer_.insert(buffer_.end(), msg.data.begin(), msg.data.end());" in append_frame
    assert "while (buffer_.size() > maxBufferedBytes())" in append_frame
    assert "while (buffer_.size() >= bytes_per_chunk)" in publish_chunks
    assert "const BufferedFrameIdentity & identity = buffered_segments_.front().identity;" in publish_chunks
    assert "out.header = identity.header;" in publish_chunks
    assert "out.source_id = identity.source_id;" in publish_chunks
    assert "out.stream_id = config_.output_topic;" in publish_chunks
    assert "out.epoch = identity.epoch;" in publish_chunks
    assert "out.data.assign(buffer_.begin()" in publish_chunks
    assert "buffer_.erase(buffer_.begin()" in publish_chunks
    assert "consumeBufferedBytes(bytes_per_chunk);" in publish_chunks
    assert "pad" not in publish_chunks


def test_overflow_drops_oldest_whole_chunk_and_reports_diagnostics() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_frame_buffer_node.cpp").read_text(encoding="utf-8")
    header = (package_root / "include" / "fa_frame_buffer" / "fa_frame_buffer_node.hpp").read_text(
        encoding="utf-8"
    )
    overflow = source.split("void FaFrameBufferNode::dropOldestChunkForOverflow")[1].split(
        "void FaFrameBufferNode::clearBufferedStream"
    )[0]
    diagnostics = source.split("void FaFrameBufferNode::publishDiagnostics")[1].split(
        "}  // namespace fa_frame_buffer"
    )[0]

    assert "std::atomic<uint64_t> overflow_count_" in header
    assert "const size_t bytes_to_drop = std::min(chunkBytes(), buffer_.size());" in overflow
    assert "buffer_.erase(buffer_.begin()" in overflow
    assert "overflow_count_.fetch_add(1);" in overflow
    assert "Frame buffer overflow: dropped oldest" in overflow
    assert "frames_per_chunk" in diagnostics
    assert "max_buffered_chunks" in diagnostics
    assert "chunk_bytes" in diagnostics
    assert "buffered_bytes" in diagnostics
    assert "overflow_count" in diagnostics
    assert "buffer_resets" in diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_fixed_chunk_buffer.md",
        "config/default.yaml",
        "launch/fa_frame_buffer.launch.py",
        "include/fa_frame_buffer/fa_frame_buffer_node.hpp",
        "src/fa_frame_buffer_node.cpp",
        "test/unit/test_fa_frame_buffer_audio_frame_contract.py",
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

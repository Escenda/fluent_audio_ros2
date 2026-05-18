from pathlib import Path

import yaml


def test_default_config_requires_explicit_float32le_layout_conversion_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_interleave"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"
    assert params["output_topic"] == "audio/layout/mic"
    assert params["input"]["layout"] == "interleaved"
    assert params["output"]["layout"] == "planar"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 2
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_interleave_does_not_hide_unrelated_processing_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_interleave_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "gain.linear",
        "threshold.linear",
        "filter.",
        "normalize(",
        "std::clamp",
        "readFloat32Le",
        "appendFloat32Le",
        "static_cast<float>",
        "static_cast<int32_t>",
        "static_cast<int64_t>",
    )
    for token in forbidden:
        assert token not in source


def test_startup_rejects_unsupported_or_implicit_layout_conversion_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_interleave_node.cpp").read_text(encoding="utf-8")
    launch_text = (package_root / "launch" / "fa_interleave.launch.py").read_text(
        encoding="utf-8"
    )
    load_parameters = source.split("void FaInterleaveNode::loadParameters")[1].split(
        "void FaInterleaveNode::setupInterfaces"
    )[0]
    backend_source = (
        package_root / "src" / "backends" / "internal_layout_reorder.cpp"
    ).read_text(encoding="utf-8")
    layout_support = backend_source.split("bool isSupportedLayoutConversion")[1].split(
        "bool isSupportedFormat"
    )[0]
    format_support = backend_source.split("bool isSupportedFormat")[1].split(
        "size_t bytesPerSample"
    )[0]

    assert "layout == kInterleavedLayout || layout == kPlanarLayout" in backend_source
    assert "input_layout != output_layout" in layout_support
    assert "isSupportedLayoutConversion(config_.input_layout, config_.output_layout)" in backend_source
    assert "encoding == kEncodingFloat32Le && bit_depth == 32" in format_support
    assert "encoding == kEncodingPcm16Le && bit_depth == 16" in format_support
    assert "encoding == kEncodingPcm32Le && bit_depth == 32" in format_support
    assert "std::make_unique<backends::InternalLayoutReorderBackend>" in source
    assert "throw std::runtime_error(" in backend_source
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'readRequiredString(*this, "input_topic")' in load_parameters
    assert 'readRequiredString(*this, "input.layout")' in load_parameters
    assert 'readRequiredInt(*this, "expected.sample_rate")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line


def test_interleave_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_interleave_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaInterleaveNode::validateFrame")[1].split(
        "bool FaInterleaveNode::convertFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "backend_->validateContract(frameContractFrom(msg))" in validate_frame
    assert "FrameContractStatus::kOk" in validate_frame
    assert "frameContractStatusName(contract_status)" in validate_frame
    assert "RCLCPP_WARN_THROTTLE" in validate_frame
    assert "return false;" in validate_frame

    backend_source = (
        package_root / "src" / "backends" / "internal_layout_reorder.cpp"
    ).read_text(encoding="utf-8")
    assert "validateContract(" in backend_source
    assert "const FrameContract & contract" in backend_source
    assert "contract.layout != config_.input_layout" in backend_source
    assert "contract.encoding != config_.expected_encoding" in backend_source
    assert "contract.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in backend_source
    assert "contract.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in backend_source
    assert "contract.channels != static_cast<uint32_t>(config_.expected_channels)" in backend_source
    assert "contract.data_size == 0" in backend_source
    assert "contract.data_size % bytes_per_frame" in backend_source


def test_interleave_preserves_metadata_and_updates_stream_layout_data_only() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_interleave_node.cpp").read_text(encoding="utf-8")
    convert_frame = source.split("bool FaInterleaveNode::convertFrame")[1].split(
        "void FaInterleaveNode::publishDiagnostics"
    )[0]

    assert "backend_->process(in.data, frameContractFrom(in), output_data)" in convert_frame
    assert "out = in;" in convert_frame
    assert "out.stream_id = config_.output_topic;" in convert_frame
    assert "out.layout = backend_->outputLayout();" in convert_frame
    assert "out.data = output_data;" in convert_frame
    assert "out.encoding =" not in convert_frame
    assert "out.bit_depth =" not in convert_frame
    assert "out.sample_rate =" not in convert_frame
    assert "out.channels =" not in convert_frame
    assert ".rms" not in convert_frame
    assert ".peak" not in convert_frame
    assert ".vad" not in convert_frame


def test_byte_reorder_algorithms_are_explicit_for_both_directions() -> None:
    package_root = Path(__file__).parents[2]
    source = (
        package_root / "src" / "backends" / "internal_layout_reorder.cpp"
    ).read_text(encoding="utf-8")
    to_planar = source.split("std::vector<uint8_t> reorderInterleavedToPlanar")[1].split(
        "std::vector<uint8_t> reorderPlanarToInterleaved"
    )[0]
    to_interleaved = source.split("std::vector<uint8_t> reorderPlanarToInterleaved")[1].split(
        "InternalLayoutReorderBackend::InternalLayoutReorderBackend"
    )[0]
    append_sample = source.split("void appendSampleBytes")[1].split(
        "std::vector<uint8_t> reorderInterleavedToPlanar"
    )[0]

    assert "for (size_t channel_index = 0; channel_index < channel_count; ++channel_index)" in to_planar
    assert "for (size_t frame_index = 0; frame_index < frame_count; ++frame_index)" in to_planar
    assert "const size_t input_sample_index = frame_index * channel_count + channel_index;" in to_planar
    assert "for (size_t frame_index = 0; frame_index < frame_count; ++frame_index)" in to_interleaved
    assert "for (size_t channel_index = 0; channel_index < channel_count; ++channel_index)" in to_interleaved
    assert "const size_t input_sample_index = channel_index * frame_count + frame_index;" in to_interleaved
    assert "const size_t byte_offset = sample_index * bytes_per_sample;" in append_sample
    assert "output_bytes.push_back(input_bytes.at(byte_offset + byte_index));" in append_sample


def test_supported_sample_formats_map_to_byte_width_without_numeric_conversion() -> None:
    package_root = Path(__file__).parents[2]
    source = (
        package_root / "src" / "backends" / "internal_layout_reorder.cpp"
    ).read_text(encoding="utf-8")
    bytes_per_sample = source.split("size_t bytesPerSample")[1].split(
        "void appendSampleBytes"
    )[0]

    assert "encoding == kEncodingPcm16Le && bit_depth == 16" in bytes_per_sample
    assert "return sizeof(uint16_t);" in bytes_per_sample
    assert "encoding == kEncodingPcm32Le && bit_depth == 32" in bytes_per_sample
    assert "encoding == kEncodingFloat32Le && bit_depth == 32" in bytes_per_sample
    assert "return sizeof(uint32_t);" in bytes_per_sample
    assert "std::memcpy" not in source


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_layout_reorder.md",
        "config/default.yaml",
        "launch/fa_interleave.launch.py",
        "include/fa_interleave/fa_interleave_node.hpp",
        "include/fa_interleave/backends/internal_layout_reorder.hpp",
        "src/fa_interleave_node.cpp",
        "src/backends/internal_layout_reorder.cpp",
        "test/unit",
        "test/cpp/test_internal_layout_reorder_backend.cpp",
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

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "add_library(fa_interleave_internal_layout_reorder" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

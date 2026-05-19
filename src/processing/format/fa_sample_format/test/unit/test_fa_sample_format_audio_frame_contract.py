from pathlib import Path

import yaml


def test_default_config_requires_explicit_pcm16_to_float32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_sample_format"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"
    assert params["output_topic"] == "audio/sample_format/mic"
    assert params["input_stream_id"] == "audio/raw/mic"
    assert params["output"]["stream_id"] == "audio/float32/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input"]["encoding"] == "PCM16LE"
    assert params["input"]["bit_depth"] == 16
    assert params["output"]["encoding"] == "FLOAT32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["layout"] == "interleaved"


def test_sample_format_does_not_hide_io_or_other_processing_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_sample_format_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "gain.linear",
        "threshold.linear",
        "filter.",
        "set_channels",
        "normalize(",
        "decode",
        "encode",
        "std::clamp",
    )
    for token in forbidden:
        assert token not in source


def test_startup_rejects_unknown_or_implicit_conversion_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_sample_format_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaSampleFormatNode::loadParameters")[1].split(
        "void FaSampleFormatNode::setupInterfaces"
    )[0]
    conversion = (
        package_root / "src" / "backends" / "internal_float32le.cpp"
    ).read_text(encoding="utf-8")
    is_supported = conversion.split("bool isSupportedSampleFormatConversion")[1].split(
        "size_t bytesPerSample"
    )[0]

    assert "output_encoding == kEncodingFloat32Le && output_bit_depth == 32" in is_supported
    assert "input_encoding == kEncodingPcm16Le && input_bit_depth == 16" in is_supported
    assert "input_encoding == kEncodingPcm32Le && input_bit_depth == 32" in is_supported
    assert "input_encoding == kEncodingFloat32Le && input_bit_depth == 32" in is_supported
    assert "output_encoding == kEncodingPcm16Le && output_bit_depth == 16" in is_supported
    assert "isSupportedSampleFormatConversion(" in conversion
    assert "throw std::runtime_error(" in conversion
    assert "FLOAT32LE/32 to PCM16LE/16" in conversion
    assert "std::make_unique<backends::InternalFloat32LeBackend>" in source
    assert 'readRequiredString(*this, "input_topic")' in load_parameters
    assert 'readRequiredString(*this, "input_stream_id")' in load_parameters
    assert 'readRequiredString(*this, "output.stream_id")' in load_parameters
    assert 'readRequiredString(*this, "input.encoding")' in load_parameters
    assert 'readRequiredInt(*this, "expected.sample_rate")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    assert 'readRequiredInt(*this, "diagnostics.qos.depth")' in load_parameters
    assert 'readRequiredBool(*this, "diagnostics.qos.reliable")' in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "metadata" not in is_supported


def test_sample_format_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_sample_format_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaSampleFormatNode::validateFrame")[1].split(
        "bool FaSampleFormatNode::convertFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "backend_->validateContract(frameContractFrom(msg))" in validate_frame
    assert "FrameContractStatus::kOk" in validate_frame
    assert "frameContractStatusName(contract_status)" in validate_frame
    assert "return false;" in validate_frame

    backend_source = (
        package_root / "src" / "backends" / "internal_float32le.cpp"
    ).read_text(encoding="utf-8")
    assert "validateContract(const FrameContract & contract)" in backend_source
    assert "contract.input_encoding != config_.input_encoding" in backend_source
    assert "contract.input_bit_depth != static_cast<uint32_t>(config_.input_bit_depth)" in backend_source
    assert "contract.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in backend_source
    assert "contract.channels != static_cast<uint32_t>(config_.expected_channels)" in backend_source
    assert "contract.layout != config_.expected_layout" in backend_source
    assert "contract.data_size == 0" in backend_source
    assert "contract.data_size % bytes_per_frame" in backend_source


def test_sample_format_preserves_identity_and_updates_format_metadata() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_sample_format_node.cpp").read_text(encoding="utf-8")
    convert_frame = source.split("bool FaSampleFormatNode::convertFrame")[1].split(
        "void FaSampleFormatNode::publishDiagnostics"
    )[0]

    assert "backend_->process(in.data, frameContractFrom(in), output_data)" in convert_frame
    assert "out = in;" in convert_frame
    assert "out.stream_id = config_.output_stream_id;" in convert_frame
    assert "out.encoding = backend_->outputEncoding();" in convert_frame
    assert "out.bit_depth = static_cast<uint32_t>(backend_->outputBitDepth());" in convert_frame
    assert "out.sample_rate = in.sample_rate;" in convert_frame
    assert "out.channels = in.channels;" in convert_frame
    assert "out.layout = in.layout;" in convert_frame
    assert "out.data = output_data;" in convert_frame
    assert ".rms" not in convert_frame
    assert ".peak" not in convert_frame
    assert ".vad" not in convert_frame


def test_sample_format_conversions_are_explicit_without_hidden_clamp() -> None:
    package_root = Path(__file__).parents[2]
    conversion = (
        package_root / "src" / "backends" / "internal_float32le.cpp"
    ).read_text(encoding="utf-8")
    pcm16 = conversion.split("ByteConversionResult convertPcm16ToFloat32")[1].split(
        "ByteConversionResult convertPcm32ToFloat32"
    )[0]
    pcm32 = conversion.split("ByteConversionResult convertPcm32ToFloat32")[1].split(
        "ByteConversionResult convertFloat32ToPcm16"
    )[0]
    float32_to_pcm16 = conversion.split(
        "ByteConversionResult convertFloat32ToPcm16"
    )[1].split("InternalFloat32LeBackend::InternalFloat32LeBackend")[0]
    append = conversion.split("void appendFloat32Le")[1].split(
        "void appendPcm16Le"
    )[0]

    assert "input_bytes.size() % sizeof(uint16_t)" in pcm16
    assert "static_cast<uint16_t>(input_bytes.at(i))" in pcm16
    assert "static_cast<int32_t>(raw) - 0x10000" in pcm16
    assert "kPcm16Scale" in pcm16
    assert "input_bytes.size() % sizeof(uint32_t)" in pcm32
    assert "static_cast<uint32_t>(input_bytes.at(i + 3)) << 24U" in pcm32
    assert "static_cast<int64_t>(raw) - 0x100000000LL" in pcm32
    assert "kPcm32Scale" in pcm32
    assert "std::memcpy(&raw, &sample, sizeof(float));" in append
    assert "out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));" in append
    assert "std::isfinite(sample)" in float32_to_pcm16
    assert "isFiniteNormalizedFloat32(sample)" in float32_to_pcm16
    assert "std::lround(scaled)" in float32_to_pcm16
    assert "appendPcm16Le(static_cast<int16_t>(rounded), out_bytes);" in float32_to_pcm16
    assert "std::clamp" not in conversion


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_float32le.md",
        "config/default.yaml",
        "launch/fa_sample_format.launch.py",
        "include/fa_sample_format/fa_sample_format_node.hpp",
        "include/fa_sample_format/backends/internal_float32le.hpp",
        "src/backends/internal_float32le.cpp",
        "src/fa_sample_format_node.cpp",
        "src/main.cpp",
        "test/cpp",
        "test/cpp/test_internal_float32le_backend.cpp",
        "test/unit",
        "test/integration",
        "test/launch",
        "test/fixtures",
    )

    for relative_path in required_paths:
        assert (package_root / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "add_library(fa_sample_format_internal_float32le" in cmake_text
    assert "add_library(fa_sample_format_node_core" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "target_link_libraries(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

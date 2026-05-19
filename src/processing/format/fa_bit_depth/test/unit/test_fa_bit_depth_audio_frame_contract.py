from pathlib import Path

import yaml


def test_default_config_requires_explicit_pcm16_to_pcm32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_bit_depth"]["ros__parameters"]

    assert params["input_topic"] == "fa_bit_depth/input"
    assert params["output_topic"] == "fa_bit_depth/output"
    assert params["input_stream_id"] == "audio/raw/mic"
    assert params["output"]["stream_id"] == "audio/bit_depth/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["input"]["encoding"] == "PCM16LE"
    assert params["input"]["bit_depth"] == 16
    assert params["output"]["encoding"] == "PCM32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["layout"] == "interleaved"


def test_bit_depth_does_not_hide_io_or_other_processing_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")

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
        "FLOAT32LE",
        "std::clamp",
        "kPcm16Scale",
        "kPcm32Scale",
        "sizeof(float)",
    )
    for token in forbidden:
        assert token not in source


def test_startup_rejects_unknown_or_implicit_conversion_config() -> None:
    package_root = Path(__file__).parents[2]
    node_source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    backend_source = (
        package_root / "src" / "backends" / "internal_integer_bit_depth.cpp"
    ).read_text(encoding="utf-8")
    is_supported = backend_source.split("bool isSupportedConversion")[1].split(
        "size_t bytesPerSample"
    )[0]

    assert "input_encoding == kEncodingPcm16Le && input_bit_depth == 16" in is_supported
    assert "output_encoding == kEncodingPcm32Le && output_bit_depth == 32" in is_supported
    assert "input_encoding == kEncodingPcm32Le && input_bit_depth == 32" not in is_supported
    assert "output_encoding == kEncodingPcm16Le && output_bit_depth == 16" not in is_supported
    assert "isSupportedConversion(" in backend_source
    assert "throw std::runtime_error(" in backend_source
    assert "lossless PCM16LE/16 -> PCM32LE/32" in backend_source
    assert "std::make_unique<backends::InternalIntegerBitDepthBackend>" in node_source
    assert "metadata" not in is_supported


def test_ros2_node_name_and_executable_match_required_contract() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    header = (
        package_root / "include" / "fa_bit_depth" / "fa_bit_depth_node.hpp"
    ).read_text(encoding="utf-8")
    main_source = (package_root / "src" / "main.cpp").read_text(encoding="utf-8")
    launch = (package_root / "launch" / "fa_bit_depth.launch.py").read_text(encoding="utf-8")
    cmake = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "class FaBitDepthNode : public rclcpp::Node" in header
    assert 'rclcpp::Node("fa_bit_depth", options)' in source
    assert "explicit FaBitDepthNode(const rclcpp::NodeOptions & options" in header
    assert "fa_bit_depth::FaBitDepthNode" in main_source
    assert "default_value" not in launch
    assert "FindPackageShare" not in launch
    assert "PathJoinSubstitution" not in launch
    assert 'executable="fa_bit_depth_node"' in launch
    assert "add_executable(fa_bit_depth_node" in cmake


def test_bit_depth_requires_explicit_runtime_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaBitDepthNode::loadParameters")[1].split(
        "void FaBitDepthNode::setupInterfaces"
    )[0]

    assert 'readRequiredString(*this, "input_topic")' in load_parameters
    assert 'readRequiredString(*this, "input_stream_id")' in load_parameters
    assert 'readRequiredString(*this, "input.encoding")' in load_parameters
    assert 'readRequiredString(*this, "output.stream_id")' in load_parameters
    assert 'readRequiredInt(*this, "expected.sample_rate")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    assert 'readRequiredInt(*this, "diagnostics.qos.depth")' in load_parameters
    assert 'readRequiredBool(*this, "diagnostics.qos.reliable")' in load_parameters
    assert "input_stream_id must be distinct from ROS topics" in load_parameters
    assert "output.stream_id must be distinct from ROS topics" in load_parameters
    assert "input_stream_id and output.stream_id must be distinct" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line


def test_bit_depth_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaBitDepthNode::validateFrame")[1].split(
        "bool FaBitDepthNode::convertFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "backend_->validateContract(frameContractFrom(msg))" in validate_frame
    assert "FrameContractStatus::kOk" in validate_frame
    assert "frameContractStatusName(contract_status)" in validate_frame
    assert "RCLCPP_WARN_THROTTLE" in validate_frame
    assert "return false;" in validate_frame

    backend_header = (
        package_root
        / "include"
        / "fa_bit_depth"
        / "backends"
        / "internal_integer_bit_depth.hpp"
    ).read_text(encoding="utf-8")
    backend_source = (
        package_root / "src" / "backends" / "internal_integer_bit_depth.cpp"
    ).read_text(encoding="utf-8")
    assert "struct FrameContract" in backend_header
    assert "validateContract(" in backend_source
    assert "const FrameContract & contract" in backend_source
    assert "contract.input_encoding != config_.input_encoding" in backend_source
    assert "contract.input_bit_depth != static_cast<uint32_t>(config_.input_bit_depth)" in backend_source
    assert "contract.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in backend_source
    assert "contract.channels != static_cast<uint32_t>(config_.expected_channels)" in backend_source
    assert "contract.layout != config_.expected_layout" in backend_source
    assert "contract.data_size == 0" in backend_source
    assert "contract.data_size % bytes_per_frame" in backend_source


def test_bit_depth_preserves_identity_and_updates_format_metadata() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    convert_frame = source.split("bool FaBitDepthNode::convertFrame")[1].split(
        "void FaBitDepthNode::publishDiagnostics"
    )[0]

    assert "backend_->process(in.data, frameContractFrom(in), output_data)" in convert_frame
    assert "out = in;" in convert_frame
    assert "out.stream_id = config_.output_stream_id;" in convert_frame
    assert "out.stream_id = config_.output_topic;" not in convert_frame
    assert "out.encoding = backend_->outputEncoding();" in convert_frame
    assert "out.bit_depth = static_cast<uint32_t>(backend_->outputBitDepth());" in convert_frame
    assert "out.sample_rate = in.sample_rate;" in convert_frame
    assert "out.channels = in.channels;" in convert_frame
    assert "out.layout = in.layout;" in convert_frame
    assert "out.data = output_data;" in convert_frame
    assert ".rms" not in convert_frame
    assert ".peak" not in convert_frame
    assert ".vad" not in convert_frame


def test_pcm_integer_bit_depth_conversion_uses_lossless_high_order_expansion_only() -> None:
    package_root = Path(__file__).parents[2]
    node_source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    backend_source = (
        package_root / "src" / "backends" / "internal_integer_bit_depth.cpp"
    ).read_text(encoding="utf-8")
    pcm16 = backend_source.split("std::vector<uint8_t> convertPcm16ToPcm32")[1].split(
        "InternalIntegerBitDepthBackend::InternalIntegerBitDepthBackend"
    )[0]
    append32 = backend_source.split("void appendPcm32Le")[1].split(
        "std::vector<uint8_t> convertPcm16ToPcm32"
    )[0]

    assert "input_bytes.size() % sizeof(uint16_t)" in pcm16
    assert "static_cast<uint16_t>(input_bytes.at(i))" in pcm16
    assert "const uint32_t aligned_sample = static_cast<uint32_t>(raw) << 16U;" in pcm16
    assert "appendPcm32Le(aligned_sample, out_bytes);" in pcm16
    assert "convertPcm32ToPcm16" not in node_source
    assert "convertPcm32ToPcm16" not in backend_source
    assert "appendPcm16Le" not in node_source
    assert "appendPcm16Le" not in backend_source
    assert "raw >> 16U" not in backend_source
    assert "high_word" not in backend_source
    assert "out_bytes.push_back(static_cast<uint8_t>((sample >> 24U) & 0xFFU));" in append32


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_integer_bit_depth.md",
        "config/default.yaml",
        "launch/fa_bit_depth.launch.py",
        "include/fa_bit_depth/fa_bit_depth_node.hpp",
        "include/fa_bit_depth/backends/internal_integer_bit_depth.hpp",
        "src/fa_bit_depth_node.cpp",
        "src/backends/internal_integer_bit_depth.cpp",
        "src/main.cpp",
        "test/unit",
        "test/cpp/test_fa_bit_depth_node_contract.cpp",
        "test/cpp/test_internal_integer_bit_depth_backend.cpp",
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
    assert "add_library(fa_bit_depth_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_node_contract_test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "add_library(fa_bit_depth_internal_integer_bit_depth" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text

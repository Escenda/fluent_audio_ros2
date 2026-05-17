from pathlib import Path

import yaml


def test_default_config_requires_explicit_pcm16_to_pcm32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_bit_depth"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"
    assert params["output_topic"] == "audio/bit_depth/mic"
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
    source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    is_supported = source.split("bool FaBitDepthNode::isSupportedConversion")[1].split(
        "size_t FaBitDepthNode::bytesPerSample"
    )[0]

    assert "input_encoding == kEncodingPcm16 && input_bit_depth == 16" in is_supported
    assert "output_encoding == kEncodingPcm32 && output_bit_depth == 32" in is_supported
    assert "input_encoding == kEncodingPcm32 && input_bit_depth == 32" not in is_supported
    assert "output_encoding == kEncodingPcm16 && output_bit_depth == 16" not in is_supported
    assert "isSupportedConversion(" in source
    assert "throw std::runtime_error(" in source
    assert "lossless PCM16LE/16 -> PCM32LE/32" in source
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
    assert 'default_value="fa_bit_depth"' in launch
    assert 'executable="fa_bit_depth_node"' in launch
    assert "add_executable(fa_bit_depth_node" in cmake


def test_bit_depth_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaBitDepthNode::validateFrame")[1].split(
        "bool FaBitDepthNode::convertFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.input_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.input_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "RCLCPP_WARN_THROTTLE" in validate_frame
    assert "return false;" in validate_frame


def test_bit_depth_preserves_identity_and_updates_format_metadata() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    convert_frame = source.split("bool FaBitDepthNode::convertFrame")[1].split(
        "bool FaBitDepthNode::isSupportedConversion"
    )[0]

    assert "out = in;" in convert_frame
    assert "out.stream_id = config_.output_topic;" in convert_frame
    assert "out.encoding = config_.output_encoding;" in convert_frame
    assert "out.bit_depth = static_cast<uint32_t>(config_.output_bit_depth);" in convert_frame
    assert "out.sample_rate = in.sample_rate;" in convert_frame
    assert "out.channels = in.channels;" in convert_frame
    assert "out.layout = in.layout;" in convert_frame
    assert "out.data = output_data;" in convert_frame
    assert ".rms" not in convert_frame
    assert ".peak" not in convert_frame
    assert ".vad" not in convert_frame


def test_pcm_integer_bit_depth_conversion_uses_lossless_high_order_expansion_only() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_bit_depth_node.cpp").read_text(encoding="utf-8")
    pcm16 = source.split("std::vector<uint8_t> FaBitDepthNode::convertPcm16ToPcm32")[1].split(
        "void FaBitDepthNode::appendPcm16Le"
    )[0]
    append16 = source.split("void FaBitDepthNode::appendPcm16Le")[1].split(
        "void FaBitDepthNode::appendPcm32Le"
    )[0]
    append32 = source.split("void FaBitDepthNode::appendPcm32Le")[1].split(
        "void FaBitDepthNode::publishDiagnostics"
    )[0]

    assert "input_bytes.size() % sizeof(uint16_t)" in pcm16
    assert "static_cast<uint16_t>(input_bytes.at(i))" in pcm16
    assert "const uint32_t aligned_sample = static_cast<uint32_t>(raw) << 16U;" in pcm16
    assert "appendPcm32Le(aligned_sample, out_bytes);" in pcm16
    assert "convertPcm32ToPcm16" not in source
    assert "raw >> 16U" not in source
    assert "high_word" not in source
    assert "out_bytes.push_back(static_cast<uint8_t>((sample >> 8U) & 0xFFU));" in append16
    assert "out_bytes.push_back(static_cast<uint8_t>((sample >> 24U) & 0xFFU));" in append32


def test_docs_reject_lossy_pcm32_to_pcm16_truncation() -> None:
    package_root = Path(__file__).parents[2]
    readme = (package_root / "README.md").read_text(encoding="utf-8")
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package_root / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    test_plan = (package_root / "docs" / "テスト設計.md").read_text(
        encoding="utf-8"
    )
    backend_doc = (
        package_root / "docs" / "backends" / "internal_integer_bit_depth.md"
    ).read_text(encoding="utf-8")

    assert "PCM32LE` / 32 bit / `interleaved` -> `PCM16LE" not in readme
    assert "`PCM32LE` / 32 / `interleaved` | `PCM16LE`" not in spec
    assert "PCM32LE から PCM16LE" not in algorithm
    assert "下位 16 bit は破棄する" not in algorithm
    assert "lossy PCM32LE/32 -> PCM16LE/16" in algorithm
    assert "lossy PCM32LE/32 -> PCM16LE/16 は fail closed" in test_plan
    assert "`PCM32LE` / 32 | `PCM16LE` / 16" not in backend_doc


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
        "src/fa_bit_depth_node.cpp",
        "src/main.cpp",
        "test/unit",
        "test/cpp/test_fa_bit_depth_node_contract.cpp",
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

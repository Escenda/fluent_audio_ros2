from pathlib import Path

import yaml


def test_default_config_requires_explicit_float32le_channel_convert_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_channel_convert"]["ros__parameters"]

    assert params["input_topic"] == "audio/sample_format/mic"
    assert params["output_topic"] == "audio/channel_converted/mic"
    assert params["input"]["channels"] == 1
    assert params["output"]["channels"] == 2
    assert params["conversion"]["mode"] == "mono_to_stereo_duplicate"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_channel_convert_does_not_hide_unrelated_processing_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_channel_convert_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "PCM16LE",
        "PCM32LE",
        "gain.linear",
        "threshold.linear",
        "noise_gate",
        "std::clamp",
        "normalize(",
        "decode",
        "encode",
    )
    for token in forbidden:
        assert token not in source


def test_startup_rejects_unsupported_or_implicit_channel_conversion() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_channel_convert_node.cpp").read_text(encoding="utf-8")
    supported = source.split("bool FaChannelConvertNode::isSupportedConversion")[1].split(
        "float FaChannelConvertNode::readFloat32Le"
    )[0]

    assert "mode == kModeMonoToStereoDuplicate && input_channels == 1 && output_channels == 2" in supported
    assert "mode == kModeStereoToMonoAverage && input_channels == 2 && output_channels == 1" in supported
    assert "isSupportedConversion(config_.conversion_mode, config_.input_channels, config_.output_channels)" in source
    assert "throw std::runtime_error(" in source
    assert "else if (config_.conversion_mode == kModeStereoToMonoAverage)" in source
    assert "else {\n    return false;\n  }" in source


def test_channel_convert_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_channel_convert_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaChannelConvertNode::validateFrame")[1].split(
        "bool FaChannelConvertNode::convertFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.input_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_channel_convert_preserves_identity_and_updates_stream_channels_data_only() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_channel_convert_node.cpp").read_text(encoding="utf-8")
    convert_frame = source.split("bool FaChannelConvertNode::convertFrame")[1].split(
        "bool FaChannelConvertNode::isSupportedConversion"
    )[0]

    assert "out = in;" in convert_frame
    assert "out.stream_id = config_.output_topic;" in convert_frame
    assert "out.channels = static_cast<uint32_t>(config_.output_channels);" in convert_frame
    assert "out.data = output_data;" in convert_frame
    assert "out.encoding =" not in convert_frame
    assert "out.bit_depth =" not in convert_frame
    assert "out.sample_rate =" not in convert_frame
    assert "out.layout =" not in convert_frame
    assert ".rms" not in convert_frame
    assert ".peak" not in convert_frame
    assert ".vad" not in convert_frame


def test_channel_convert_algorithms_are_explicit_and_validate_normalized_samples() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_channel_convert_node.cpp").read_text(encoding="utf-8")
    convert_frame = source.split("bool FaChannelConvertNode::convertFrame")[1].split(
        "bool FaChannelConvertNode::isSupportedConversion"
    )[0]
    read_float = source.split("float FaChannelConvertNode::readFloat32Le")[1].split(
        "void FaChannelConvertNode::appendFloat32Le"
    )[0]
    append = source.split("void FaChannelConvertNode::appendFloat32Le")[1].split(
        "bool FaChannelConvertNode::isNormalizedFinite"
    )[0]
    range_check = source.split("bool FaChannelConvertNode::isNormalizedFinite")[1].split(
        "void FaChannelConvertNode::publishDiagnostics"
    )[0]

    assert "config_.conversion_mode == kModeMonoToStereoDuplicate" in convert_frame
    assert "appendFloat32Le(sample, output_data);\n      appendFloat32Le(sample, output_data);" in convert_frame
    assert "config_.conversion_mode == kModeStereoToMonoAverage" in convert_frame
    assert "const float averaged = (left + right) * 0.5F;" in convert_frame
    assert "isNormalizedFinite(sample)" in convert_frame
    assert "isNormalizedFinite(left) || !isNormalizedFinite(right)" in convert_frame
    assert "isNormalizedFinite(averaged)" in convert_frame
    assert "static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 3U)) << 24U" in read_float
    assert "std::memcpy(&sample, &raw, sizeof(float));" in read_float
    assert "std::memcpy(&raw, &sample, sizeof(float));" in append
    assert "out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));" in append
    assert "std::isfinite(sample)" in range_check
    assert "sample >= kMinNormalizedSample" in range_check
    assert "sample <= kMaxNormalizedSample" in range_check


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_float32le_channel_convert.md",
        "config/default.yaml",
        "launch/fa_channel_convert.launch.py",
        "include/fa_channel_convert/fa_channel_convert_node.hpp",
        "src/fa_channel_convert_node.cpp",
        "test/unit",
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

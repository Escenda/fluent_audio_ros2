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
    launch_text = (package_root / "launch" / "fa_channel_convert.launch.py").read_text(
        encoding="utf-8"
    )
    load_parameters = source.split("void FaChannelConvertNode::loadParameters")[1].split(
        "void FaChannelConvertNode::setupInterfaces"
    )[0]
    backend_source = (
        package_root / "src" / "backends" / "internal_float32le_channel_convert.cpp"
    ).read_text(encoding="utf-8")
    supported = backend_source.split("bool isSupportedChannelConversion")[1].split(
        "float readFloat32Le"
    )[0]

    assert "mode == kModeMonoToStereoDuplicate && input_channels == 1 && output_channels == 2" in supported
    assert "mode == kModeStereoToMonoAverage && input_channels == 2 && output_channels == 1" in supported
    assert "isSupportedChannelConversion(" in backend_source
    assert "throw std::runtime_error(" in backend_source
    assert "std::make_unique<backends::InternalFloat32LeChannelConvertBackend>" in source
    assert "std::max<int>(1, config_.qos_depth)" not in source
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'readRequiredString(*this, "input_topic")' in load_parameters
    assert 'readRequiredInt(*this, "input.channels")' in load_parameters
    assert 'readRequiredString(*this, "conversion.mode")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    assert 'readRequiredInt(*this, "diagnostics.qos.depth")' in load_parameters
    assert 'readRequiredBool(*this, "diagnostics.qos.reliable")' in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line


def test_channel_convert_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_channel_convert_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaChannelConvertNode::validateFrame")[1].split(
        "bool FaChannelConvertNode::convertFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "backend_->validateContract(frameContractFrom(msg))" in validate_frame
    assert "backends::frameContractStatusName(contract_status)" in validate_frame
    assert "return false;" in validate_frame


def test_channel_convert_preserves_identity_and_updates_stream_channels_data_only() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_channel_convert_node.cpp").read_text(encoding="utf-8")
    convert_frame = source.split("bool FaChannelConvertNode::convertFrame")[1].split(
        "void FaChannelConvertNode::publishDiagnostics"
    )[0]

    assert "out = in;" in convert_frame
    assert "out.stream_id = config_.output_topic;" in convert_frame
    assert "out.channels = static_cast<uint32_t>(backend_->outputChannels());" in convert_frame
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
    node_source = (package_root / "src" / "fa_channel_convert_node.cpp").read_text(encoding="utf-8")
    backend_source = (
        package_root / "src" / "backends" / "internal_float32le_channel_convert.cpp"
    ).read_text(encoding="utf-8")
    read_float = backend_source.split("float readFloat32Le")[1].split(
        "void appendFloat32Le"
    )[0]
    append = backend_source.split("void appendFloat32Le")[1].split(
        "bool isNormalizedFinite"
    )[0]
    range_check = backend_source.split("bool isNormalizedFinite")[1].split(
        "ChannelConversionResult convertMonoToStereoDuplicate"
    )[0]
    mono_convert = backend_source.split("ChannelConversionResult convertMonoToStereoDuplicate")[1].split(
        "ChannelConversionResult convertStereoToMonoAverage"
    )[0]
    stereo_convert = backend_source.split("ChannelConversionResult convertStereoToMonoAverage")[1].split(
        "InternalFloat32LeChannelConvertBackend::InternalFloat32LeChannelConvertBackend"
    )[0]

    assert "backend_->process(in.data, frameContractFrom(in), output_data)" in node_source
    assert "config_.conversion_mode == kModeMonoToStereoDuplicate" not in node_source
    assert "FaChannelConvertNode::readFloat32Le" not in node_source
    assert "FaChannelConvertNode::appendFloat32Le" not in node_source
    assert "FaChannelConvertNode::isNormalizedFinite" not in node_source
    assert "appendFloat32Le(sample, output_bytes);\n    appendFloat32Le(sample, output_bytes);" in mono_convert
    assert "const float averaged = (left + right) * 0.5F;" in stereo_convert
    assert "isNormalizedFinite(sample)" in mono_convert
    assert "isNormalizedFinite(left) || !isNormalizedFinite(right)" in stereo_convert
    assert "isNormalizedFinite(averaged)" in stereo_convert
    assert "static_cast<uint32_t>(bytes.at(offset + 3U)) << 24U" in read_float
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
        "include/fa_channel_convert/backends/internal_float32le_channel_convert.hpp",
        "include/fa_channel_convert/fa_channel_convert_node.hpp",
        "src/backends/internal_float32le_channel_convert.cpp",
        "src/fa_channel_convert_node.cpp",
        "test/cpp/test_internal_float32le_channel_convert_backend.cpp",
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
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "fa_channel_convert_internal_float32le_channel_convert" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

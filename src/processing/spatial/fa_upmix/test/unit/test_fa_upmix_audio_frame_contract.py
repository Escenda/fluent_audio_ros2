from pathlib import Path

import yaml


def test_default_config_requires_explicit_float32le_upmix_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_upmix"]["ros__parameters"]

    assert params["input_topic"] == "audio/mono/mic"
    assert params["output_topic"] == "audio/upmixed/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["input_channels"] == 1
    assert params["output"]["channels"] == 2
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["mode"] == "mono_duplicate"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_launch_requires_explicit_node_name_and_config_file() -> None:
    package_root = Path(__file__).parents[2]
    launch_text = (package_root / "launch" / "fa_upmix.launch.py").read_text(encoding="utf-8")

    assert "DeclareLaunchArgument(" in launch_text
    assert '"node_name"' in launch_text
    assert '"config_file"' in launch_text
    assert 'LaunchConfiguration("node_name")' in launch_text
    assert 'LaunchConfiguration("config_file")' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text


def test_upmix_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_upmix_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "PCM16LE",
        "PCM32LE",
        "resample",
        "average_to_mono",
        "pair_average_to_stereo",
        "gain.linear",
        "threshold.linear",
        "filter.",
        "cutoff_hz",
        "center_hz",
        "denoise",
        "std::clamp",
        "normalize(",
        ".rms",
        ".peak",
        ".vad",
    )
    for token in forbidden:
        assert token not in source


def test_startup_validation_fails_closed_for_invalid_or_non_upmix_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_upmix_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaUpmixNode::loadParameters")[1].split(
        "void FaUpmixNode::setupInterfaces"
    )[0]
    supported = source.split("bool FaUpmixNode::isSupportedUpmix")[1].split(
        "float FaUpmixNode::readFloat32Le"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_input_channels <= 0" in load_parameters
    assert "config_.output_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "!isSupportedUpmix(config_.mode, config_.expected_input_channels, config_.output_channels)" in source
    assert "throw std::runtime_error" in load_parameters
    assert "mode == kModeMonoDuplicate && input_channels == 1 && output_channels > 1" in supported
    assert "mode == kModeStereoDuplicatePairs && input_channels == 2" in supported
    assert "output_channels > 2 && (output_channels % 2) == 0" in supported


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_upmix_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaUpmixNode::loadParameters")[1].split(
        "void FaUpmixNode::setupInterfaces"
    )[0]

    assert 'readRequiredString(*this, "input_topic")' in load_parameters
    assert 'readRequiredString(*this, "output_topic")' in load_parameters
    assert 'readRequiredInt(*this, "expected.sample_rate")' in load_parameters
    assert 'readRequiredInt(*this, "expected.input_channels")' in load_parameters
    assert 'readRequiredInt(*this, "output.channels")' in load_parameters
    assert 'readRequiredString(*this, "expected.encoding")' in load_parameters
    assert 'readRequiredInt(*this, "expected.bit_depth")' in load_parameters
    assert 'readRequiredString(*this, "expected.layout")' in load_parameters
    assert 'readRequiredString(*this, "mode")' in load_parameters
    assert 'readRequiredInt(*this, "qos.depth")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    assert "config_.diagnostics_publish_period_ms = readRequiredInt(" in load_parameters
    assert '"diagnostics.publish_period_ms"' in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert "config_." not in line


def test_upmix_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_upmix_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaUpmixNode::validateFrame")[1].split(
        "bool FaUpmixNode::upmixFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_input_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_upmix_preserves_metadata_and_updates_stream_channels_data_only() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_upmix_node.cpp").read_text(encoding="utf-8")
    upmix_frame = source.split("bool FaUpmixNode::upmixFrame")[1].split(
        "bool FaUpmixNode::isSupportedUpmix"
    )[0]

    assert "out = in;" in upmix_frame
    assert "out.stream_id = config_.output_topic;" in upmix_frame
    assert "out.channels = static_cast<uint32_t>(config_.output_channels);" in upmix_frame
    assert "out.data = output_data;" in upmix_frame
    assert "out.encoding =" not in upmix_frame
    assert "out.bit_depth =" not in upmix_frame
    assert "out.sample_rate =" not in upmix_frame
    assert "out.layout =" not in upmix_frame
    assert "out.epoch =" not in upmix_frame


def test_upmix_algorithms_are_explicit_and_validate_normalized_samples() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_upmix_node.cpp").read_text(encoding="utf-8")
    upmix_frame = source.split("bool FaUpmixNode::upmixFrame")[1].split(
        "bool FaUpmixNode::isSupportedUpmix"
    )[0]
    read_float = source.split("float FaUpmixNode::readFloat32Le")[1].split(
        "void FaUpmixNode::appendFloat32Le"
    )[0]
    append = source.split("void FaUpmixNode::appendFloat32Le")[1].split(
        "bool FaUpmixNode::isNormalizedFinite"
    )[0]
    range_check = source.split("bool FaUpmixNode::isNormalizedFinite")[1].split(
        "void FaUpmixNode::publishDiagnostics"
    )[0]

    assert "config_.mode == kModeMonoDuplicate" in upmix_frame
    assert "const float sample = readFloat32Le(in.data, frame_index);" in upmix_frame
    assert "for (size_t channel_index = 0; channel_index < output_channels; ++channel_index)" in upmix_frame
    assert "appendFloat32Le(sample, output_data);" in upmix_frame
    assert "config_.mode == kModeStereoDuplicatePairs" in upmix_frame
    assert "const size_t output_pairs = output_channels / 2U;" in upmix_frame
    assert "const float left = readFloat32Le(in.data, frame_sample_offset);" in upmix_frame
    assert "const float right = readFloat32Le(in.data, frame_sample_offset + 1U);" in upmix_frame
    assert "for (size_t pair_index = 0; pair_index < output_pairs; ++pair_index)" in upmix_frame
    assert "appendFloat32Le(left, output_data);" in upmix_frame
    assert "appendFloat32Le(right, output_data);" in upmix_frame
    assert "static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 3U)) << 24U" in read_float
    assert "std::memcpy(&sample, &raw, sizeof(float));" in read_float
    assert "std::memcpy(&raw, &sample, sizeof(float));" in append
    assert "out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));" in append
    assert "std::isfinite(sample)" in range_check
    assert "sample >= kMinNormalizedSample" in range_check
    assert "sample <= kMaxNormalizedSample" in range_check


def test_upmix_drops_invalid_runtime_samples_instead_of_clamping_or_normalizing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_upmix_node.cpp").read_text(encoding="utf-8")
    upmix_frame = source.split("bool FaUpmixNode::upmixFrame")[1].split(
        "bool FaUpmixNode::isSupportedUpmix"
    )[0]

    assert "!isNormalizedFinite(sample)" in upmix_frame
    assert "!isNormalizedFinite(left) || !isNormalizedFinite(right)" in upmix_frame
    assert "Dropping frame because input sample is outside normalized FLOAT32LE range" in upmix_frame
    assert "Dropping frame because upmix output is outside normalized FLOAT32LE range" in upmix_frame
    assert "return false;" in upmix_frame
    assert "std::clamp" not in source
    assert "normalize(" not in source


def test_diagnostics_include_upmix_config_and_counters() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_upmix_node.cpp").read_text(encoding="utf-8")
    diagnostics = source.split("void FaUpmixNode::publishDiagnostics")[1].split(
        "}  // namespace fa_upmix"
    )[0]

    assert 'status.name = "fa_upmix";' in diagnostics
    assert 'pushKeyValue(status, "mode", config_.mode);' in diagnostics
    assert (
        'pushKeyValue(status, "expected.input_channels", '
        'std::to_string(config_.expected_input_channels));'
    ) in diagnostics
    assert 'pushKeyValue(status, "output.channels", std::to_string(config_.output_channels));' in diagnostics
    assert 'pushKeyValue(status, "frames.in", std::to_string(frames_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames.out", std::to_string(frames_out_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames.drop", std::to_string(frames_dropped_.load()));' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/no_runtime_backend.md",
        "config/default.yaml",
        "launch/fa_upmix.launch.py",
        "include/fa_upmix/fa_upmix_node.hpp",
        "src/fa_upmix_node.cpp",
        "test/unit/test_fa_upmix_audio_frame_contract.py",
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
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

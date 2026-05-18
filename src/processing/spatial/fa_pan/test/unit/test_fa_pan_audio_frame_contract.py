from pathlib import Path

import yaml


def test_default_config_requires_stereo_float32le_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_pan"]["ros__parameters"]

    assert params["input_topic"] == "fa_pan/input"
    assert params["output_topic"] == "fa_pan/output"
    assert params["input_stream_id"] == "audio/channel_converted/mic"
    assert params["output"]["stream_id"] == "audio/panned/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["pan"]["position"] == 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 2
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_launch_requires_explicit_node_name_and_config_file() -> None:
    package_root = Path(__file__).parents[2]
    launch_text = (package_root / "launch" / "fa_pan.launch.py").read_text(encoding="utf-8")

    assert "DeclareLaunchArgument(" in launch_text
    assert '"node_name"' in launch_text
    assert '"config_file"' in launch_text
    assert 'LaunchConfiguration("node_name")' in launch_text
    assert 'LaunchConfiguration("config_file")' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text


def test_pan_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_pan_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "PCM16LE",
        "PCM32LE",
        "resample",
        "set_channels",
        "conversion.mode",
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


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_pan_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaPanNode::loadParameters")[1].split(
        "void FaPanNode::configurePan"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "config_.input_stream_id.empty()" in load_parameters
    assert "config_.output_stream_id.empty()" in load_parameters
    assert "resolve_topic_name(config_.input_topic)" in load_parameters
    assert "resolve_topic_name(config_.output_topic)" in load_parameters
    assert "input_stream_id must be distinct from ROS topics" in load_parameters
    assert "output.stream_id must be distinct from ROS topics" in load_parameters
    assert "input_stream_id and output.stream_id must be distinct" in load_parameters
    assert "!isFinite(config_.pan_position)" in load_parameters
    assert "config_.pan_position < -1.0" in load_parameters
    assert "config_.pan_position > 1.0" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels != 2" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "config_.diagnostics_qos_depth <= 0" in load_parameters
    assert "throw std::runtime_error" in load_parameters


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_pan_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaPanNode::loadParameters")[1].split(
        "void FaPanNode::configurePan"
    )[0]

    assert 'readRequiredString(*this, "input_topic")' in load_parameters
    assert 'readRequiredString(*this, "output_topic")' in load_parameters
    assert 'readRequiredString(*this, "input_stream_id")' in load_parameters
    assert 'readRequiredString(*this, "output.stream_id")' in load_parameters
    assert 'readRequiredDouble(*this, "pan.position")' in load_parameters
    assert 'readRequiredInt(*this, "expected.sample_rate")' in load_parameters
    assert 'readRequiredInt(*this, "expected.channels")' in load_parameters
    assert 'readRequiredString(*this, "expected.encoding")' in load_parameters
    assert 'readRequiredInt(*this, "expected.bit_depth")' in load_parameters
    assert 'readRequiredString(*this, "expected.layout")' in load_parameters
    assert 'readRequiredInt(*this, "qos.depth")' in load_parameters
    assert 'readRequiredInt(*this, "diagnostics.qos.depth")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    assert 'readRequiredBool(*this, "diagnostics.qos.reliable")' in load_parameters
    assert "config_.diagnostics_publish_period_ms = readRequiredInt(" in load_parameters
    assert "config_.diagnostics_qos_depth = readRequiredInt(" in load_parameters
    assert "config_.diagnostics_qos_reliable = readRequiredBool(" in load_parameters
    assert '"diagnostics.publish_period_ms"' in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert "config_." not in line


def test_pan_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_pan_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaPanNode::validateFrame")[1].split(
        "bool FaPanNode::applyPan"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_pan_preserves_identity_and_updates_stream_data_only() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_pan_node.cpp").read_text(encoding="utf-8")
    apply_pan = source.split("bool FaPanNode::applyPan")[1].split(
        "float FaPanNode::readFloat32Le"
    )[0]

    assert "out = in;" in apply_pan
    assert "out.stream_id = config_.output_stream_id;" in apply_pan
    assert "out.data = output_data;" in apply_pan
    assert "out.encoding =" not in apply_pan
    assert "out.bit_depth =" not in apply_pan
    assert "out.sample_rate =" not in apply_pan
    assert "out.channels =" not in apply_pan
    assert "out.layout =" not in apply_pan
    assert "out.epoch =" not in apply_pan


def test_constant_power_pan_algorithm_and_normalized_sample_validation() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_pan_node.cpp").read_text(encoding="utf-8")
    configure_pan = source.split("void FaPanNode::configurePan")[1].split(
        "void FaPanNode::setupInterfaces"
    )[0]
    apply_pan = source.split("bool FaPanNode::applyPan")[1].split(
        "float FaPanNode::readFloat32Le"
    )[0]
    read_float = source.split("float FaPanNode::readFloat32Le")[1].split(
        "void FaPanNode::appendFloat32Le"
    )[0]
    append = source.split("void FaPanNode::appendFloat32Le")[1].split(
        "bool FaPanNode::isNormalizedFinite"
    )[0]
    range_check = source.split("bool FaPanNode::isNormalizedFinite")[1].split(
        "void FaPanNode::publishDiagnostics"
    )[0]

    assert "const double angle = (config_.pan_position + 1.0) * kPi / 4.0;" in configure_pan
    assert "left_gain_ = std::cos(angle);" in configure_pan
    assert "right_gain_ = std::sin(angle);" in configure_pan
    assert "for (size_t sample_index = 0; sample_index < sample_count; sample_index += 2U)" in apply_pan
    assert "const float left = readFloat32Le(in.data, sample_index);" in apply_pan
    assert "const float right = readFloat32Le(in.data, sample_index + 1U);" in apply_pan
    assert "static_cast<double>(left) * left_gain_" in apply_pan
    assert "static_cast<double>(right) * right_gain_" in apply_pan
    assert "appendFloat32Le(static_cast<float>(panned_left), output_data);" in apply_pan
    assert "appendFloat32Le(static_cast<float>(panned_right), output_data);" in apply_pan
    assert "static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 3U)) << 24U" in read_float
    assert "std::memcpy(&sample, &raw, sizeof(float));" in read_float
    assert "std::memcpy(&raw, &sample, sizeof(float));" in append
    assert "std::isfinite(sample)" in range_check
    assert "sample >= kMinNormalizedSample" in range_check
    assert "sample <= kMaxNormalizedSample" in range_check
    assert "std::clamp" not in apply_pan


def test_pan_drops_invalid_runtime_samples_instead_of_clamping_or_normalizing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_pan_node.cpp").read_text(encoding="utf-8")
    apply_pan = source.split("bool FaPanNode::applyPan")[1].split(
        "float FaPanNode::readFloat32Le"
    )[0]

    assert "!isNormalizedFinite(left) || !isNormalizedFinite(right)" in apply_pan
    assert "!isFinite(panned_left) || !isFinite(panned_right)" in apply_pan
    assert "panned_left < kMinNormalizedSample" in apply_pan
    assert "panned_right > kMaxNormalizedSample" in apply_pan
    assert "Dropping frame because input sample is outside normalized FLOAT32LE range" in apply_pan
    assert "Dropping frame because pan output is outside normalized FLOAT32LE range" in apply_pan
    assert "return false;" in apply_pan
    assert "std::clamp" not in source
    assert "normalize(" not in source


def test_diagnostics_include_pan_gains_and_counters() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_pan_node.cpp").read_text(encoding="utf-8")
    diagnostics = source.split("void FaPanNode::publishDiagnostics")[1].split(
        "}  // namespace fa_pan"
    )[0]

    assert 'status.name = "fa_pan";' in diagnostics
    assert 'pushKeyValue(status, "input_topic", config_.input_topic);' in diagnostics
    assert 'pushKeyValue(status, "output_topic", config_.output_topic);' in diagnostics
    assert 'pushKeyValue(status, "input_stream_id", config_.input_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "output_stream_id", config_.output_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "pan.position", std::to_string(config_.pan_position));' in diagnostics
    assert 'pushKeyValue(status, "pan.left_gain", std::to_string(left_gain_));' in diagnostics
    assert 'pushKeyValue(status, "pan.right_gain", std::to_string(right_gain_));' in diagnostics
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
        "docs/backends/internal_constant_power_pan.md",
        "config/default.yaml",
        "launch/fa_pan.launch.py",
        "include/fa_pan/fa_pan_node.hpp",
        "src/fa_pan_node.cpp",
        "test/unit/test_fa_pan_audio_frame_contract.py",
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

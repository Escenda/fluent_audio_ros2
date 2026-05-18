from pathlib import Path

import yaml


def test_default_config_requires_explicit_float32le_beamforming_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_beamforming"]["ros__parameters"]

    assert params["input_topic"] == "fa_beamforming/input"
    assert params["output_topic"] == "fa_beamforming/output"
    assert params["input_stream_id"] == "audio/spatial/mic"
    assert params["output"]["stream_id"] == "audio/beamformed/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["beamforming"]["weights"] == [1.0, 0.0, 0.0, 0.0]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 4
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["output"]["channels"] == 1
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_launch_requires_explicit_node_name_and_config_file() -> None:
    package_root = Path(__file__).parents[2]
    launch_text = (package_root / "launch" / "fa_beamforming.launch.py").read_text(encoding="utf-8")

    assert "DeclareLaunchArgument(" in launch_text
    assert '"node_name"' in launch_text
    assert '"config_file"' in launch_text
    assert 'LaunchConfiguration("node_name")' in launch_text
    assert 'LaunchConfiguration("config_file")' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_beamforming_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaBeamformingNode::loadParameters")[1].split(
        "void FaBeamformingNode::declareRequiredParameter"
    )[0]
    declare_required = source.split("void FaBeamformingNode::declareRequiredParameter")[1].split(
        "std::string FaBeamformingNode::requireStringParameter"
    )[0]

    required_names = (
        "input_topic",
        "output_topic",
        "input_stream_id",
        "output.stream_id",
        "beamforming.weights",
        "output.channels",
        "expected.sample_rate",
        "expected.channels",
        "expected.encoding",
        "expected.bit_depth",
        "expected.layout",
        "qos.depth",
        "qos.reliable",
        "diagnostics.publish_period_ms",
        "diagnostics.qos.depth",
        "diagnostics.qos.reliable",
    )
    for name in required_names:
        assert f'declareRequiredParameter("{name}");' in load_parameters

    assert "this->declare_parameter(name, rclcpp::ParameterValue{});" in declare_required
    assert "declare_parameter(\"input_topic\", config_" not in source
    assert "declare_parameter(\"output_topic\", config_" not in source
    assert "declare_parameter<int>(\"output.channels\", config_" not in source
    assert "declare_parameter<bool>(\"qos.reliable\", config_" not in source


def test_beamforming_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_beamforming_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "PCM16LE",
        "PCM32LE",
        "resample",
        "average_to_mono",
        "pair_average_to_stereo",
        "equal_weight",
        "fallback",
        "std::clamp",
        "normalize(",
        "limiter",
        "denoise",
        ".vad",
        "source_separation",
    )
    for token in forbidden:
        assert token not in source


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_beamforming_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaBeamformingNode::loadParameters")[1].split(
        "void FaBeamformingNode::declareRequiredParameter"
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
    assert "config_.output_channels != 1" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "config_.diagnostics_qos_depth <= 0" in load_parameters
    assert "throw std::runtime_error" in load_parameters


def test_weight_validation_requires_explicit_finite_nonzero_channel_weights() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_beamforming_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaBeamformingNode::loadParameters")[1].split(
        "void FaBeamformingNode::declareRequiredParameter"
    )[0]
    double_array = source.split("std::vector<double> FaBeamformingNode::requireDoubleArrayParameter")[1].split(
        "void FaBeamformingNode::setupInterfaces"
    )[0]

    assert "config_.weights = requireDoubleArrayParameter(\"beamforming.weights\");" in load_parameters
    assert "config_.weights.size() != static_cast<size_t>(config_.expected_channels)" in load_parameters
    assert "for (const double weight : config_.weights)" in load_parameters
    assert "!std::isfinite(weight)" in load_parameters
    assert "weights_sum_abs_ += std::abs(weight);" in load_parameters
    assert "weights_sum_abs_ <= 0.0" in load_parameters
    assert "parameter.as_double_array()" in double_array
    assert "must be a double array parameter" in double_array


def test_beamforming_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_beamforming_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaBeamformingNode::validateFrame")[1].split(
        "bool FaBeamformingNode::beamformFrame"
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


def test_beamforming_outputs_mono_float32le_interleaved_stream() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_beamforming_node.cpp").read_text(encoding="utf-8")
    beamform_frame = source.split("bool FaBeamformingNode::beamformFrame")[1].split(
        "float FaBeamformingNode::readFloat32Le"
    )[0]

    assert "out = in;" in beamform_frame
    assert "out.stream_id = config_.output_stream_id;" in beamform_frame
    assert "out.stream_id = config_.output_topic;" not in beamform_frame
    assert "out.channels = static_cast<uint32_t>(config_.output_channels);" in beamform_frame
    assert "out.encoding = kEncodingFloat32;" in beamform_frame
    assert "out.bit_depth = 32;" in beamform_frame
    assert "out.layout = kInterleavedLayout;" in beamform_frame
    assert "out.data = output_data;" in beamform_frame
    assert "out.sample_rate =" not in beamform_frame
    assert "out.epoch =" not in beamform_frame


def test_static_delay_sum_algorithm_uses_explicit_weights() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_beamforming_node.cpp").read_text(encoding="utf-8")
    beamform_frame = source.split("bool FaBeamformingNode::beamformFrame")[1].split(
        "float FaBeamformingNode::readFloat32Le"
    )[0]
    read_float = source.split("float FaBeamformingNode::readFloat32Le")[1].split(
        "void FaBeamformingNode::appendFloat32Le"
    )[0]
    append = source.split("void FaBeamformingNode::appendFloat32Le")[1].split(
        "bool FaBeamformingNode::isNormalizedFinite"
    )[0]
    range_check = source.split("bool FaBeamformingNode::isNormalizedFinite")[1].split(
        "std::string FaBeamformingNode::formatWeights"
    )[0]

    assert "const size_t input_channels = static_cast<size_t>(config_.expected_channels);" in beamform_frame
    assert "for (size_t frame_index = 0; frame_index < frame_count; ++frame_index)" in beamform_frame
    assert "for (size_t channel_index = 0; channel_index < input_channels; ++channel_index)" in beamform_frame
    assert "const float sample = readFloat32Le(in.data, frame_sample_offset + channel_index);" in beamform_frame
    assert "weighted_sum += static_cast<double>(sample) * config_.weights.at(channel_index);" in beamform_frame
    assert "appendFloat32Le(static_cast<float>(weighted_sum), output_data);" in beamform_frame
    assert "static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 3U)) << 24U" in read_float
    assert "std::memcpy(&sample, &raw, sizeof(float));" in read_float
    assert "std::memcpy(&raw, &sample, sizeof(float));" in append
    assert "std::isfinite(sample)" in range_check
    assert "sample >= kMinNormalizedSample" in range_check
    assert "sample <= kMaxNormalizedSample" in range_check
    assert "config_.weights.size()" not in beamform_frame


def test_beamforming_drops_invalid_runtime_samples_instead_of_clamping_or_normalizing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_beamforming_node.cpp").read_text(encoding="utf-8")
    beamform_frame = source.split("bool FaBeamformingNode::beamformFrame")[1].split(
        "float FaBeamformingNode::readFloat32Le"
    )[0]

    assert "!isNormalizedFinite(sample)" in beamform_frame
    assert "!isNormalizedFinite(weighted_sum)" in beamform_frame
    assert "Dropping frame because input sample is outside normalized FLOAT32LE range" in beamform_frame
    assert "Dropping frame because beamforming output is outside normalized FLOAT32LE range" in beamform_frame
    assert "return false;" in beamform_frame
    assert "std::clamp" not in source
    assert "normalize(" not in source
    assert "weights_sum_abs_" not in beamform_frame


def test_diagnostics_include_beamforming_config_and_counters() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_beamforming_node.cpp").read_text(encoding="utf-8")
    diagnostics = source.split("void FaBeamformingNode::publishDiagnostics")[1].split(
        "}  // namespace fa_beamforming"
    )[0]

    assert 'status.name = "fa_beamforming";' in diagnostics
    assert 'pushKeyValue(status, "input_topic", config_.input_topic);' in diagnostics
    assert 'pushKeyValue(status, "output_topic", config_.output_topic);' in diagnostics
    assert 'pushKeyValue(status, "input_stream_id", config_.input_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "output_stream_id", config_.output_stream_id);' in diagnostics
    assert 'pushKeyValue(status, "beamforming.weights", formatWeights(config_.weights));' in diagnostics
    assert (
        'pushKeyValue(status, "beamforming.weights_sum_abs", '
        'std::to_string(weights_sum_abs_));'
    ) in diagnostics
    assert 'pushKeyValue(status, "expected.channels", std::to_string(config_.expected_channels));' in diagnostics
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
        "docs/backends/static_delay_sum.md",
        "config/default.yaml",
        "config/profiles/.gitkeep",
        "launch/fa_beamforming.launch.py",
        "include/fa_beamforming/fa_beamforming_node.hpp",
        "src/fa_beamforming_node.cpp",
        "test/unit/test_fa_beamforming_audio_frame_contract.py",
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
    assert "find_package(ament_lint_auto REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_lint_auto_find_test_dependencies()" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

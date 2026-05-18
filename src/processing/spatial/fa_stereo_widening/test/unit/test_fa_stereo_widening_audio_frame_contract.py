from pathlib import Path

import yaml


def _package_root() -> Path:
    return Path(__file__).parents[2]


def _source_text() -> str:
    return (_package_root() / "src" / "fa_stereo_widening_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_stereo_float32le_interleaved_contract() -> None:
    config = yaml.safe_load((_package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_stereo_widening"]["ros__parameters"]

    assert params["input_topic"] == "audio/spatial/mic"
    assert params["output_topic"] == "audio/stereo_widened/mic"
    assert params["width"] == 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 2
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_launch_requires_explicit_node_name_and_config_file() -> None:
    launch_text = (_package_root() / "launch" / "fa_stereo_widening.launch.py").read_text(
        encoding="utf-8"
    )

    assert "DeclareLaunchArgument(" in launch_text
    assert '"node_name"' in launch_text
    assert '"config_file"' in launch_text
    assert 'LaunchConfiguration("node_name")' in launch_text
    assert 'LaunchConfiguration("config_file")' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text


def test_stereo_widening_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    source = _source_text()

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "PCM16LE",
        "PCM32LE",
        "resample",
        "mono_to_stereo",
        "duplicate",
        "pan.position",
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
    source = _source_text()
    load_parameters = source.split("void FaStereoWideningNode::loadParameters")[1].split(
        "void FaStereoWideningNode::setupInterfaces"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "!std::isfinite(config_.width)" in load_parameters
    assert "config_.width < 0.0" in load_parameters
    assert "config_.width > kMaxWidth" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels != 2" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "throw std::runtime_error" in load_parameters


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    source = _source_text()
    load_parameters = source.split("void FaStereoWideningNode::loadParameters")[1].split(
        "void FaStereoWideningNode::setupInterfaces"
    )[0]

    assert 'readRequiredString(*this, "input_topic")' in load_parameters
    assert 'readRequiredString(*this, "output_topic")' in load_parameters
    assert 'readRequiredDouble(*this, "width")' in load_parameters
    assert 'readRequiredInt(*this, "expected.sample_rate")' in load_parameters
    assert 'readRequiredInt(*this, "expected.channels")' in load_parameters
    assert 'readRequiredString(*this, "expected.encoding")' in load_parameters
    assert 'readRequiredInt(*this, "expected.bit_depth")' in load_parameters
    assert 'readRequiredString(*this, "expected.layout")' in load_parameters
    assert 'readRequiredInt(*this, "qos.depth")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    assert "config_.diagnostics_publish_period_ms = readRequiredInt(" in load_parameters
    assert '"diagnostics.publish_period_ms"' in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert "config_." not in line


def test_stereo_widening_validates_frame_contract_before_processing() -> None:
    source = _source_text()
    validate_frame = source.split("bool FaStereoWideningNode::validateFrame")[1].split(
        "bool FaStereoWideningNode::applyStereoWidening"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_stereo_widening_preserves_metadata_and_updates_stream_data_only() -> None:
    source = _source_text()
    apply_widening = source.split("bool FaStereoWideningNode::applyStereoWidening")[1].split(
        "float FaStereoWideningNode::readFloat32Le"
    )[0]

    assert "out = in;" in apply_widening
    assert "out.stream_id = config_.output_topic;" in apply_widening
    assert "out.data = output_data;" in apply_widening
    assert "out.encoding =" not in apply_widening
    assert "out.bit_depth =" not in apply_widening
    assert "out.sample_rate =" not in apply_widening
    assert "out.channels =" not in apply_widening
    assert "out.layout =" not in apply_widening
    assert "out.epoch =" not in apply_widening


def test_mid_side_width_algorithm_and_normalized_sample_validation() -> None:
    source = _source_text()
    apply_widening = source.split("bool FaStereoWideningNode::applyStereoWidening")[1].split(
        "float FaStereoWideningNode::readFloat32Le"
    )[0]
    read_float = source.split("float FaStereoWideningNode::readFloat32Le")[1].split(
        "void FaStereoWideningNode::appendFloat32Le"
    )[0]
    append = source.split("void FaStereoWideningNode::appendFloat32Le")[1].split(
        "bool FaStereoWideningNode::isNormalizedFinite"
    )[0]
    range_check = source.split("bool FaStereoWideningNode::isNormalizedFinite")[1].split(
        "void FaStereoWideningNode::publishDiagnostics"
    )[0]

    assert "for (size_t sample_index = 0; sample_index < sample_count; sample_index += 2U)" in apply_widening
    assert "const float left = readFloat32Le(in.data, sample_index);" in apply_widening
    assert "const float right = readFloat32Le(in.data, sample_index + 1U);" in apply_widening
    assert "const double mid = (static_cast<double>(left) + static_cast<double>(right)) / 2.0;" in apply_widening
    assert "((static_cast<double>(left) - static_cast<double>(right)) / 2.0) * config_.width" in apply_widening
    assert "const double widened_left = mid + side;" in apply_widening
    assert "const double widened_right = mid - side;" in apply_widening
    assert "appendFloat32Le(static_cast<float>(widened_left), output_data);" in apply_widening
    assert "appendFloat32Le(static_cast<float>(widened_right), output_data);" in apply_widening
    assert "static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 3U)) << 24U" in read_float
    assert "std::memcpy(&sample, &raw, sizeof(float));" in read_float
    assert "std::memcpy(&raw, &sample, sizeof(float));" in append
    assert "out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));" in append
    assert "std::isfinite(sample)" in range_check
    assert "sample >= kMinNormalizedSample" in range_check
    assert "sample <= kMaxNormalizedSample" in range_check
    assert "std::clamp" not in apply_widening


def test_stereo_widening_drops_invalid_runtime_samples_instead_of_clamping_or_normalizing() -> None:
    source = _source_text()
    apply_widening = source.split("bool FaStereoWideningNode::applyStereoWidening")[1].split(
        "float FaStereoWideningNode::readFloat32Le"
    )[0]

    assert "!isNormalizedFinite(left) || !isNormalizedFinite(right)" in apply_widening
    assert "!isNormalizedFinite(widened_left) || !isNormalizedFinite(widened_right)" in apply_widening
    assert "Dropping frame because input sample is outside normalized FLOAT32LE range" in apply_widening
    assert (
        "Dropping frame because stereo widening output is outside normalized FLOAT32LE range"
        in apply_widening
    )
    assert "return false;" in apply_widening
    assert "std::clamp" not in source
    assert "normalize(" not in source


def test_diagnostics_include_width_and_counters() -> None:
    source = _source_text()
    diagnostics = source.split("void FaStereoWideningNode::publishDiagnostics")[1].split(
        "}  // namespace fa_stereo_widening"
    )[0]

    assert 'status.name = "fa_stereo_widening";' in diagnostics
    assert 'pushKeyValue(status, "width", std::to_string(config_.width));' in diagnostics
    assert 'pushKeyValue(status, "expected.channels", std::to_string(config_.expected_channels));' in diagnostics
    assert 'pushKeyValue(status, "frames.in", std::to_string(frames_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames.out", std::to_string(frames_out_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames.drop", std::to_string(frames_dropped_.load()));' in diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    package_root = _package_root()
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_mid_side_width.md",
        "config/default.yaml",
        "launch/fa_stereo_widening.launch.py",
        "include/fa_stereo_widening/fa_stereo_widening_node.hpp",
        "src/fa_stereo_widening_node.cpp",
        "test/unit/test_fa_stereo_widening_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    package_root = _package_root()
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

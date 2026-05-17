from pathlib import Path

import yaml


def test_default_config_requires_explicit_float32le_downmix_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_downmix"]["ros__parameters"]

    assert params["input_topic"] == "audio/spatial/mic"
    assert params["output_topic"] == "audio/downmixed/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["input_channels"] == 4
    assert params["output"]["channels"] == 2
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["mode"] == "pair_average_to_stereo"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_downmix_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_downmix_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "PCM16LE",
        "PCM32LE",
        "resample",
        "mono_to_stereo",
        "duplicate",
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


def test_startup_validation_fails_closed_for_invalid_or_non_downmix_config() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_downmix_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaDownmixNode::loadParameters")[1].split(
        "void FaDownmixNode::setupInterfaces"
    )[0]
    supported = source.split("bool FaDownmixNode::isSupportedDownmix")[1].split(
        "float FaDownmixNode::readFloat32Le"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_input_channels <= 0" in load_parameters
    assert "config_.output_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "!isSupportedDownmix(config_.mode, config_.expected_input_channels, config_.output_channels)" in source
    assert "throw std::runtime_error" in load_parameters
    assert "mode == kModeAverageToMono && input_channels > output_channels && output_channels == 1" in supported
    assert "mode == kModePairAverageToStereo && input_channels > output_channels" in supported
    assert "output_channels == 2 && (input_channels % 2) == 0" in supported


def test_downmix_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_downmix_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaDownmixNode::validateFrame")[1].split(
        "bool FaDownmixNode::downmixFrame"
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


def test_downmix_preserves_metadata_and_updates_stream_channels_data_only() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_downmix_node.cpp").read_text(encoding="utf-8")
    downmix_frame = source.split("bool FaDownmixNode::downmixFrame")[1].split(
        "bool FaDownmixNode::isSupportedDownmix"
    )[0]

    assert "out = in;" in downmix_frame
    assert "out.stream_id = config_.output_topic;" in downmix_frame
    assert "out.channels = static_cast<uint32_t>(config_.output_channels);" in downmix_frame
    assert "out.data = output_data;" in downmix_frame
    assert "out.encoding =" not in downmix_frame
    assert "out.bit_depth =" not in downmix_frame
    assert "out.sample_rate =" not in downmix_frame
    assert "out.layout =" not in downmix_frame
    assert "out.epoch =" not in downmix_frame


def test_downmix_algorithms_are_explicit_and_validate_normalized_samples() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_downmix_node.cpp").read_text(encoding="utf-8")
    downmix_frame = source.split("bool FaDownmixNode::downmixFrame")[1].split(
        "bool FaDownmixNode::isSupportedDownmix"
    )[0]
    read_float = source.split("float FaDownmixNode::readFloat32Le")[1].split(
        "void FaDownmixNode::appendFloat32Le"
    )[0]
    append = source.split("void FaDownmixNode::appendFloat32Le")[1].split(
        "bool FaDownmixNode::isNormalizedFinite"
    )[0]
    range_check = source.split("bool FaDownmixNode::isNormalizedFinite")[1].split(
        "void FaDownmixNode::publishDiagnostics"
    )[0]

    assert "config_.mode == kModeAverageToMono" in downmix_frame
    assert "sum += static_cast<double>(sample);" in downmix_frame
    assert "const double averaged = sum / static_cast<double>(input_channels);" in downmix_frame
    assert "appendFloat32Le(static_cast<float>(averaged), output_data);" in downmix_frame
    assert "config_.mode == kModePairAverageToStereo" in downmix_frame
    assert "const size_t channel_pairs = input_channels / 2U;" in downmix_frame
    assert "left_sum += static_cast<double>(left);" in downmix_frame
    assert "right_sum += static_cast<double>(right);" in downmix_frame
    assert "const double averaged_left = left_sum / static_cast<double>(channel_pairs);" in downmix_frame
    assert "const double averaged_right = right_sum / static_cast<double>(channel_pairs);" in downmix_frame
    assert "appendFloat32Le(static_cast<float>(averaged_left), output_data);" in downmix_frame
    assert "appendFloat32Le(static_cast<float>(averaged_right), output_data);" in downmix_frame
    assert "static_cast<uint32_t>(bytes.at((sample_index * sizeof(float)) + 3U)) << 24U" in read_float
    assert "std::memcpy(&sample, &raw, sizeof(float));" in read_float
    assert "std::memcpy(&raw, &sample, sizeof(float));" in append
    assert "out_bytes.push_back(static_cast<uint8_t>(raw & 0xFFU));" in append
    assert "std::isfinite(sample)" in range_check
    assert "sample >= kMinNormalizedSample" in range_check
    assert "sample <= kMaxNormalizedSample" in range_check


def test_downmix_drops_invalid_runtime_samples_instead_of_clamping_or_normalizing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_downmix_node.cpp").read_text(encoding="utf-8")
    downmix_frame = source.split("bool FaDownmixNode::downmixFrame")[1].split(
        "bool FaDownmixNode::isSupportedDownmix"
    )[0]

    assert "!isNormalizedFinite(sample)" in downmix_frame
    assert "!isNormalizedFinite(left) || !isNormalizedFinite(right)" in downmix_frame
    assert "!isNormalizedFinite(averaged)" in downmix_frame
    assert "!isNormalizedFinite(averaged_left) || !isNormalizedFinite(averaged_right)" in downmix_frame
    assert "Dropping frame because input sample is outside normalized FLOAT32LE range" in downmix_frame
    assert "Dropping frame because downmix output is outside normalized FLOAT32LE range" in downmix_frame
    assert "return false;" in downmix_frame
    assert "std::clamp" not in source
    assert "normalize(" not in source


def test_diagnostics_include_downmix_config_and_counters() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_downmix_node.cpp").read_text(encoding="utf-8")
    diagnostics = source.split("void FaDownmixNode::publishDiagnostics")[1].split(
        "}  // namespace fa_downmix"
    )[0]

    assert 'status.name = "fa_downmix";' in diagnostics
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
        "docs/backends/internal_downmix_matrix.md",
        "config/default.yaml",
        "launch/fa_downmix.launch.py",
        "include/fa_downmix/fa_downmix_node.hpp",
        "src/fa_downmix_node.cpp",
        "test/unit/test_fa_downmix_audio_frame_contract.py",
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

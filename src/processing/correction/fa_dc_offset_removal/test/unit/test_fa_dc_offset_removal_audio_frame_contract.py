from pathlib import Path

import yaml


def test_default_config_requires_float32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_dc_offset_removal"]["ros__parameters"]

    assert params["input_topic"] == "audio/sample_format/mic"
    assert params["output_topic"] == "audio/dc_offset_removed/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_dc_offset_removal_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_dc_offset_removal_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "convertPcm",
        "gain.linear",
        "threshold.linear",
        "filter.",
        "cutoff_hz",
        "center_hz",
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
    source = (package_root / "src" / "fa_dc_offset_removal_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaDcOffsetRemovalNode::loadParameters")[1].split(
        "void FaDcOffsetRemovalNode::setupInterfaces"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "throw std::runtime_error" in load_parameters


def test_dc_offset_removal_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_dc_offset_removal_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaDcOffsetRemovalNode::validateFrame")[1].split(
        "bool FaDcOffsetRemovalNode::removeDcOffset"
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


def test_dc_offset_removal_preserves_identity_and_updates_stream_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_dc_offset_removal_node.cpp").read_text(encoding="utf-8")
    remove_dc_offset = source.split("bool FaDcOffsetRemovalNode::removeDcOffset")[1].split(
        "void FaDcOffsetRemovalNode::publishDiagnostics"
    )[0]

    assert "out = in;" in remove_dc_offset
    assert "out.stream_id = config_.output_topic;" in remove_dc_offset
    assert "out.data.resize(in.data.size());" in remove_dc_offset
    assert "out.encoding =" not in remove_dc_offset
    assert "out.bit_depth =" not in remove_dc_offset
    assert "out.sample_rate =" not in remove_dc_offset
    assert "out.channels =" not in remove_dc_offset
    assert "out.layout =" not in remove_dc_offset


def test_dc_offset_algorithm_uses_per_frame_per_channel_mean_subtraction() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_dc_offset_removal_node.cpp").read_text(encoding="utf-8")
    remove_dc_offset = source.split("bool FaDcOffsetRemovalNode::removeDcOffset")[1].split(
        "void FaDcOffsetRemovalNode::publishDiagnostics"
    )[0]

    assert "const size_t channel_count = static_cast<size_t>(config_.expected_channels);" in remove_dc_offset
    assert "const size_t sample_count = in.data.size() / sizeof(float);" in remove_dc_offset
    assert "const size_t frame_count = sample_count / channel_count;" in remove_dc_offset
    assert "std::vector<double> channel_sums(channel_count, 0.0);" in remove_dc_offset
    assert "std::vector<float> samples(sample_count, 0.0F);" in remove_dc_offset
    assert "channel_sums.at(i % channel_count) += static_cast<double>(sample);" in remove_dc_offset
    assert "std::vector<double> channel_means(channel_count, 0.0);" in remove_dc_offset
    assert "channel_sums.at(channel) / static_cast<double>(frame_count)" in remove_dc_offset
    assert "static_cast<double>(samples.at(i)) - channel_means.at(i % channel_count)" in remove_dc_offset
    assert "std::memcpy(out.data.data() + (i * sizeof(float)), &out_sample, sizeof(float));" in remove_dc_offset


def test_dc_offset_removal_drops_non_finite_input_and_output_samples() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_dc_offset_removal_node.cpp").read_text(encoding="utf-8")
    remove_dc_offset = source.split("bool FaDcOffsetRemovalNode::removeDcOffset")[1].split(
        "void FaDcOffsetRemovalNode::publishDiagnostics"
    )[0]

    assert "!std::isfinite(sample)" in remove_dc_offset
    assert "!std::isfinite(mean)" in remove_dc_offset
    assert "!std::isfinite(corrected)" in remove_dc_offset
    assert "!std::isfinite(out_sample)" in remove_dc_offset
    assert "Dropping frame because input sample is not finite" in remove_dc_offset
    assert "Dropping frame because output sample is not finite" in remove_dc_offset
    assert "return false;" in remove_dc_offset


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_frame_mean.md",
        "config/default.yaml",
        "launch/fa_dc_offset_removal.launch.py",
        "include/fa_dc_offset_removal/fa_dc_offset_removal_node.hpp",
        "src/fa_dc_offset_removal_node.cpp",
        "test/unit/test_fa_dc_offset_removal_audio_frame_contract.py",
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

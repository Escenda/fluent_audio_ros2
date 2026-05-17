from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_declick_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_declick"]["ros__parameters"]

    assert params["input_topic"] == "audio/noise_gated/mic"
    assert params["output_topic"] == "audio/declicked/mic"
    assert params["threshold"]["delta"] == 0.25
    assert 0.0 < params["threshold"]["delta"] <= 2.0
    assert params["window"]["max_samples"] == 1
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_declick_does_not_hide_other_processing_or_io_responsibilities() -> None:
    source = read_source()

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "normalize(",
        "std::clamp",
        "gain.linear",
        "threshold.linear",
        "cutoff_hz",
        "center_hz",
        "denoise",
        "decrackle",
        "declip",
        "limiter",
        "reverb",
        "echo",
    )
    for token in forbidden:
        assert token not in source


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = read_source()
    load_parameters = source.split("void FaDeclickNode::loadParameters")[1].split(
        "void FaDeclickNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter<double>("threshold.delta", config_.threshold_delta);' in load_parameters
    assert 'this->declare_parameter<int>("window.max_samples", config_.window_max_samples);' in load_parameters
    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "!isFinite(config_.threshold_delta)" in load_parameters
    assert "config_.threshold_delta <= 0.0" in load_parameters
    assert "config_.threshold_delta > 2.0" in load_parameters
    assert "config_.window_max_samples <= 0" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "throw std::runtime_error" in load_parameters


def test_declick_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaDeclickNode::validateFrame")[1].split(
        "bool FaDeclickNode::validateSamples"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_declick_drops_invalid_samples_before_processing() -> None:
    source = read_source()
    handle_frame = source.split("void FaDeclickNode::handleFrame")[1].split(
        "bool FaDeclickNode::validateFrame"
    )[0]
    validate_samples = source.split("bool FaDeclickNode::validateSamples")[1].split(
        "bool FaDeclickNode::applyDeclick"
    )[0]

    assert "if (!validateSamples(*msg))" in handle_frame
    assert "if (!applyDeclick(*msg, out))" in handle_frame
    assert handle_frame.index("if (!validateSamples(*msg))") < handle_frame.index(
        "if (!applyDeclick(*msg, out))"
    )
    assert "readFloatSample(msg.data, sample_index)" in validate_samples
    assert "!isNormalizedSample(static_cast<double>(sample))" in validate_samples
    assert "return false;" in validate_samples
    assert "std::clamp" not in validate_samples


def test_declick_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_source()
    apply_declick = source.split("bool FaDeclickNode::applyDeclick")[1].split(
        "size_t FaDeclickNode::detectClickRun"
    )[0]

    assert "out = in;" in apply_declick
    assert "out.stream_id = config_.output_topic;" in apply_declick
    assert "out.data.resize(in.data.size());" in apply_declick
    assert "out.encoding =" not in apply_declick
    assert "out.bit_depth =" not in apply_declick
    assert "out.sample_rate =" not in apply_declick
    assert "out.channels =" not in apply_declick
    assert "out.layout =" not in apply_declick


def test_declick_algorithm_uses_per_channel_previous_current_next_delta_rule() -> None:
    source = read_source()
    detect_run = source.split("size_t FaDeclickNode::detectClickRun")[1].split(
        "size_t FaDeclickNode::bytesPerFrame"
    )[0]

    assert "const double delta = config_.threshold_delta;" in detect_run
    assert "const size_t max_window = std::min(max_click_samples_, frame_count - frame_index - 1);" in detect_run
    assert "sampleIndex(frame_index - 1, channel_index, channel_count)" in detect_run
    assert "sampleIndex(frame_index + run_length, channel_index, channel_count)" in detect_run
    assert "std::abs(static_cast<double>(previous) - static_cast<double>(next)) > delta" in detect_run
    assert "std::abs(static_cast<double>(current) - static_cast<double>(previous)) > delta" in detect_run
    assert "std::abs(static_cast<double>(current) - static_cast<double>(next)) > delta" in detect_run
    assert "return run_length;" in detect_run


def test_declick_replaces_click_run_with_average_of_boundaries() -> None:
    source = read_source()
    apply_declick = source.split("bool FaDeclickNode::applyDeclick")[1].split(
        "size_t FaDeclickNode::detectClickRun"
    )[0]

    assert "detectClickRun(" in apply_declick
    assert "const float previous = input_samples.at(" in apply_declick
    assert "const float next = input_samples.at(" in apply_declick
    assert "(static_cast<double>(previous) + static_cast<double>(next)) / 2.0" in apply_declick
    assert "output_samples.at(sampleIndex(frame_index + offset, channel_index, channel_count)) =" in apply_declick
    assert "corrected_samples += run_length;" in apply_declick
    assert "++corrected_runs;" in apply_declick
    assert "samples_corrected_.fetch_add(corrected_samples);" in apply_declick
    assert "click_runs_corrected_.fetch_add(corrected_runs);" in apply_declick


def test_declick_drops_invalid_output_instead_of_clamping_or_normalizing() -> None:
    source = read_source()
    apply_declick = source.split("bool FaDeclickNode::applyDeclick")[1].split(
        "size_t FaDeclickNode::detectClickRun"
    )[0]

    assert "if (!isNormalizedSample(corrected))" in apply_declick
    assert "if (!isNormalizedSample(static_cast<double>(corrected_sample)))" in apply_declick
    assert "Dropping frame because declick output is outside normalized FLOAT32LE range" in apply_declick
    assert "return false;" in apply_declick
    assert "std::clamp" not in apply_declick
    assert "normalize(" not in apply_declick


def test_diagnostics_publish_config_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaDeclickNode::publishDiagnostics")[1].split(
        "}  // namespace fa_declick"
    )[0]

    assert 'status.name = "fa_declick";' in diagnostics
    assert '"threshold_delta"' in diagnostics
    assert '"window_max_samples"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"samples_corrected"' in diagnostics
    assert '"click_runs_corrected"' in diagnostics
    assert '"output_topic"' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_impulse_declick.md",
        "config/default.yaml",
        "launch/fa_declick.launch.py",
        "include/fa_declick/fa_declick_node.hpp",
        "src/fa_declick_node.cpp",
        "test/unit/test_fa_declick_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

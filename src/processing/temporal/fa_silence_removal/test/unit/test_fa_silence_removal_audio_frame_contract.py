from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_silence_removal_node.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32_interleaved_rms_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_silence_removal"]["ros__parameters"]

    assert params["input_topic"] == "audio/buffered/mic"
    assert params["output_topic"] == "audio/silence_removed/mic"
    assert params["threshold"]["rms"] == 0.02
    assert 0.0 <= params["threshold"]["rms"] <= 1.0
    assert params["hangover_ms"] == 200.0
    assert params["hangover_ms"] >= 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_silence_removal_does_not_hide_io_or_other_processing_responsibilities() -> None:
    source = read_source()

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "normalize(",
        "std::clamp",
        "closed_gain",
        "gain.linear",
        "filter.",
        "denoise",
        "compress",
        "limiter",
        "fade",
        "window",
    )
    for token in forbidden:
        assert token not in source


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = read_source()
    load_parameters = source.split("void FaSilenceRemovalNode::loadParameters")[1].split(
        "void FaSilenceRemovalNode::setupInterfaces"
    )[0]

    assert 'this->declare_parameter<double>("threshold.rms", config_.threshold_rms);' in load_parameters
    assert 'this->declare_parameter<double>("hangover_ms", config_.hangover_ms);' in load_parameters
    assert "!isFinite(config_.threshold_rms)" in load_parameters
    assert "config_.threshold_rms < 0.0" in load_parameters
    assert "config_.threshold_rms > 1.0" in load_parameters
    assert "!isFinite(config_.hangover_ms) || config_.hangover_ms < 0.0" in load_parameters
    assert "threshold.rms must be finite and in [0.0, 1.0]" in load_parameters
    assert "hangover_ms must be finite and >= 0" in load_parameters
    assert "fa_silence_removal requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_silence_removal requires expected.bit_depth=32" in load_parameters
    assert "fa_silence_removal requires expected.layout=interleaved" in load_parameters
    assert "throw std::runtime_error" in load_parameters


def test_hangover_ms_converts_to_samples_from_expected_sample_rate() -> None:
    source = read_source()
    load_parameters = source.split("void FaSilenceRemovalNode::loadParameters")[1].split(
        "void FaSilenceRemovalNode::setupInterfaces"
    )[0]

    assert (
        "config_.hangover_ms * static_cast<double>(config_.expected_sample_rate) / "
        "kMillisecondsPerSecond"
        in load_parameters
    )
    assert "hangover_samples_ = static_cast<size_t>(std::ceil(raw_hangover_samples));" in load_parameters


def test_frame_contract_is_validated_before_rms_or_hangover_state() -> None:
    source = read_source()
    handle_frame = source.split("void FaSilenceRemovalNode::handleFrame")[1].split(
        "bool FaSilenceRemovalNode::validateFrame"
    )[0]
    validate_frame = source.split("bool FaSilenceRemovalNode::validateFrame")[1].split(
        "bool FaSilenceRemovalNode::computeRms"
    )[0]

    assert "if (!validateFrame(*msg))" in handle_frame
    assert "if (!computeRms(*msg, rms))" in handle_frame
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index(
        "if (!computeRms(*msg, rms))"
    )
    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_rms_algorithm_drops_invalid_samples_without_clamping() -> None:
    source = read_source()
    compute_rms = source.split("bool FaSilenceRemovalNode::computeRms")[1].split(
        "void FaSilenceRemovalNode::publishAcceptedFrame"
    )[0]

    assert "const size_t sample_count = msg.data.size() / sizeof(float);" in compute_rms
    assert "sum_squares += sample_value * sample_value;" in compute_rms
    assert "rms = std::sqrt(sum_squares / static_cast<double>(sample_count));" in compute_rms
    assert "!std::isfinite(sample)" in compute_rms
    assert "sample < kMinNormalizedSample" in compute_rms
    assert "sample > kMaxNormalizedSample" in compute_rms
    assert "return false;" in compute_rms
    assert "std::clamp" not in compute_rms
    assert "normalize(" not in compute_rms


def test_silent_frames_are_dropped_without_publish_and_hangover_keeps_chunks() -> None:
    source = read_source()
    handle_frame = source.split("void FaSilenceRemovalNode::handleFrame")[1].split(
        "bool FaSilenceRemovalNode::validateFrame"
    )[0]

    assert "if (rms >= config_.threshold_rms)" in handle_frame
    assert "hangover_samples_remaining_ = hangover_samples_;" in handle_frame
    assert "publishAcceptedFrame(*msg);" in handle_frame
    assert "if (hangover_samples_remaining_ > 0)" in handle_frame
    assert "consumeHangoverSamples(frameCount(*msg));" in handle_frame
    assert "silent_frames_dropped_.fetch_add(1);" in handle_frame
    assert "messages_dropped_.fetch_add(1);" in handle_frame
    silent_branch = handle_frame.split("silent_frames_dropped_.fetch_add(1);")[1]
    assert "audio_pub_->publish" not in silent_branch
    assert "publishAcceptedFrame" not in silent_branch


def test_published_frames_preserve_metadata_and_only_update_stream_identity() -> None:
    source = read_source()
    publish_frame = source.split("void FaSilenceRemovalNode::publishAcceptedFrame")[1].split(
        "void FaSilenceRemovalNode::consumeHangoverSamples"
    )[0]

    assert "fa_interfaces::msg::AudioFrame out = msg;" in publish_frame
    assert "out.stream_id = config_.output_topic;" in publish_frame
    assert "audio_pub_->publish(out);" in publish_frame
    assert ".rms" not in publish_frame
    assert ".peak" not in publish_frame
    assert ".vad" not in publish_frame
    assert "out.data" not in publish_frame


def test_diagnostics_include_threshold_hangover_and_drop_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaSilenceRemovalNode::publishDiagnostics")[1].split(
        "}  // namespace fa_silence_removal"
    )[0]

    assert 'status.name = "fa_silence_removal";' in diagnostics
    assert '"threshold_rms"' in diagnostics
    assert '"hangover_ms"' in diagnostics
    assert '"hangover_samples"' in diagnostics
    assert '"hangover_samples_remaining"' in diagnostics
    assert '"last_rms"' in diagnostics
    assert '"messages_in"' in diagnostics
    assert '"messages_out"' in diagnostics
    assert '"messages_dropped"' in diagnostics
    assert '"invalid_frames_dropped"' in diagnostics
    assert '"silent_frames_dropped"' in diagnostics
    assert '"hangover_frames"' in diagnostics
    assert '"active_frames"' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_rms_silence_removal.md",
        "config/default.yaml",
        "launch/fa_silence_removal.launch.py",
        "include/fa_silence_removal/fa_silence_removal_node.hpp",
        "src/fa_silence_removal_node.cpp",
        "test/unit/test_fa_silence_removal_audio_frame_contract.py",
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
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

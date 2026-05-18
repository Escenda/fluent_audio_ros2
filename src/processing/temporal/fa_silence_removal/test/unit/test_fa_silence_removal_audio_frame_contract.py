from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_silence_removal_node.cpp").read_text(
        encoding="utf-8"
    )


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_silence_removal"
        / "backends"
        / "internal_rms_silence_removal.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_rms_silence_removal.cpp"
    ).read_text(encoding="utf-8")


def test_example_config_requires_float32_interleaved_rms_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_silence_removal"]["ros__parameters"]

    assert params["input_topic"] == "fa_silence_removal/input"
    assert params["output_topic"] == "fa_silence_removal/output"
    assert params["input_stream_id"] == "audio/buffered/mic"
    assert params["output"]["stream_id"] == "audio/silence_removed/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
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


def test_launch_requires_explicit_config_file_without_package_default() -> None:
    launch_source = (
        package_root() / "launch" / "fa_silence_removal.launch.py"
    ).read_text(encoding="utf-8")
    config_argument = launch_source.split('DeclareLaunchArgument(\n            "config_file"')[1].split(
        "        ),",
        1,
    )[0]

    assert "default_value" not in launch_source
    assert "FindPackageShare" not in launch_source
    assert "PathJoinSubstitution" not in launch_source
    assert "config/default.yaml" not in launch_source
    assert "parameters=[config_file]" in launch_source


def test_silence_removal_does_not_hide_io_or_other_processing_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

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
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_validation_fails_closed_for_invalid_config_without_runtime_defaults() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaSilenceRemovalNode::loadParameters")[1].split(
        "void FaSilenceRemovalNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<double>("threshold.rms");' in load_parameters
    assert 'this->declare_parameter<double>("hangover_ms");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredDouble(*this, \"threshold.rms\")" in load_parameters
    assert "readRequiredDouble(*this, \"hangover_ms\")" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, config_.input_topic)" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, resolved_input_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, config_.output_topic)" in load_parameters
    assert "sameIdentityString(config_.output_stream_id, resolved_output_topic)" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "!isFinite(config_.threshold_rms)" in load_parameters
    assert "config_.threshold_rms < 0.0" in load_parameters
    assert "config_.threshold_rms > 1.0" in load_parameters
    assert "!isFinite(config_.hangover_ms) || config_.hangover_ms < 0.0" in load_parameters
    assert "threshold.rms must be finite and in [0.0, 1.0]" in load_parameters
    assert "hangover_ms must be finite and >= 0" in load_parameters
    assert "config_.expected_sample_rate > kMaxExpectedSampleRate" in load_parameters
    assert "config_.expected_channels > kMaxExpectedChannels" in load_parameters
    assert "fa_silence_removal requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_silence_removal requires expected.bit_depth=32" in load_parameters
    assert "fa_silence_removal requires expected.layout=interleaved" in load_parameters


def test_hangover_ms_converts_to_samples_from_expected_sample_rate() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaSilenceRemovalNode::loadParameters")[1].split(
        "void FaSilenceRemovalNode::configureBackend"
    )[0]
    configure_backend = source.split("void FaSilenceRemovalNode::configureBackend")[1].split(
        "void FaSilenceRemovalNode::setupInterfaces"
    )[0]

    assert (
        "config_.hangover_ms * static_cast<double>(config_.expected_sample_rate) / "
        "kMillisecondsPerSecond"
        in load_parameters
    )
    assert "hangover_samples_ = static_cast<size_t>(std::ceil(raw_hangover_samples));" in load_parameters
    assert "hangover_samples_" in configure_backend


def test_frame_contract_is_validated_before_backend() -> None:
    source = read_node_source()
    handle_frame = source.split("void FaSilenceRemovalNode::handleFrame")[1].split(
        "bool FaSilenceRemovalNode::validateFrame"
    )[0]
    validate_frame = source.split("bool FaSilenceRemovalNode::validateFrame")[1].split(
        "void FaSilenceRemovalNode::publishAcceptedFrame"
    )[0]

    assert "if (!validateFrame(*msg))" in handle_frame
    assert "backend_->process(msg->data)" in handle_frame
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index(
        "backend_->process(msg->data)"
    )
    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame


def test_rms_algorithm_and_hangover_state_are_backend_responsibilities() -> None:
    source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalRmsSilenceRemovalBackend::process")[1].split(
        "const char * decisionName"
    )[0]
    handle_frame = source.split("void FaSilenceRemovalNode::handleFrame")[1].split(
        "bool FaSilenceRemovalNode::validateFrame"
    )[0]

    assert "computeRms" not in source
    assert "consumeHangoverSamples" not in source
    assert "enum class Decision" in backend_header
    assert "sum_squares += sample_value * sample_value;" in process
    assert "std::sqrt(sum_squares / static_cast<double>(sample_count));" in process
    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "hangover_samples_remaining_ = config_.hangover_samples;" in process
    assert "consumeHangoverSamples(frame_count);" in process
    assert "backend_->process(msg->data)" in handle_frame
    assert "backends::decisionName(result.decision)" in handle_frame


def test_silent_frames_are_dropped_without_publish_and_hangover_keeps_chunks() -> None:
    source = read_node_source()
    handle_frame = source.split("void FaSilenceRemovalNode::handleFrame")[1].split(
        "bool FaSilenceRemovalNode::validateFrame"
    )[0]

    assert "case backends::Decision::kAcceptedActive:" in handle_frame
    assert "active_frames_.fetch_add(1);" in handle_frame
    assert "case backends::Decision::kAcceptedHangover:" in handle_frame
    assert "hangover_frames_.fetch_add(1);" in handle_frame
    assert "case backends::Decision::kDroppedSilent:" in handle_frame
    assert "silent_frames_dropped_.fetch_add(1);" in handle_frame
    assert "messages_dropped_.fetch_add(1);" in handle_frame
    silent_branch = handle_frame.split("case backends::Decision::kDroppedSilent:")[1]
    assert "audio_pub_->publish" not in silent_branch
    assert "publishAcceptedFrame" not in silent_branch


def test_published_frames_preserve_metadata_and_only_update_stream_identity() -> None:
    source = read_node_source()
    publish_frame = source.split("void FaSilenceRemovalNode::publishAcceptedFrame")[1].split(
        "size_t FaSilenceRemovalNode::bytesPerFrame"
    )[0]

    assert "fa_interfaces::msg::AudioFrame out = msg;" in publish_frame
    assert "out.stream_id = config_.output_stream_id;" in publish_frame
    assert "audio_pub_->publish(out);" in publish_frame
    assert ".rms" not in publish_frame
    assert ".peak" not in publish_frame
    assert ".vad" not in publish_frame
    assert "out.data" not in publish_frame


def test_diagnostics_include_threshold_hangover_stream_identity_and_drop_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaSilenceRemovalNode::publishDiagnostics")[1].split(
        "}  // namespace fa_silence_removal"
    )[0]

    assert 'status.name = "fa_silence_removal";' in diagnostics
    assert '"threshold_rms"' in diagnostics
    assert '"hangover_ms"' in diagnostics
    assert '"hangover_samples"' in diagnostics
    assert '"hangover_samples_remaining"' in diagnostics
    assert "backend_->hangoverSamples()" in diagnostics
    assert "backend_->hangoverSamplesRemaining()" in diagnostics
    assert "backend_->lastRms()" in diagnostics
    assert '"input_stream_id"' in diagnostics
    assert '"output_stream_id"' in diagnostics
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
        "include/fa_silence_removal/backends/internal_rms_silence_removal.hpp",
        "src/fa_silence_removal_node.cpp",
        "src/backends/internal_rms_silence_removal.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_rms_silence_removal_backend.cpp",
        "test/unit/test_fa_silence_removal_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_and_backend_gtest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

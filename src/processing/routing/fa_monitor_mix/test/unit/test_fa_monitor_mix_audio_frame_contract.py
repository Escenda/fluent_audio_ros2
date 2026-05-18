from pathlib import Path

import yaml


def _package_root() -> Path:
    return Path(__file__).parents[2]


def _source_text() -> str:
    return (_package_root() / "src" / "fa_monitor_mix_node.cpp").read_text(encoding="utf-8")


def test_default_config_and_launch_use_monitor_mix_contract() -> None:
    package_root = _package_root()
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_monitor_mix"]["ros__parameters"]
    launch = (package_root / "launch" / "fa_monitor_mix.launch.py").read_text(encoding="utf-8")

    assert params["input_topics"] == ["audio/program/frame", "audio/tts/frame"]
    assert params["input_stream_ids"] == ["audio/program/frame", "audio/tts/frame"]
    assert params["input_gains_db"] == [0.0]
    assert params["master_index"] == 0
    assert params["output_topic"] == "audio/monitor/frame"
    assert params["output"]["stream_id"] == "audio/monitor/frame"
    assert params["output"]["source_id"] == "monitor_mix"
    assert params["expected"]["sample_rate"] == 48000
    assert params["expected"]["channels"] == 2
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["max_frame_age_ms"] > 0
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
    assert "default_value" not in launch
    assert "FindPackageShare" not in launch
    assert "PathJoinSubstitution" not in launch


def test_startup_validation_fails_closed_for_required_parameters_and_format() -> None:
    source = _source_text()
    load_parameters = source.split("void FaMonitorMixNode::loadParameters")[1].split(
        "void FaMonitorMixNode::setupInterfaces"
    )[0]

    assert "readRequiredStringArray(*this, \"input_topics\")" in load_parameters
    assert "readRequiredStringArray(*this, \"input_stream_ids\")" in load_parameters
    assert "readRequiredDoubleArray(*this, \"input_gains_db\")" in load_parameters
    assert "readRequiredString(*this, \"output_topic\")" in load_parameters
    assert "readRequiredInt(*this, \"expected.sample_rate\")" in load_parameters
    assert "readRequiredBool(*this, \"qos.reliable\")" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "input_topics must contain at least one topic" in load_parameters
    assert "input_topics must not contain an empty topic" in load_parameters
    assert "input_stream_ids must match input_topics length" in load_parameters
    assert "input_stream_ids must not contain an empty stream_id" in load_parameters
    assert "resolved input_topics must be unique" in load_parameters
    assert "input_gains_db must be size 1 or match input_topics length" in load_parameters
    assert "master_index out of range" in load_parameters
    assert "output_topic is required" in load_parameters
    assert "output.stream_id is required" in load_parameters
    assert "output.source_id is required" in load_parameters
    assert "resolved output_topic must be distinct from input_topics" in load_parameters
    assert "expected.sample_rate must be > 0" in load_parameters
    assert "expected.channels must be > 0" in load_parameters
    assert "fa_monitor_mix requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_monitor_mix requires expected.bit_depth=32" in load_parameters
    assert "fa_monitor_mix requires expected.layout=interleaved" in load_parameters
    assert "max_frame_age_ms must be > 0" in load_parameters
    assert "qos.depth must be > 0" in load_parameters
    assert "diagnostics.publish_period_ms must be > 0" in load_parameters
    assert "input_gains_db must resolve to finite linear gains" in load_parameters


def test_runtime_frame_validation_is_strict_per_input_topic() -> None:
    source = _source_text()
    validate_frame = source.split("bool FaMonitorMixNode::validateFrame")[1].split(
        "bool FaMonitorMixNode::mixAndPublish"
    )[0]

    assert "msg.source_id.empty()" in validate_frame
    assert "msg.stream_id != expected_stream_id" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "backend_->validateFrameBytes(msg.data)" in validate_frame
    assert "backends::processStatusMessage(status)" in validate_frame
    assert "std::memcpy" not in validate_frame


def test_master_frame_requires_all_inputs_recent_and_same_byte_length() -> None:
    source = _source_text()
    mix_and_publish = source.split("bool FaMonitorMixNode::mixAndPublish")[1].split(
        "double FaMonitorMixNode::gainDbForIndex"
    )[0]

    assert "if (!frames[i])" in mix_and_publish
    assert "missing_frame_drops_.fetch_add(1);" in mix_and_publish
    assert "age_ms < 0 || age_ms > config_.max_frame_age_ms" in mix_and_publish
    assert "stale_frame_drops_.fetch_add(1);" in mix_and_publish
    assert "backend_->mix(input_bytes)" in mix_and_publish
    assert "backends::ProcessStatus::kByteLengthMismatch" in mix_and_publish
    assert "return false;" in mix_and_publish
    assert "std::min" not in mix_and_publish


def test_monitor_mix_delegates_sample_mixing_to_backend() -> None:
    source = _source_text()
    mix_and_publish = source.split("bool FaMonitorMixNode::mixAndPublish")[1].split(
        "double FaMonitorMixNode::gainDbForIndex"
    )[0]

    assert "input_bytes[static_cast<size_t>(config_.master_index)] = master_frame.data;" in mix_and_publish
    assert "input_bytes[i] = frames[i]->data;" in mix_and_publish
    assert "const backends::ProcessResult result = backend_->mix(input_bytes);" in mix_and_publish
    assert "backends::ProcessStatus::kOutOfRangeOutput" in mix_and_publish
    assert "std::memcpy" not in mix_and_publish
    hidden_clamp_token = "std::" + "clamp"
    assert hidden_clamp_token not in source


def test_backend_sums_float32_samples_and_drops_out_of_range_output() -> None:
    package_root = _package_root()
    backend_source = (
        package_root / "src" / "backends" / "internal_monitor_mix.cpp"
    ).read_text(encoding="utf-8")

    assert "sample *= master_gain;" in backend_source
    assert "const float output = mixed[sample_index] + (input_samples[sample_index] * gain);" in backend_source
    assert "!std::isfinite(output)" in backend_source
    assert "output < kMinNormalizedSample" in backend_source
    assert "output > kMaxNormalizedSample" in backend_source
    assert "mixed output sample is outside normalized FLOAT32LE range" in backend_source


def test_output_identity_comes_from_master_and_monitor_config() -> None:
    source = _source_text()
    mix_and_publish = source.split("bool FaMonitorMixNode::mixAndPublish")[1].split(
        "double FaMonitorMixNode::gainDbForIndex"
    )[0]

    assert "out.header = master_frame.header;" in mix_and_publish
    assert "out.source_id = config_.output_source_id;" in mix_and_publish
    assert "out.stream_id = config_.output_stream_id;" in mix_and_publish
    assert "out.encoding = config_.expected_encoding;" in mix_and_publish
    assert "out.sample_rate = static_cast<uint32_t>(config_.expected_sample_rate);" in mix_and_publish
    assert "out.channels = static_cast<uint32_t>(config_.expected_channels);" in mix_and_publish
    assert "out.bit_depth = static_cast<uint32_t>(config_.expected_bit_depth);" in mix_and_publish
    assert "out.layout = config_.expected_layout;" in mix_and_publish
    assert "out.epoch = master_frame.epoch;" in mix_and_publish


def test_monitor_mix_does_not_take_over_other_processing_responsibilities() -> None:
    source = _source_text()
    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "convert",
        "applyLimiter",
        "applyCompressor",
        "applyNormalize",
        "low_pass",
        "high_pass",
        "denoise",
        "silence",
        "std::" + "clamp",
    )

    for token in forbidden:
        assert token not in source


def test_diagnostics_publish_required_counters() -> None:
    source = _source_text()
    publish_diagnostics = source.split("void FaMonitorMixNode::publishDiagnostics")[1].split(
        "}  // namespace fa_monitor_mix"
    )[0]

    for key in (
        "frames_in",
        "frames_valid",
        "frames_dropped",
        "mix_frames_out",
        "mix_frames_dropped",
        "stale_frame_drops",
        "missing_frame_drops",
        "range_drops",
    ):
        assert f'pushKeyValue(status, "{key}",' in publish_diagnostics
    assert 'pushKeyValue(status, "backend.name",' in publish_diagnostics
    assert 'pushKeyValue(status, "output.stream_id",' in publish_diagnostics


def test_cmake_builds_node_core_and_backend_contract() -> None:
    package_root = _package_root()
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")
    node_source = _source_text()
    main_source = (package_root / "src" / "main.cpp").read_text(encoding="utf-8")

    assert "add_library(fa_monitor_mix_internal_monitor_mix STATIC" in cmake_text
    assert "src/backends/internal_monitor_mix.cpp" in cmake_text
    assert "add_library(fa_monitor_mix_node_core" in cmake_text
    assert "target_link_libraries(fa_monitor_mix_node_core" in cmake_text
    assert "fa_monitor_mix_internal_monitor_mix" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_contract_test" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "int main(" not in node_source
    assert "int main(" in main_source

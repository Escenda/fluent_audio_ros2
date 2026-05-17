from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_trim_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_trim"]["ros__parameters"]

    assert params["input_topic"] == "audio/windowed/mic"
    assert params["output_topic"] == "audio/trimmed/mic"
    assert params["trim"]["leading_frames"] == 16
    assert params["trim"]["trailing_frames"] == 16
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_trim_does_not_hide_other_processing_or_io_responsibilities() -> None:
    source = read_source()

    forbidden = (
        "std::clamp",
        "clip",
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "gain.linear",
        "threshold.linear",
        "cutoff_hz",
        "center_hz",
        "denoise",
        "reverb",
        "echo",
        "fade",
        "window_type",
    )
    for token in forbidden:
        assert token not in source


def test_parameters_are_required_without_runtime_defaults_and_range_checked() -> None:
    source = read_source()
    load_parameters = source.split("void FaTrimNode::loadParameters")[1].split(
        "void FaTrimNode::setupInterfaces"
    )[0]

    assert 'declareRequiredParameter<std::string>(*this, "input_topic")' in load_parameters
    assert 'declareRequiredParameter<std::string>(*this, "output_topic")' in load_parameters
    assert 'declareRequiredParameter<int>(*this, "trim.leading_frames")' in load_parameters
    assert 'declareRequiredParameter<int>(*this, "trim.trailing_frames")' in load_parameters
    assert 'declareRequiredParameter<int>(*this, "expected.sample_rate")' in load_parameters
    assert 'declareRequiredParameter<int>(*this, "expected.channels")' in load_parameters
    assert 'declareRequiredParameter<std::string>(*this, "expected.encoding")' in load_parameters
    assert 'declareRequiredParameter<int>(*this, "expected.bit_depth")' in load_parameters
    assert 'declareRequiredParameter<std::string>(*this, "expected.layout")' in load_parameters
    assert 'declareRequiredParameter<int>(*this, "qos.depth")' in load_parameters
    assert 'declareRequiredParameter<bool>(*this, "qos.reliable")' in load_parameters
    assert (
        'declareRequiredParameter<int>(*this, "diagnostics.publish_period_ms")'
        in load_parameters
    )
    assert 'declare_parameter<int>("trim.leading_frames",' not in source
    assert 'declare_parameter<int>("trim.trailing_frames",' not in source
    assert 'declare_parameter<bool>("qos.reliable",' not in source

    assert "config_.leading_frames < 0" in load_parameters
    assert "trim.leading_frames must be >= 0" in load_parameters
    assert "config_.trailing_frames < 0" in load_parameters
    assert "trim.trailing_frames must be >= 0" in load_parameters
    assert "config_.leading_frames == 0 && config_.trailing_frames == 0" in load_parameters
    assert "at least one of trim.leading_frames or trim.trailing_frames must be > 0" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters


def test_runtime_validates_audio_frame_contract_before_trimming() -> None:
    source = read_source()
    validate_frame = source.split("bool FaTrimNode::validateFrame")[1].split(
        "bool FaTrimNode::validateSamples"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "contract_drops_.fetch_add(1);" in validate_frame


def test_invalid_samples_are_dropped_without_clamping_or_normalizing() -> None:
    source = read_source()
    handle_frame = source.split("void FaTrimNode::handleFrame")[1].split(
        "bool FaTrimNode::validateFrame"
    )[0]
    validate_samples = source.split("bool FaTrimNode::validateSamples")[1].split(
        "bool FaTrimNode::trimFrame"
    )[0]

    assert "if (!validateSamples(*msg))" in handle_frame
    assert "!std::isfinite(sample)" in validate_samples
    assert "sample < kMinNormalizedSample" in validate_samples
    assert "sample > kMaxNormalizedSample" in validate_samples
    assert "invalid_sample_drops_.fetch_add(1);" in validate_samples
    assert "return false;" in validate_samples
    assert "std::clamp" not in validate_samples
    assert "normalize(" not in validate_samples


def test_trim_preserves_declared_metadata_updates_stream_epoch_and_payload() -> None:
    source = read_source()
    trim_frame = source.split("bool FaTrimNode::trimFrame")[1].split(
        "size_t FaTrimNode::bytesPerFrame"
    )[0]

    assert "const size_t frame_count = in.data.size() / bytesPerFrame();" in trim_frame
    assert "const size_t output_frame_count = frame_count - leading_frames - trailing_frames;" in trim_frame
    assert "const size_t start_byte = leading_frames * bytesPerFrame();" in trim_frame
    assert "const size_t byte_count = output_frame_count * bytesPerFrame();" in trim_frame
    assert "out = in;" in trim_frame
    assert "out.stream_id = config_.output_topic;" in trim_frame
    assert "out.epoch = in.epoch + 1U;" in trim_frame
    assert "out.data.assign(" in trim_frame
    assert "start_byte + byte_count" in trim_frame
    assert "out.source_id" not in trim_frame
    assert "out.header" not in trim_frame
    assert "out.encoding" not in trim_frame
    assert "out.sample_rate" not in trim_frame
    assert "out.channels" not in trim_frame
    assert "out.bit_depth" not in trim_frame
    assert "out.layout" not in trim_frame


def test_trim_drops_frames_that_would_be_empty() -> None:
    source = read_source()
    trim_frame = source.split("bool FaTrimNode::trimFrame")[1].split(
        "size_t FaTrimNode::bytesPerFrame"
    )[0]
    handle_frame = source.split("void FaTrimNode::handleFrame")[1].split(
        "bool FaTrimNode::validateFrame"
    )[0]

    assert "leading_frames >= frame_count" in trim_frame
    assert "trailing_frames >= (frame_count - leading_frames)" in trim_frame
    assert "trim_exhausted_drops_.fetch_add(1);" in trim_frame
    assert "Dropping frame because trim removes all sample frames" in trim_frame
    assert "return false;" in trim_frame
    assert "if (!trimFrame(*msg, out))" in handle_frame
    assert "audio_pub_->publish(out);" in handle_frame
    assert handle_frame.index("if (!trimFrame(*msg, out))") < handle_frame.index(
        "audio_pub_->publish(out);"
    )


def test_epoch_increment_wrap_is_dropped() -> None:
    source = read_source()
    trim_frame = source.split("bool FaTrimNode::trimFrame")[1].split(
        "size_t FaTrimNode::bytesPerFrame"
    )[0]

    assert "in.epoch == std::numeric_limits<uint32_t>::max()" in trim_frame
    assert "epoch_overflow_drops_.fetch_add(1);" in trim_frame
    assert "Dropping frame because epoch increment would wrap uint32" in trim_frame
    assert "out.epoch = in.epoch + 1U;" in trim_frame


def test_diagnostics_include_trim_policy_and_drop_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaTrimNode::publishDiagnostics")[1].split(
        "}  // namespace fa_trim"
    )[0]

    assert 'status.name = "fa_trim";' in diagnostics
    assert 'pushKeyValue(status, "leading_frames", std::to_string(config_.leading_frames));' in diagnostics
    assert 'pushKeyValue(status, "trailing_frames", std::to_string(config_.trailing_frames));' in diagnostics
    assert 'pushKeyValue(status, "last_input_frame_count",' in diagnostics
    assert 'pushKeyValue(status, "last_output_frame_count",' in diagnostics
    assert 'pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));' in diagnostics
    assert 'pushKeyValue(status, "contract_drops", std::to_string(contract_drops_.load()));' in diagnostics
    assert (
        'pushKeyValue(status, "invalid_sample_drops", '
        "std::to_string(invalid_sample_drops_.load()));"
    ) in diagnostics
    assert (
        'pushKeyValue(status, "trim_exhausted_drops", '
        "std::to_string(trim_exhausted_drops_.load()));"
    ) in diagnostics
    assert (
        'pushKeyValue(status, "epoch_overflow_drops", '
        "std::to_string(epoch_overflow_drops_.load()));"
    ) in diagnostics


def test_docs_record_epoch_and_header_contract() -> None:
    spec = (package_root() / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package_root() / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )

    assert "出力 `epoch` は `input.epoch + 1`" in spec
    assert "`source_id`、`header`" in spec
    assert "`UINT32_MAX` の入力は wrap を避けるため drop" in spec
    assert "`header` は入力 frame の capture boundary として保持" in algorithm


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "CMakeLists.txt",
        "package.xml",
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/no_runtime_backend.md",
        "config/default.yaml",
        "config/profiles/.gitkeep",
        "launch/fa_trim.launch.py",
        "include/fa_trim/fa_trim_node.hpp",
        "include/fa_trim/backends/.gitkeep",
        "src/fa_trim_node.cpp",
        "src/backends/.gitkeep",
        "test/unit/test_fa_trim_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_contracts_and_lint_dependencies() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "find_package(ament_lint_auto REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_lint_auto_find_test_dependencies()" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

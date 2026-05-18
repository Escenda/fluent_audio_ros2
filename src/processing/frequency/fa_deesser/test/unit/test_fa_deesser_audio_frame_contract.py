from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_deesser_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_deesser"
        / "backends"
        / "internal_split_band_deesser.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_split_band_deesser.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_deesser"]["ros__parameters"]

    assert params["input_topic"] == "audio/normalized/mic"
    assert params["output_topic"] == "audio/deessed/mic"
    assert params["detector"]["cutoff_hz"] == 4500.0
    assert params["detector"]["threshold"] == 0.08
    assert params["detector"]["attenuation_db"] == -9.0
    assert 0.0 < params["detector"]["cutoff_hz"] < params["expected"]["sample_rate"] / 2
    assert 0.0 <= params["detector"]["threshold"] <= 1.0
    assert -120.0 <= params["detector"]["attenuation_db"] <= 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is True


def test_deesser_does_not_hide_io_or_other_processing_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "normalize(",
        "std::clamp",
        "denoise",
        "compress",
        "limiter",
        "VAD",
        "ASR",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_rejects_invalid_config_without_fallback() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaDeesserNode::loadParameters")[1].split(
        "void FaDeesserNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<double>("detector.cutoff_hz", config_.cutoff_hz);' in load_parameters
    assert 'this->declare_parameter<double>("detector.threshold", config_.threshold);' in load_parameters
    assert 'this->declare_parameter<double>("detector.attenuation_db", config_.attenuation_db);' in load_parameters
    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "const double nyquist_hz" in load_parameters
    assert "!isFinite(config_.cutoff_hz)" in load_parameters
    assert "config_.cutoff_hz <= 0.0" in load_parameters
    assert "config_.cutoff_hz >= nyquist_hz" in load_parameters
    assert "!isFinite(config_.threshold)" in load_parameters
    assert "config_.threshold < 0.0" in load_parameters
    assert "config_.threshold > 1.0" in load_parameters
    assert "!isFinite(config_.attenuation_db)" in load_parameters
    assert "config_.attenuation_db < kMinimumAttenuationDb" in load_parameters
    assert "config_.attenuation_db > 0.0" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "requires expected.encoding=FLOAT32LE" in load_parameters
    assert "requires expected.bit_depth=32" in load_parameters
    assert "requires expected.layout=interleaved" in load_parameters

    setup_interfaces = source.split("void FaDeesserNode::setupInterfaces")[1].split(
        "void FaDeesserNode::handleFrame"
    )[0]
    assert "if (config_.qos_reliable)" in setup_interfaces
    assert "qos.reliable();" in setup_interfaces
    assert "qos.best_effort();" in setup_interfaces


def test_deesser_validates_frame_contract_before_processing() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaDeesserNode::validateFrame")[1].split(
        "bool FaDeesserNode::applyDeesser"
    )[0]
    handle_frame = source.split("void FaDeesserNode::handleFrame")[1].split(
        "bool FaDeesserNode::validateFrame"
    )[0]

    assert "if (!msg)" in handle_frame
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_deesser_preserves_frame_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_deesser = source.split("bool FaDeesserNode::applyDeesser")[1].split(
        "void FaDeesserNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_deesser
    assert "out.stream_id = config_.output_topic;" in apply_deesser
    assert ".rms" not in apply_deesser
    assert ".peak" not in apply_deesser
    assert ".vad" not in apply_deesser


def test_deesser_algorithm_uses_backend_split_band_threshold_and_recombine() -> None:
    header = read_backend_header()
    source = read_backend_source()
    process = source.split("ProcessResult InternalSplitBandDeesserBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "class InternalSplitBandDeesserBackend" in header
    assert "std::vector<double> low_band_state_" in header
    assert "struct ProcessResult" in header
    assert "uint64_t samples_attenuated" in header
    assert "alpha_ =" in source
    assert "1.0 - std::exp" in source
    assert "std::vector<double> next_low_band_state =" in process
    assert "double & channel_low_state =" in process
    assert "const double low_band = channel_low_state + (alpha_ *" in process
    assert "const double high_band = input_sample - low_band;" in process
    assert "channel_low_state = low_band;" in process
    assert "double processed_high_band = high_band;" in process
    assert "if (std::abs(high_band) >= config_.threshold)" in process
    assert "processed_high_band = high_band * attenuation_gain_;" in process
    assert "const double output_sample = low_band + processed_high_band;" in process
    assert "low_band_state_ = std::move(next_low_band_state);" in process
    assert "std::clamp" not in process


def test_deesser_resets_per_channel_filter_state_on_source_change() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    apply_deesser = source.split("bool FaDeesserNode::applyDeesser")[1].split(
        "void FaDeesserNode::publishDiagnostics"
    )[0]
    process = backend_source.split("ProcessResult InternalSplitBandDeesserBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "const bool source_changed = !active_source_id_.empty() &&" in apply_deesser
    assert "in.source_id != active_source_id_" in apply_deesser
    assert "backend_->process(in.data, out.data, source_changed)" in apply_deesser
    assert "active_source_id_ = in.source_id;" in apply_deesser
    assert "filter_resets_.fetch_add(1);" in apply_deesser
    assert "reset_state ?" in process
    assert "std::vector<double>(static_cast<size_t>(config_.channels), 0.0)" in process
    assert "low_band_state_;" in process


def test_deesser_drops_invalid_samples_and_outputs_instead_of_coercing() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalSplitBandDeesserBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(output_sample)" in process
    assert "output_sample < kNormalizedMin || output_sample > kNormalizedMax" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeInput, 0};" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeOutput, 0};" in process
    assert "std::clamp" not in process
    assert "normalize(" not in process
    assert "backends::processStatusMessage(result.status)" in node_source


def test_deesser_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "enum class ProcessStatus" in header
    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "return ProcessResult{ProcessStatus::kMisalignedInput, 0};" in backend_source
    assert "std::numeric_limits<float>::max()" in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_diagnostics_publish_config_state_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaDeesserNode::publishDiagnostics")[1].split(
        "}  // namespace fa_deesser"
    )[0]

    assert 'status.name = "fa_deesser";' in diagnostics
    assert '"detector_cutoff_hz"' in diagnostics
    assert '"detector_alpha"' in diagnostics
    assert '"detector_threshold"' in diagnostics
    assert '"detector_attenuation_db"' in diagnostics
    assert '"detector_attenuation_gain"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"samples_attenuated"' in diagnostics
    assert '"filter_resets"' in diagnostics
    assert '"output_topic"' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_split_band_deesser.md",
        "config/default.yaml",
        "launch/fa_deesser.launch.py",
        "include/fa_deesser/fa_deesser_node.hpp",
        "include/fa_deesser/backends/internal_split_band_deesser.hpp",
        "src/fa_deesser_node.cpp",
        "src/backends/internal_split_band_deesser.cpp",
        "test/cpp/test_internal_split_band_deesser_backend.cpp",
        "test/unit/test_fa_deesser_audio_frame_contract.py",
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
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

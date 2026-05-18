from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")


def read_node_header() -> str:
    return (
        package_root() / "include" / "fa_normalize" / "fa_normalize_node.hpp"
    ).read_text(encoding="utf-8")


def read_main_source() -> str:
    return (package_root() / "src" / "main.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_normalize"
        / "backends"
        / "internal_peak_normalize.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_peak_normalize.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_normalize"]["ros__parameters"]

    assert params["input_topic"] == "audio/noise_gated/mic"
    assert params["output_topic"] == "audio/normalized/mic"
    assert params["normalize"]["target_peak_linear"] == 0.9
    assert params["normalize"]["silence_threshold_linear"] == 0.0001
    assert 0.0 < params["normalize"]["target_peak_linear"] <= 1.0
    assert 0.0 <= params["normalize"]["silence_threshold_linear"]
    assert params["normalize"]["silence_threshold_linear"] < params["normalize"]["target_peak_linear"]
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_normalize_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "std::clamp",
        "compress",
        "limiter",
        "limit",
        "gate.",
        "filter.",
        "denoise",
        "lufs",
        "loudness",
        "legacy",
        "compat",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_rejects_invalid_config_without_fallback() -> None:
    node_source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = node_source.split("void FaNormalizeNode::loadParameters")[1].split(
        "void FaNormalizeNode::configureBackend"
    )[0]

    assert (
        'this->declare_parameter<double>("normalize.target_peak_linear", '
        "config_.target_peak_linear);"
    ) in load_parameters
    assert (
        'this->declare_parameter<double>("normalize.silence_threshold_linear", '
        "config_.silence_threshold_linear);"
    ) in load_parameters
    assert "!isFinite(config_.target_peak_linear)" in load_parameters
    assert "config_.target_peak_linear <= 0.0" in load_parameters
    assert "config_.target_peak_linear > 1.0" in load_parameters
    assert "!isFinite(config_.silence_threshold_linear)" in load_parameters
    assert "config_.silence_threshold_linear < 0.0" in load_parameters
    assert "config_.silence_threshold_linear >= config_.target_peak_linear" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "requires expected.encoding=FLOAT32LE" in load_parameters
    assert "requires expected.bit_depth=32" in load_parameters
    assert "requires expected.layout=interleaved" in load_parameters
    assert "config_.channels <= 0" in backend_source
    assert "config_.target_peak_linear <= 0.0" in backend_source
    assert "config_.target_peak_linear > 1.0" in backend_source
    assert "config_.silence_threshold_linear >= config_.target_peak_linear" in backend_source


def test_normalize_validates_frame_contract_before_processing() -> None:
    header = read_node_header()
    source = read_node_source()
    handle_frame = source.split("void FaNormalizeNode::handleFrame")[1].split(
        "bool FaNormalizeNode::validateFrame"
    )[0]
    validate_frame = source.split("bool FaNormalizeNode::validateFrame")[1].split(
        "bool FaNormalizeNode::applyNormalize"
    )[0]

    assert "explicit FaNormalizeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());" in header
    assert ': rclcpp::Node("fa_normalize", options)' in source
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


def test_normalize_preserves_frame_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_normalize = source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_normalize
    assert "out.stream_id = config_.output_topic;" in apply_normalize
    assert "backend_->process(in.data, out.data)" in apply_normalize
    assert ".rms" not in apply_normalize
    assert ".vad" not in apply_normalize
    assert ".epoch" not in apply_normalize


def test_peak_normalize_algorithm_is_backend_owned() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalPeakNormalizeBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_normalize = node_source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "class InternalPeakNormalizeBackend" in header
    assert "enum class ProcessMode" in header
    assert "struct ProcessResult" in header
    assert "float peak = 0.0F;" in process
    assert "peak = std::max(peak, std::abs(sample));" in process
    assert "config_.target_peak_linear / static_cast<double>(peak)" in process
    assert "const double normalized = static_cast<double>(samples[i]) * gain;" in process
    assert "backend_->process(in.data, out.data)" in apply_normalize
    assert "frames_normalized_.fetch_add(1);" in apply_normalize
    assert "last_gain_.store(result.gain);" in apply_normalize
    assert "std::max(peak, std::abs(sample))" not in node_source
    assert "std::memcpy" not in node_source
    assert "std::clamp" not in process


def test_silence_pass_through_changes_only_stream_identity_and_does_not_amplify() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalPeakNormalizeBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_normalize = node_source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "peak < static_cast<float>(config_.silence_threshold_linear)" in process
    assert "output = input;" in process
    assert "ProcessMode::kSilencePassthrough" in process
    assert "frames_silence_passthrough_.fetch_add(1);" in apply_normalize
    assert "last_gain_.store(1.0);" in apply_normalize


def test_normalize_drops_invalid_samples_and_outputs_instead_of_clamping() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalPeakNormalizeBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(gain)" in process
    assert "!isFinite(normalized)" in process
    assert "normalized < kNormalizedMin || normalized > kNormalizedMax" in process
    assert "ProcessStatus::kNonFiniteInput" in process
    assert "ProcessStatus::kOutOfRangeInput" in process
    assert "ProcessStatus::kNonFiniteGain" in process
    assert "ProcessStatus::kOutOfRangeOutput" in process
    assert "std::clamp" not in process


def test_normalize_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "enum class ProcessStatus" in header
    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteGain" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "ProcessStatus::kMisalignedInput" in backend_source
    assert "ProcessStatus::kNonFiniteInput" in backend_source
    assert "ProcessStatus::kOutOfRangeInput" in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_rejected_frame_does_not_overwrite_output_or_commit_last_gain() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalPeakNormalizeBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_normalize = node_source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    rejection_section = process.split("output = std::move(next_output);")[0]
    assert "std::vector<uint8_t> next_output(input.size());" in rejection_section
    assert "ProcessStatus::kOutOfRangeInput" in rejection_section
    assert "ProcessStatus::kOutOfRangeOutput" in rejection_section
    assert "output = std::move(next_output);" not in rejection_section
    assert "last_gain_.store" not in apply_normalize.split("return false;")[0]
    assert "last_gain_.store(result.gain);" in apply_normalize


def test_diagnostics_publish_config_last_gain_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaNormalizeNode::publishDiagnostics")[1].split(
        "}  // namespace fa_normalize"
    )[0]

    assert 'status.name = "fa_normalize";' in diagnostics
    assert '"target_peak_linear"' in diagnostics
    assert '"silence_threshold_linear"' in diagnostics
    assert "backend_->targetPeakLinear()" in diagnostics
    assert "backend_->silenceThresholdLinear()" in diagnostics
    assert '"last_gain"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"frames_silence_passthrough"' in diagnostics
    assert '"frames_normalized"' in diagnostics
    assert '"output_topic"' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_peak_normalize.md",
        "config/default.yaml",
        "launch/fa_normalize.launch.py",
        "include/fa_normalize/fa_normalize_node.hpp",
        "include/fa_normalize/backends/internal_peak_normalize.hpp",
        "src/fa_normalize_node.cpp",
        "src/main.cpp",
        "src/backends/internal_peak_normalize.cpp",
        "test/cpp/test_internal_peak_normalize_backend.cpp",
        "test/cpp/test_normalize_graph.cpp",
        "test/launch/test_fa_normalize_launch_contract.py",
        "test/unit/test_fa_normalize_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_and_backend_gtest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")
    node_source = read_node_source()
    main_source = read_main_source()

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "add_library(fa_normalize_node_core" in cmake_text
    assert "src/fa_normalize_node.cpp" in cmake_text
    assert "add_executable(fa_normalize_node" in cmake_text
    assert "src/main.cpp" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "test/cpp/test_normalize_graph.cpp" in cmake_text
    assert "fa_normalize_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "int main(" not in node_source
    assert "int main(int argc, char ** argv)" in main_source

from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_resample_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_resample"
        / "backends"
        / "internal_linear_resampler.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_linear_resampler.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32le_output_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_resample"]["ros__parameters"]

    assert params["target_sample_rate"] == 16000
    assert params["input"]["encoding"] == "FLOAT32LE"
    assert params["input"]["bit_depth"] == 32
    assert params["input"]["layout"] == "interleaved"
    assert params["output"]["encoding"] == "FLOAT32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["mic"]["input_topic"] == "audio/frame"
    assert params["mic"]["output_topic"] == "audio/resample16k/mic"
    assert params["mic"]["input_stream_id"] == "audio/float32/mic"
    assert params["mic"]["output"]["stream_id"] == "audio/preprocessed/mono16k"
    assert params["mic"]["input_topic"] != params["mic"]["input_stream_id"]
    assert params["mic"]["output_topic"] != params["mic"]["output"]["stream_id"]


def test_resample_uses_validated_qos_depth_without_hidden_fallback() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaResampleNode::loadParameters")[1].split(
        "void FaResampleNode::configureBackend"
    )[0]

    assert "qos.depth must be > 0" in source
    assert "diagnostics.qos.depth must be > 0" in source
    assert "rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));" in source
    assert "std::max<int>(1, config_.qos_depth)" not in source
    assert 'readRequiredInt(*this, "target_sample_rate")' in load_parameters
    assert 'readRequiredString(*this, "input.encoding")' in load_parameters
    assert 'readRequiredBool(*this, "mic.enabled")' in load_parameters
    assert 'readRequiredString(*this, "mic.input_stream_id")' in load_parameters
    assert 'readRequiredString(*this, "mic.output.stream_id")' in load_parameters
    assert 'readRequiredString(*this, "ref.input_stream_id")' in load_parameters
    assert 'readRequiredString(*this, "ref.output.stream_id")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    assert 'readRequiredInt(*this, "diagnostics.qos.depth")' in load_parameters
    assert 'readRequiredBool(*this, "diagnostics.qos.reliable")' in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line


def test_resample_preserves_source_and_publishes_stream_identity() -> None:
    source = read_node_source()

    assert "out.source_id = in.source_id;" in source
    assert "out.stream_id = output_stream_id;" in source
    assert "out.layout = backends::kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source


def test_resample_rejects_non_float32le_or_unidentified_frames() -> None:
    source = read_node_source()
    backend_source = read_backend_source()

    assert "in.source_id.empty() || in.stream_id.empty()" in source
    assert "in.stream_id != expected_input_stream_id" in source
    assert "stream_id mismatch" in source
    assert "target_sample_rate must be > 0" in source
    assert "requires target_sample_rate=16000" not in source
    assert "fa_resample input.encoding must be FLOAT32LE" in source
    assert "fa_resample input.bit_depth must be 32" in source
    assert "fa_resample input.layout must be interleaved" in source
    assert "validateFloat32InterleavedContract(contract)" in backend_source
    assert "contract.encoding != kEncodingFloat32Le" in backend_source
    assert "contract.bit_depth != 32" in backend_source
    assert "contract.layout != kInterleavedLayout" in backend_source
    assert "contract.data_size == 0" in backend_source
    assert "contract.data_size % bytes_per_frame" in backend_source


def test_resample_backend_owns_decode_validate_resample_and_encode() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "class InternalLinearResamplerBackend" in header
    assert "enum class ProcessStatus" in header
    assert "struct ProcessResult" in header
    assert "std::vector<float> decodeFloat32Le" in backend_source
    assert "std::vector<uint8_t> encodeFloat32Le" in backend_source
    assert "std::vector<float> resampleLinear" in backend_source
    assert "const double ratio = static_cast<double>(out_rate) / static_cast<double>(in_rate);" in backend_source
    assert "std::lround(out_frames_f)" in backend_source
    assert "backend_->process(in.data, frame_contract, out_bytes)" in node_source
    assert "decodeFloat32Le" not in node_source
    assert "resampleLinear" not in node_source
    assert "encodeFloat32Le" not in node_source


def test_resample_does_not_hide_sample_format_conversion_or_clamping() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    combined = source + backend_source
    forbidden = (
        "PCM16LE",
        "PCM32LE",
        "int16_t",
        "std::clamp",
        "32768.0",
        "32767.0",
        "2147483648.0",
    )
    for token in forbidden:
        assert token not in combined

    assert "containsOnlyFiniteNormalizedSamples" in backend_source
    assert "ProcessStatus::kInvalidInputSamples" in backend_source
    assert "ProcessStatus::kEncodeFailed" in backend_source
    assert 'throw std::logic_error("unhandled resample backend process status")' in backend_source
    assert 'throw std::logic_error("unhandled resample frame contract status")' in backend_source
    assert "unknown resample backend status" not in backend_source


def test_resample_backend_reports_typed_status_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "kInvalidFrameContract" in header
    assert "kInvalidInputSamples" in header
    assert "kResampleFailed" in header
    assert "kEncodeFailed" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "frameContractStatusName(FrameContractStatus status)" in header
    assert "ProcessStatus::kInvalidFrameContract" in backend_source
    assert "ProcessStatus::kInvalidInputSamples" in backend_source
    assert "ProcessStatus::kResampleFailed" in backend_source
    assert "ProcessStatus::kEncodeFailed" in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source
    assert "backends::frameContractStatusName(result.frame_contract_status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_colcon_runs_cpp_and_pytest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "add_library(fa_resample_internal_linear_resampler STATIC" in cmake_text
    assert "add_library(fa_resample_node_core" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_test" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml

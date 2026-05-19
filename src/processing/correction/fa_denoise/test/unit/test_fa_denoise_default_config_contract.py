from pathlib import Path

import yaml


def test_default_config_requires_explicit_dtln_models() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_denoise"]["ros__parameters"]

    assert params["enabled"] is True
    assert params["backend.name"] == "dtln_onnx"
    assert "backend" not in params
    assert params["input_topic"] == "fa_denoise/input"
    assert params["output_topic"] == "fa_denoise/output"
    assert params["input_stream_id"] == "audio/resample16k/mic"
    assert params["output"]["stream_id"] == "audio/denoised/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["dtln"]["model_1_path"] == ""
    assert params["dtln"]["model_2_path"] == ""


def test_denoise_outputs_audio_frame_identity_without_analysis_metadata() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_denoise_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "msg.source_id.empty() || msg.stream_id.empty()" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "fa_denoise received a null AudioFrame pointer" in source
    assert "fa_denoise publisher is not initialized" in source
    assert "out_msg.stream_id = config_.output_stream_id;" in source
    assert "out_msg.stream_id = config_.output_topic;" not in source
    assert "out_msg.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    source = (Path(__file__).parents[2] / "src" / "fa_denoise_node.cpp").read_text(
        encoding="utf-8"
    )
    load_parameters = source.split("void FaDenoiseNode::loadParameters")[1].split(
        "if (config_.input_topic.empty())"
    )[0]

    required_reads = (
        'readRequiredBool(*this, "enabled")',
        'readRequiredString(*this, "backend.name")',
        'readRequiredString(*this, "input_topic")',
        'readRequiredString(*this, "output_topic")',
        'readRequiredString(*this, "input_stream_id")',
        'readRequiredString(*this, "output.stream_id")',
        'readRequiredInt(*this, "expected_sample_rate")',
        'readRequiredInt(*this, "expected_channels")',
        'readRequiredString(*this, "expected.encoding")',
        'readRequiredInt(*this, "expected.bit_depth")',
        'readRequiredString(*this, "output.encoding")',
        'readRequiredInt(*this, "output.bit_depth")',
        'readRequiredInt(*this, "dtln.block_len")',
        'readRequiredInt(*this, "dtln.block_shift")',
        'readRequiredString(*this, "dtln.model_1_path")',
        'readRequiredString(*this, "dtln.model_2_path")',
        'readRequiredInt(*this, "dtln.intra_op_num_threads")',
        'readRequiredInt(*this, "dtln.inter_op_num_threads")',
        'readRequiredBool(',
        '"dtln.enable_ort_optimizations"',
        'readRequiredInt(*this, "qos.depth")',
        'readRequiredInt(*this, "diagnostics.qos.depth")',
        'readRequiredBool(*this, "qos.reliable")',
        'readRequiredBool(*this, "diagnostics.qos.reliable")',
    )
    for read in required_reads:
        assert read in load_parameters

    assert "readRequiredInt(" in load_parameters
    assert '"diagnostics.publish_period_ms"' in load_parameters
    assert "this->get_parameter(" not in load_parameters
    assert "SystemDefaultsQoS" not in source
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert "config_." not in line


def test_dtln_backend_lives_under_backend_boundary() -> None:
    package_root = Path(__file__).parents[2]
    contract_header = (
        package_root / "include" / "fa_denoise" / "backends" / "denoise_backend.hpp"
    )
    contract_source = package_root / "src" / "backends" / "denoise_backend.cpp"
    passthrough_header = (
        package_root / "include" / "fa_denoise" / "backends" / "passthrough_backend.hpp"
    )
    passthrough_source = package_root / "src" / "backends" / "passthrough_backend.cpp"
    wrapper_header = (
        package_root / "include" / "fa_denoise" / "backends" / "dtln_onnx_backend.hpp"
    )
    wrapper_source = package_root / "src" / "backends" / "dtln_onnx_backend.cpp"
    backend_header = (
        package_root / "include" / "fa_denoise" / "backends" / "dtln_onnx_engine.hpp"
    )
    backend_source = package_root / "src" / "backends" / "dtln_onnx_engine.cpp"
    old_header = package_root / "include" / "fa_denoise" / "dtln_onnx_engine.hpp"
    old_source = package_root / "src" / "dtln_onnx_engine.cpp"
    node_source = package_root / "src" / "fa_denoise_node.cpp"

    assert contract_header.is_file()
    assert contract_source.is_file()
    assert passthrough_header.is_file()
    assert passthrough_source.is_file()
    assert wrapper_header.is_file()
    assert wrapper_source.is_file()
    assert backend_header.is_file()
    assert backend_source.is_file()
    assert not old_header.exists()
    assert not old_source.exists()
    node_text = node_source.read_text(encoding="utf-8")
    assert '#include "fa_denoise/backends/dtln_onnx_engine.hpp"' not in node_text
    assert '#include "fa_denoise/backends/denoise_backend.hpp"' in node_text
    assert '#include "fa_denoise/backends/passthrough_backend.hpp"' in node_text
    assert '#include "fa_denoise/backends/dtln_onnx_backend.hpp"' in node_text
    assert "DtlnOnnxEngine" not in node_text


def test_backend_files_are_ros_free() -> None:
    package_root = Path(__file__).parents[2]
    backend_files = [
        package_root / "include" / "fa_denoise" / "backends" / "denoise_backend.hpp",
        package_root / "src" / "backends" / "denoise_backend.cpp",
        package_root / "include" / "fa_denoise" / "backends" / "passthrough_backend.hpp",
        package_root / "src" / "backends" / "passthrough_backend.cpp",
        package_root / "include" / "fa_denoise" / "backends" / "dtln_onnx_backend.hpp",
        package_root / "src" / "backends" / "dtln_onnx_backend.cpp",
        package_root / "include" / "fa_denoise" / "backends" / "dtln_onnx_engine.hpp",
        package_root / "src" / "backends" / "dtln_onnx_engine.cpp",
    ]
    forbidden_tokens = (
        "rclcpp",
        "fa_interfaces",
        "diagnostic_msgs",
        "std_msgs/msg",
    )

    for backend_file in backend_files:
        source = backend_file.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source


def test_dtln_overlap_add_is_documented_as_model_internal_reconstruction() -> None:
    package_root = Path(__file__).parents[2]
    engine_header = (
        package_root
        / "include"
        / "fa_denoise"
        / "backends"
        / "dtln_onnx_engine.hpp"
    ).read_text(encoding="utf-8")


    assert "DTLN model reconstruction buffers" in engine_header
    assert "streaming buffers" not in engine_header


def test_cmake_builds_backend_library_and_registers_pytest() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert 'set(FA_DENOISE_ONNXRUNTIME "AUTO"' in cmake_text
    assert 'FA_DENOISE_ONNXRUNTIME MATCHES "^(AUTO|ON|OFF)$"' in cmake_text
    assert 'FA_DENOISE_ONNXRUNTIME STREQUAL "ON"' in cmake_text
    assert 'FA_DENOISE_WITH_ONNXRUNTIME' in cmake_text
    assert "Selecting backend.name=dtln_onnx will fail closed at node startup" in cmake_text
    assert "add_library(fa_denoise_backend_contract STATIC" in cmake_text
    assert "src/backends/denoise_backend.cpp" in cmake_text
    assert "src/backends/passthrough_backend.cpp" in cmake_text
    assert "add_library(fa_denoise_dtln_onnx_backend STATIC" in cmake_text
    assert "src/backends/dtln_onnx_backend.cpp" in cmake_text
    assert "src/backends/dtln_onnx_engine.cpp" in cmake_text
    assert "third_party/kissfft/kiss_fft.c" in cmake_text
    assert "add_library(fa_denoise_node_core" in cmake_text
    assert "target_link_libraries(fa_denoise_node_core" in cmake_text
    assert "fa_denoise_backend_contract" in cmake_text
    assert "fa_denoise_dtln_onnx_backend" in cmake_text
    assert "target_link_libraries(fa_denoise_node" in cmake_text
    assert "fa_denoise_node_core" in cmake_text
    assert "target_sources(fa_denoise_node" not in cmake_text
    assert "src/dtln_onnx_engine.cpp" not in cmake_text
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_node_contract_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_contract_test" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml


def test_node_fails_closed_when_dtln_selected_without_onnxruntime() -> None:
    node_text = (Path(__file__).parents[2] / "src" / "fa_denoise_node.cpp").read_text(
        encoding="utf-8"
    )

    assert (
        "fa_denoise was built without ONNX Runtime support "
        "(FA_DENOISE_WITH_ONNXRUNTIME=0)"
    ) in node_text


def test_dtln_model_paths_are_required_before_backend_initialization() -> None:
    node_text = (Path(__file__).parents[2] / "src" / "fa_denoise_node.cpp").read_text(
        encoding="utf-8"
    )
    load_parameters = node_text.split("void FaDenoiseNode::loadParameters")[1].split(
        "void FaDenoiseNode::configureBackend"
    )[0]

    assert "config_.dtln_model_1_path.empty()" in load_parameters
    assert "dtln.model_1_path is required for dtln_onnx backend" in load_parameters
    assert "config_.dtln_model_2_path.empty()" in load_parameters
    assert "dtln.model_2_path is required for dtln_onnx backend" in load_parameters


def test_denoise_requires_explicit_format_pairs_and_no_hidden_clamp() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_denoise_node.cpp").read_text(
        encoding="utf-8"
    )

    assert "isSupportedAudioFormatPair" in source
    assert "expected.encoding/expected.bit_depth must be PCM16LE/16 or FLOAT32LE/32" in source
    assert "output.encoding/output.bit_depth must be PCM16LE/16 or FLOAT32LE/32" in source
    assert "msg.encoding != config_.expected_encoding" in source
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in source
    assert "msg.stream_id != config_.input_stream_id" in source
    assert "config_.input_topic == config_.output_topic" in source
    assert "resolved input_topic and output_topic must be distinct" in source
    assert "sameIdentityString(config_.input_stream_id, config_.input_topic)" in source
    assert "sameIdentityString(config_.input_stream_id, config_.output_topic)" in source
    assert "sameIdentityString(config_.input_stream_id, config_.resolved_input_topic)" in source
    assert "sameIdentityString(config_.input_stream_id, config_.resolved_output_topic)" in source
    assert "sameIdentityString(config_.output_stream_id, config_.input_topic)" in source
    assert "sameIdentityString(config_.output_stream_id, config_.output_topic)" in source
    assert "sameIdentityString(config_.output_stream_id, config_.resolved_input_topic)" in source
    assert "sameIdentityString(config_.output_stream_id, config_.resolved_output_topic)" in source
    assert "sameIdentityString(config_.input_stream_id, config_.output_stream_id)" in source
    assert "hasValidStamp(msg.header.stamp)" in source
    assert "std::max<int>(1, config_.qos_depth)" not in source
    assert "decodeToFloat" not in source
    assert "encodeFromFloat" not in source
    backend_source = (package_root / "src" / "backends" / "denoise_backend.cpp").read_text(
        encoding="utf-8"
    )
    assert "input.format.encoding != kEncodingFloat32 || input.format.bit_depth != 32" in backend_source
    assert "fa_denoise passthrough requires output format to match expected input format" in source
    assert "sample count is not aligned to dtln.block_shift" in backend_source
    assert "output sample count does not match input sample count" in backend_source
    assert "denoise output sample is outside normalized [-1.0, 1.0] range" in backend_source
    hidden_clamp_token = "std::" + "clamp"
    assert hidden_clamp_token not in source
    assert hidden_clamp_token not in backend_source


def test_dtln_backend_validates_onnx_output_shapes_before_copy() -> None:
    backend_source = (
        Path(__file__).parents[2] / "src" / "backends" / "dtln_onnx_engine.cpp"
    ).read_text(encoding="utf-8")

    assert "requireShapeEquals(mask_shape, mag_shape_" in backend_source
    assert "requireShapeEquals(state_output_shape_1, state_shape_1_" in backend_source
    assert "requireShapeEquals(time_output_shape, time_shape_" in backend_source
    assert "requireShapeEquals(state_output_shape_2, state_shape_2_" in backend_source
    assert "DTLN model_1 mask output" in backend_source
    assert "DTLN model_2 time output" in backend_source
    assert "dim = 1" not in backend_source
    assert "DTLN ONNX tensor shape must be fully static and positive" in backend_source
    assert "DTLN process requires non-empty sample input" in backend_source


def test_main_shutdown_is_guarded_by_rclcpp_ok() -> None:
    main_source = (Path(__file__).parents[2] / "src" / "main.cpp").read_text(
        encoding="utf-8"
    )

    assert "if (rclcpp::ok()) {\n      rclcpp::shutdown();" in main_source
    assert "}\n  if (rclcpp::ok()) {\n    rclcpp::shutdown();" in main_source

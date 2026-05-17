from pathlib import Path

import yaml


def test_default_config_requires_explicit_dtln_models() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_denoise"]["ros__parameters"]

    assert params["enabled"] is True
    assert params["backend.name"] == "dtln_onnx"
    assert "backend" not in params
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["dtln"]["model_1_path"] == ""
    assert params["dtln"]["model_2_path"] == ""


def test_denoise_outputs_audio_frame_identity_without_analysis_metadata() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_denoise_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "msg.source_id.empty() || msg.stream_id.empty()" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "out_msg.stream_id = config_.output_topic;" in source
    assert "out_msg.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source


def test_dtln_backend_lives_under_backend_boundary() -> None:
    package_root = Path(__file__).parents[2]
    backend_header = (
        package_root / "include" / "fa_denoise" / "backends" / "dtln_onnx_engine.hpp"
    )
    backend_source = package_root / "src" / "backends" / "dtln_onnx_engine.cpp"
    old_header = package_root / "include" / "fa_denoise" / "dtln_onnx_engine.hpp"
    old_source = package_root / "src" / "dtln_onnx_engine.cpp"
    node_source = package_root / "src" / "fa_denoise_node.cpp"

    assert backend_header.is_file()
    assert backend_source.is_file()
    assert not old_header.exists()
    assert not old_source.exists()
    assert '#include "fa_denoise/backends/dtln_onnx_engine.hpp"' in node_source.read_text(
        encoding="utf-8"
    )


def test_backend_files_are_ros_free() -> None:
    package_root = Path(__file__).parents[2]
    backend_files = [
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


def test_cmake_builds_backend_library_and_registers_pytest() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert 'set(FA_DENOISE_ONNXRUNTIME "AUTO"' in cmake_text
    assert 'FA_DENOISE_ONNXRUNTIME MATCHES "^(AUTO|ON|OFF)$"' in cmake_text
    assert 'FA_DENOISE_ONNXRUNTIME STREQUAL "ON"' in cmake_text
    assert 'FA_DENOISE_WITH_ONNXRUNTIME' in cmake_text
    assert "Selecting backend.name=dtln_onnx will fail closed at node startup" in cmake_text
    assert "add_library(fa_denoise_backends STATIC" in cmake_text
    assert "src/backends/dtln_onnx_engine.cpp" in cmake_text
    assert "third_party/kissfft/kiss_fft.c" in cmake_text
    assert "add_library(fa_denoise_node_core" in cmake_text
    assert "target_link_libraries(fa_denoise_node_core" in cmake_text
    assert "fa_denoise_backends" in cmake_text
    assert "target_link_libraries(fa_denoise_node" in cmake_text
    assert "fa_denoise_node_core" in cmake_text
    assert "target_sources(fa_denoise_node" not in cmake_text
    assert "src/dtln_onnx_engine.cpp" not in cmake_text
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_node_contract_test" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml


def test_node_fails_closed_when_dtln_selected_without_onnxruntime() -> None:
    node_text = (Path(__file__).parents[2] / "src" / "fa_denoise_node.cpp").read_text(
        encoding="utf-8"
    )
    spec_text = (Path(__file__).parents[2] / "docs" / "仕様書.md").read_text(
        encoding="utf-8"
    )

    assert (
        "fa_denoise was built without ONNX Runtime support "
        "(FA_DENOISE_WITH_ONNXRUNTIME=0)"
    ) in node_text
    assert "別 backend へ暗黙に切り替えない" in spec_text


def test_denoise_requires_explicit_format_pairs_and_no_hidden_clamp() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_denoise_node.cpp").read_text(
        encoding="utf-8"
    )
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package_root / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )

    assert "isSupportedAudioFormatPair" in source
    assert "expected.encoding/expected.bit_depth must be PCM16LE/16 or FLOAT32LE/32" in source
    assert "output.encoding/output.bit_depth must be PCM16LE/16 or FLOAT32LE/32" in source
    assert "msg.encoding != config_.expected_encoding" in source
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in source
    assert "msg.stream_id != config_.input_topic" in source
    assert "std::max<int>(1, config_.qos_depth)" not in source
    assert "msg.encoding != kEncodingFloat32 || msg.bit_depth != 32" in source
    assert "denoise output sample out of normalized range" in source
    assert "std::clamp" not in source
    assert "hidden range clamp" in spec
    assert "PCM32LE/32" in algorithm

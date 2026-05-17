from pathlib import Path

import yaml


def test_default_config_requires_float32le_output_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_resample"]["ros__parameters"]

    assert params["target_sample_rate"] == 16000
    assert params["input"]["encoding"] == "FLOAT32LE"
    assert params["input"]["bit_depth"] == 32
    assert params["input"]["layout"] == "interleaved"
    assert params["output"]["encoding"] == "FLOAT32LE"
    assert params["output"]["bit_depth"] == 32
    assert params["mic"]["input_topic"] == "audio/frame"
    assert params["mic"]["output_topic"] == "audio/resample16k/mic"


def test_resample_preserves_source_and_publishes_stream_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_resample_node.cpp").read_text(encoding="utf-8")

    assert "out.source_id = in.source_id;" in source
    assert "out.stream_id = output_stream_id;" in source
    assert "out.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source


def test_resample_rejects_non_float32le_or_unidentified_frames() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_resample_node.cpp").read_text(encoding="utf-8")
    core = (
        package_root / "include" / "fa_resample" / "resample_core.hpp"
    ).read_text(encoding="utf-8")

    assert "in.source_id.empty() || in.stream_id.empty()" in source
    assert "target_sample_rate must be > 0" in source
    assert "requires target_sample_rate=16000" not in source
    assert "fa_resample input.encoding must be FLOAT32LE" in source
    assert "fa_resample input.bit_depth must be 32" in source
    assert "fa_resample input.layout must be interleaved" in source
    assert "validateFloat32InterleavedContract(frame_contract)" in source
    assert "contract.encoding != kEncodingFloat32Le" in core
    assert "contract.bit_depth != 32" in core
    assert "contract.layout != kInterleavedLayout" in core
    assert "contract.data_size == 0" in core
    assert "contract.data_size % bytes_per_frame" in core


def test_resample_does_not_hide_sample_format_conversion_or_clamping() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_resample_node.cpp").read_text(encoding="utf-8")
    core = (
        package_root / "include" / "fa_resample" / "resample_core.hpp"
    ).read_text(encoding="utf-8")

    combined = source + core
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

    assert "containsOnlyFiniteNormalizedSamples" in source
    assert "encodeFloat32Le(out_f32)" in source


def test_colcon_runs_cpp_and_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "add_library(fa_resample_node_core" in cmake_text
    assert "ament_add_gtest(test_resample_core" in cmake_text
    assert "ament_add_gtest(test_resample_graph" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml

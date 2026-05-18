from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_gain_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_gain" / "backends" / "internal_gain.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_gain.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_gain"]["ros__parameters"]

    assert params["input_topic"] == "audio/resample16k/mic"
    assert params["output_topic"] == "audio/gain/mic"
    assert params["gain"]["linear"] == 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"


def test_gain_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "std::clamp",
        "clip",
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_gain_validates_frame_contract_before_processing() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaGainNode::validateFrame")[1].split(
        "bool FaGainNode::applyGain"
    )[0]
    handle_frame = source.split("void FaGainNode::handleFrame")[1].split(
        "bool FaGainNode::validateFrame"
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


def test_gain_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_gain = source.split("bool FaGainNode::applyGain")[1].split(
        "void FaGainNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_gain
    assert "out.stream_id = config_.output_topic;" in apply_gain
    assert ".rms" not in apply_gain
    assert ".peak" not in apply_gain
    assert ".vad" not in apply_gain


def test_gain_algorithm_uses_ros_free_backend() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessStatus InternalGainBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "class InternalGainBackend" in header
    assert "enum class ProcessStatus" in header
    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "static_cast<double>(sample) * config_.linear_gain" in process
    assert "output = std::move(next_output);" in process
    assert "backends::processStatusMessage(status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_gain_drops_out_of_range_samples_instead_of_limiting() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessStatus InternalGainBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(gained)" in process
    assert "gained < kNormalizedMin || gained > kNormalizedMax" in process
    assert "return ProcessStatus::kOutOfRangeInput;" in process
    assert "return ProcessStatus::kOutOfRangeOutput;" in process
    assert "std::clamp" not in process
    assert "backends::processStatusMessage(status)" in node_source


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_gain.md",
        "config/default.yaml",
        "launch/fa_gain.launch.py",
        "include/fa_gain/fa_gain_node.hpp",
        "include/fa_gain/backends/internal_gain.hpp",
        "src/fa_gain_node.cpp",
        "src/backends/internal_gain.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_gain_backend.cpp",
        "test/cpp/test_gain_graph.cpp",
        "test/unit/test_fa_gain_audio_frame_contract.py",
        "test/launch/test_fa_gain_launch_contract.py",
        "test/integration/.gitkeep",
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
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

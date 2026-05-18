from pathlib import Path

import yaml


def _package_root() -> Path:
    return Path(__file__).parents[2]


def _source() -> str:
    return (_package_root() / "src" / "fa_time_alignment_node.cpp").read_text(encoding="utf-8")


def _header() -> str:
    return (_package_root() / "include" / "fa_time_alignment" / "fa_time_alignment_node.hpp").read_text(
        encoding="utf-8"
    )


def test_default_config_declares_explicit_time_alignment_contract() -> None:
    config = yaml.safe_load((_package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_time_alignment"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame_buffer/mic"
    assert params["output_topic"] == "audio/time_aligned/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["alignment"]["period_ms"] == 20.0
    assert params["alignment"]["phase_ms"] == 0.0
    assert params["alignment"]["max_adjust_ms"] == 2.0
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = _source()
    load_parameters = source.split("void FaTimeAlignmentNode::loadParameters")[1].split(
        "void FaTimeAlignmentNode::setupInterfaces"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding.empty()" in load_parameters
    assert "config_.expected_bit_depth <= 0" in load_parameters
    assert "(config_.expected_bit_depth % 8) != 0" in load_parameters
    assert "config_.expected_layout.empty()" in load_parameters
    assert "!std::isfinite(config_.alignment_period_ms)" in load_parameters
    assert "config_.alignment_period_ms <= 0.0" in load_parameters
    assert "!std::isfinite(config_.alignment_phase_ms)" in load_parameters
    assert "config_.alignment_phase_ms < 0.0" in load_parameters
    assert "config_.alignment_phase_ms >= config_.alignment_period_ms" in load_parameters
    assert "!std::isfinite(config_.alignment_max_adjust_ms)" in load_parameters
    assert "config_.alignment_max_adjust_ms < 0.0" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert 'declare_parameter<double>("alignment.period_ms");' in load_parameters
    assert 'declare_parameter<double>("alignment.phase_ms");' in load_parameters
    assert 'declare_parameter<double>("alignment.max_adjust_ms");' in load_parameters
    assert 'declare_parameter<bool>("qos.reliable");' in load_parameters
    assert 'declare_parameter<bool>("qos.reliable", config_.qos_reliable)' not in load_parameters
    assert "throw std::runtime_error" in load_parameters


def test_runtime_validates_audio_frame_identity_format_and_byte_alignment() -> None:
    source = _source()
    validate_frame = source.split("bool FaTimeAlignmentNode::validateFrame")[1].split(
        "bool FaTimeAlignmentNode::alignFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "const size_t bytes_per_sample = static_cast<size_t>(config_.expected_bit_depth) / 8U;" in validate_frame
    assert "const size_t bytes_per_frame = static_cast<size_t>(config_.expected_channels) * bytes_per_sample;" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0U" in validate_frame
    assert "return false;" in validate_frame


def test_time_alignment_uses_nearest_grid_and_drops_excess_adjustment() -> None:
    source = _source()
    align_frame = source.split("bool FaTimeAlignmentNode::alignFrame")[1].split(
        "void FaTimeAlignmentNode::publishDiagnostics"
    )[0]

    assert "stampToNanoseconds(in.header.stamp)" in align_frame
    assert "config_.alignment_period_ms" in align_frame
    assert "config_.alignment_phase_ms" in align_frame
    assert "std::round((input_ns - phase_ns) / period_ns)" in align_frame
    assert "phase_ns + (grid_index * period_ns)" in align_frame
    assert "aligned_ns_decimal < 0.0L" in align_frame
    assert "aligned_ns_decimal > static_cast<long double>(kMaxBuiltinTimeNanoseconds)" in align_frame
    assert "std::fabs(adjustment_ns) > max_adjust_ns" in align_frame
    assert "frames_excess_adjust_.fetch_add(1)" in align_frame
    assert "return false;" in align_frame


def test_output_preserves_audio_data_and_updates_only_stamp_and_stream_id() -> None:
    source = _source()
    align_frame = source.split("bool FaTimeAlignmentNode::alignFrame")[1].split(
        "void FaTimeAlignmentNode::publishDiagnostics"
    )[0]

    assert "out = in;" in align_frame
    assert "out.header.stamp = nanosecondsToStamp(aligned_ns);" in align_frame
    assert "out.stream_id = config_.output_topic;" in align_frame
    assert "out.data" not in align_frame
    assert "out.encoding =" not in align_frame
    assert "out.bit_depth =" not in align_frame
    assert "out.sample_rate =" not in align_frame
    assert "out.channels =" not in align_frame
    assert "out.layout =" not in align_frame
    assert "std::memcpy" not in align_frame


def test_streaming_node_has_no_device_io_sample_editing_or_legacy_aliases() -> None:
    source = _source()
    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "convertPcm",
        "std::memcpy",
        "std::clamp",
        "normalize(",
        "decodeToFloat",
        "encode",
        "legacy",
        "deprecated",
        "alias",
        "try { import",
    )
    for token in forbidden:
        assert token not in source


def test_ros2_node_name_and_executable_match_required_contract() -> None:
    source = _source()
    header = _header()
    main_source = (_package_root() / "src" / "main.cpp").read_text(encoding="utf-8")
    launch = (_package_root() / "launch" / "fa_time_alignment.launch.py").read_text(
        encoding="utf-8"
    )
    cmake = (_package_root() / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "class FaTimeAlignmentNode : public rclcpp::Node" in header
    assert 'rclcpp::Node("fa_time_alignment", options)' in source
    assert "explicit FaTimeAlignmentNode(const rclcpp::NodeOptions & options" in header
    assert "fa_time_alignment::FaTimeAlignmentNode" in main_source
    assert 'executable="fa_time_alignment_node"' in launch
    assert "default_value" not in launch
    assert "FindPackageShare" not in launch
    assert "add_executable(fa_time_alignment_node" in cmake


def test_diagnostics_publish_required_counters() -> None:
    source = _source()
    publish_diagnostics = source.split("void FaTimeAlignmentNode::publishDiagnostics")[1]

    assert '"frames_in"' in publish_diagnostics
    assert '"frames_out"' in publish_diagnostics
    assert '"frames_dropped"' in publish_diagnostics
    assert '"frames_aligned"' in publish_diagnostics
    assert '"frames_excess_adjust"' in publish_diagnostics
    assert '"backend.name", "no_runtime_backend"' in publish_diagnostics


def test_package_layout_matches_standard_streaming_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/no_runtime_backend.md",
        "config/default.yaml",
        "launch/fa_time_alignment.launch.py",
        "include/fa_time_alignment/fa_time_alignment_node.hpp",
        "src/fa_time_alignment_node.cpp",
        "src/main.cpp",
        "test/unit/test_fa_time_alignment_audio_frame_contract.py",
        "test/cpp/test_fa_time_alignment_node_contract.cpp",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (_package_root() / relative_path).exists()


def test_colcon_runs_pytest_contracts_and_lint_auto() -> None:
    cmake_text = (_package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (_package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_lint_auto REQUIRED)" in cmake_text
    assert "add_library(fa_time_alignment_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_node_contract_test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "ament_lint_auto_find_test_dependencies()" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

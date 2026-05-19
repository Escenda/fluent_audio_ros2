from pathlib import Path

import yaml


def test_default_config_requires_explicit_backend() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    params = config["fa_aec_nn"]["ros__parameters"]

    assert params["enabled"] is True
    assert params["backend.name"] == ""
    assert "backend" not in params
    assert params["input_topic"] == "fa_aec_nn/input"
    assert params["output_topic"] == "fa_aec_nn/output"
    assert params["input_stream_id"] == "audio/aec_linear/frame"
    assert params["output"]["stream_id"] == "audio/aec/frame"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["expected_channels"] == 1
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert "-1" not in config_text


def test_disabled_branch_drops_before_publish() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_aec_nn_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    disabled_index = source.index("if (!config_.enabled)")
    publish_index = source.index("pub_->publish(out_msg)")

    assert disabled_index < publish_index
    assert "Dropping frame because fa_aec_nn is disabled" in source


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    source = (Path(__file__).parents[2] / "src" / "fa_aec_nn_node.cpp").read_text(
        encoding="utf-8"
    )
    load_parameters = source.split("void FaAecNnNode::loadParameters")[1].split(
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


def test_aec_nn_passthrough_updates_output_stream_identity() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_aec_nn_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "msg.source_id.empty()" in source
    assert "msg.stream_id != config_.input_stream_id" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "processed_chunk = backend_->process(chunk);" in source
    assert "validateProcessedAudioChunk(chunk, processed_chunk)" in source
    assert "out_msg.data = std::move(processed_chunk.data);" in source
    assert "out_msg.stream_id = config_.output_stream_id;" in source
    assert "out_msg.stream_id = config_.output_topic;" not in source
    assert "out_msg.layout = processed_chunk.layout;" in source


def test_passthrough_backend_lives_under_backend_boundary() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "include/fa_aec_nn/backends/aec_nn_backend.hpp",
        "include/fa_aec_nn/backends/passthrough_backend.hpp",
        "src/backends/passthrough_backend.cpp",
    )

    for relative_path in required_paths:
        assert (package_root / relative_path).is_file()


def test_backend_files_are_ros_free() -> None:
    package_root = Path(__file__).parents[2]
    backend_files = [
        package_root / "include" / "fa_aec_nn" / "backends" / "aec_nn_backend.hpp",
        package_root / "include" / "fa_aec_nn" / "backends" / "passthrough_backend.hpp",
        package_root / "src" / "backends" / "passthrough_backend.cpp",
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


def test_backend_output_contract_is_explicit_and_validated() -> None:
    package_root = Path(__file__).parents[2]
    backend_header = (
        package_root / "include" / "fa_aec_nn" / "backends" / "aec_nn_backend.hpp"
    ).read_text(encoding="utf-8")
    source = (package_root / "src" / "fa_aec_nn_node.cpp").read_text(
        encoding="utf-8"
    )
    passthrough_source = (
        package_root / "src" / "backends" / "passthrough_backend.cpp"
    ).read_text(encoding="utf-8")

    assert "struct ProcessedAudioChunk" in backend_header
    assert "std::vector<uint8_t> data;" in backend_header
    assert "validateProcessedAudioChunk" in backend_header
    assert "backend output sample_rate must match input sample_rate" in backend_header
    assert "backend output audio data must be non-empty and PCM frame aligned" in backend_header
    assert "backend output audio data size must match input audio data size" in backend_header
    assert "chunk.encoding = msg->encoding;" in source
    assert "backends::validateProcessedAudioChunk(chunk, processed_chunk)" in source
    assert "fa_aec_nn backend violated output contract" in source
    assert "output.encoding = chunk.encoding;" in passthrough_source
    assert "output.data = std::vector<uint8_t>(chunk.data, chunk.data + chunk.data_size)" in passthrough_source


def test_cmake_builds_backend_library_and_node_links_it() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_library(fa_aec_nn_backends STATIC" in cmake_text
    assert "src/backends/passthrough_backend.cpp" in cmake_text
    assert "add_library(fa_aec_nn_node_core" in cmake_text
    assert "target_link_libraries(fa_aec_nn_node_core\n  fa_aec_nn_backends" in cmake_text
    assert "target_link_libraries(fa_aec_nn_node\n  fa_aec_nn_node_core" in cmake_text
    assert "target_sources(fa_aec_nn_node" not in cmake_text


def test_main_shutdown_is_guarded_by_rclcpp_ok() -> None:
    package_root = Path(__file__).parents[2]
    main_source = (package_root / "src" / "main.cpp").read_text(encoding="utf-8")

    assert "if (rclcpp::ok()) {\n      rclcpp::shutdown();" in main_source
    assert "}\n  if (rclcpp::ok()) {\n    rclcpp::shutdown();" in main_source


def test_aec_nn_rejects_channel_wildcards_and_unsupported_format_pairs() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_aec_nn_node.cpp").read_text(
        encoding="utf-8"
    )

    assert "config_.expected_channels <= 0" in source
    assert "config_.input_stream_id.empty()" in source
    assert "config_.output_stream_id.empty()" in source
    assert "resolve_topic_name(config_.input_topic)" in source
    assert "resolve_topic_name(config_.output_topic)" in source
    assert "input_stream_id must be distinct from ROS topics" in source
    assert "output.stream_id must be distinct from ROS topics" in source
    assert "input_stream_id and output.stream_id must be distinct" in source
    assert "config_.expected_channels > 0" not in source
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in source
    assert "msg.encoding != config_.expected_encoding" in source
    assert "expected encoding/bit_depth must be PCM16LE/16 or FLOAT32LE/32" in source
    assert 'throw std::logic_error("fa_aec_nn backend is not initialized")' in source
    assert "Dropping frame because fa_aec_nn backend is not initialized" not in source


def test_package_manifest_declares_launch_and_yaml_dependencies() -> None:
    package_xml = (Path(__file__).parents[2] / "package.xml").read_text(
        encoding="utf-8"
    )

    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

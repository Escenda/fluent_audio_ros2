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


def test_aec_nn_passthrough_updates_output_stream_identity() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_aec_nn_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "msg.source_id.empty()" in source
    assert "msg.stream_id != config_.input_topic" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "processed_data = backend_->process(chunk);" in source
    assert "out_msg.data = std::move(processed_data);" in source
    assert "out_msg.stream_id = config_.output_topic;" in source
    assert "out_msg.layout = kInterleavedLayout;" in source


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


def test_cmake_builds_backend_library_and_node_links_it() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_library(fa_aec_nn_backends STATIC" in cmake_text
    assert "src/backends/passthrough_backend.cpp" in cmake_text
    assert "add_library(fa_aec_nn_node_core" in cmake_text
    assert "target_link_libraries(fa_aec_nn_node_core\n  fa_aec_nn_backends" in cmake_text
    assert "target_link_libraries(fa_aec_nn_node\n  fa_aec_nn_node_core" in cmake_text
    assert "target_sources(fa_aec_nn_node" not in cmake_text


def test_aec_nn_rejects_channel_wildcards_and_unsupported_format_pairs() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_aec_nn_node.cpp").read_text(
        encoding="utf-8"
    )
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package_root / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    test_plan = (package_root / "docs" / "テスト設計.md").read_text(
        encoding="utf-8"
    )

    assert "config_.expected_channels <= 0" in source
    assert "config_.expected_channels > 0" not in source
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in source
    assert "msg.encoding != config_.expected_encoding" in source
    assert "expected encoding/bit_depth must be PCM16LE/16 or FLOAT32LE/32" in source
    assert "channel 検査の無効化は禁止" in spec
    assert "`stream_id` は `input_topic`" in spec
    assert "PCM32LE/32" in spec
    assert "channel 検査の wildcard はない" in algorithm
    assert "encoding / bit depth は `PCM16LE/16`" in algorithm
    assert "stream_id` が `input_topic`" in test_plan

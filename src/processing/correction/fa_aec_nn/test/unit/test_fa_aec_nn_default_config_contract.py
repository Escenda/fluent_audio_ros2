from pathlib import Path

import yaml


def test_default_config_requires_explicit_backend() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_aec_nn"]["ros__parameters"]

    assert params["enabled"] is True
    assert params["backend.name"] == ""
    assert "backend" not in params


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

    assert "msg.source_id.empty() || msg.stream_id.empty()" in source
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
    assert "target_link_libraries(fa_aec_nn_node\n  fa_aec_nn_backends" in cmake_text
    assert "target_sources(fa_aec_nn_node" not in cmake_text

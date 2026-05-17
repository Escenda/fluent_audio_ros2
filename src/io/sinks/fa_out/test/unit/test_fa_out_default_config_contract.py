from pathlib import Path

import yaml


def test_default_config_requires_explicit_sink_device() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    params = config["fa_out"]["ros__parameters"]

    assert params["backend.name"] == "alsa_playback"
    assert params["audio.device_id"] == ""
    assert params["audio.encoding"] == "PCM16LE"
    assert params["audio.bit_depth"] == 16
    assert params["audio.chunk_duration_ms"] == 30
    assert params["audio.qos.depth"] == 10
    assert params["audio.qos.reliable"] is True
    assert '"default"' not in config_text


def test_sink_backend_has_no_struct_default() -> None:
    header_path = Path(__file__).parents[2] / "include" / "fa_out" / "fa_out_node.hpp"
    header_text = header_path.read_text(encoding="utf-8")

    assert "std::string backend_name{};" in header_text
    assert "std::string encoding{};" in header_text
    assert "uint32_t sample_rate{0};" in header_text
    assert "uint32_t channels{0};" in header_text
    assert "uint32_t bit_depth{0};" in header_text
    assert "size_t max_queue_frames{0};" in header_text
    assert "uint32_t chunk_duration_ms{0};" in header_text
    assert "size_t qos_depth{0};" in header_text


def test_alsa_sink_rejects_plugin_pcm_devices() -> None:
    backend_source_path = (
        Path(__file__).parents[2] / "src" / "backends" / "alsa_playback_backend.cpp"
    )
    backend_source = backend_source_path.read_text(encoding="utf-8")

    assert "isRawHardwareDevice" in backend_source
    assert 'rfind("hw:", 0)' in backend_source
    assert "audio.device_id must be an ALSA raw hardware id" in backend_source


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert 'declare_parameter<std::string>("audio.encoding")' in source
    assert 'declare_parameter<int>("audio.sample_rate")' in source
    assert 'declare_parameter<int>("audio.channels")' in source
    assert 'declare_parameter<int>("audio.bit_depth")' in source
    assert 'declare_parameter<int>("queue.max_frames")' in source
    assert 'declare_parameter<int>("audio.chunk_duration_ms")' in source
    assert 'declare_parameter<int>("audio.qos.depth")' in source
    assert 'declare_parameter<bool>("audio.qos.reliable")' in source


def test_playback_contract_is_pcm16_only_at_startup() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    backend_source_path = (
        Path(__file__).parents[2] / "src" / "backends" / "alsa_playback_backend.cpp"
    )
    source = source_path.read_text(encoding="utf-8")
    backend_source = backend_source_path.read_text(encoding="utf-8")

    assert "audio.encoding must be PCM16LE for backend.name=alsa_playback" in source
    assert "audio.bit_depth must be 16 for PCM16LE playback" in source
    assert "SND_PCM_FORMAT_S16_LE" in backend_source
    assert "SND_PCM_FORMAT_S32_LE" not in source
    assert "SND_PCM_FORMAT_S32_LE" not in backend_source
    assert "msg.encoding != config_.encoding" in source
    assert "AudioFrame source_id and stream_id are required" in source
    assert "Unsupported audio layout" in source


def test_runtime_write_failure_fails_closed_without_reopen_retry() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    source = source_path.read_text(encoding="utf-8")
    playback_thread = source.split("void FaOutNode::playbackThread()")[1].split(
        "}  // namespace fa_out"
    )[0]

    assert "failClosed(" in playback_thread
    assert "openBackend()" not in playback_thread
    assert "ALSA device unavailable, dropping frame" not in playback_thread
    assert "snd_pcm_prepare" not in playback_thread
    assert "writeFrames(" in playback_thread
    assert "rclcpp::shutdown()" in source


def test_alsa_backend_files_are_ros_free() -> None:
    package_root = Path(__file__).parents[2]
    backend_paths = [
        package_root / "include" / "fa_out" / "backends" / "sink_backend.hpp",
        package_root / "include" / "fa_out" / "backends" / "alsa_playback_backend.hpp",
        package_root / "src" / "backends" / "sink_backend.cpp",
        package_root / "src" / "backends" / "alsa_playback_backend.cpp",
    ]
    forbidden_tokens = [
        "rclcpp",
        "fa_interfaces",
        "std_msgs/msg",
        "diagnostic_msgs",
        "rosidl",
    ]

    for backend_path in backend_paths:
        backend_text = backend_path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in backend_text


def test_backend_builds_as_separate_library() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_library(fa_out_backends" in cmake_text
    assert "src/backends/sink_backend.cpp" in cmake_text
    assert "src/backends/alsa_playback_backend.cpp" in cmake_text
    assert "target_link_libraries(fa_out_node fa_out_backends)" in cmake_text


def test_node_stores_abstract_sink_backend() -> None:
    package_root = Path(__file__).parents[2]
    header_text = (
        package_root / "include" / "fa_out" / "fa_out_node.hpp"
    ).read_text(encoding="utf-8")
    source_text = (package_root / "src" / "fa_out_node.cpp").read_text(encoding="utf-8")

    assert "#include \"fa_out/backends/sink_backend.hpp\"" in header_text
    assert "std::unique_ptr<backends::SinkBackend> sink_backend_;" in header_text
    assert "std::unique_ptr<backends::AlsaPlaybackBackend>" not in header_text
    assert "std::make_unique<backends::AlsaPlaybackBackend>" in source_text


def test_fa_out_node_header_does_not_store_alsa_handle() -> None:
    header_path = Path(__file__).parents[2] / "include" / "fa_out" / "fa_out_node.hpp"
    header_text = header_path.read_text(encoding="utf-8")

    assert "snd_pcm_t" not in header_text
    assert "#include <alsa/asoundlib.h>" not in header_text
    assert "std::unique_ptr<backends::SinkBackend> sink_backend_;" in header_text


def test_colcon_runs_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml

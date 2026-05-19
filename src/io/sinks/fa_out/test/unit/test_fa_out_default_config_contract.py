from pathlib import Path

import yaml


def test_default_config_requires_explicit_sink_device() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    params = config["fa_out"]["ros__parameters"]

    assert params["backend.name"] == "alsa_playback"
    assert params["input_topic"] == "fa_out/input"
    assert params["input_stream_id"] == "audio/playback/main"
    assert params["playback_done_topic"] == "fa_out/playback_done"
    assert params["playback_control_service"] == "fa_out/playback_control"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["playback_done_topic"] != params["input_topic"]
    assert params["audio.device_id"] == ""
    assert params["audio.encoding"] == "PCM16LE"
    assert params["audio.bit_depth"] == 16
    assert params["audio.chunk_duration_ms"] == 30
    assert params["audio.alsa.buffer_frames"] == 16384
    assert params["audio.alsa.period_frames"] == 4096
    assert params["audio.qos.depth"] == 10
    assert params["audio.qos.reliable"] is True
    assert params["lifecycle.qos.depth"] == 10
    assert params["lifecycle.qos.reliable"] is True
    assert '"default"' not in config_text


def test_sink_backend_has_no_struct_default() -> None:
    package_root = Path(__file__).parents[2]
    header_path = package_root / "include" / "fa_out" / "fa_out_node.hpp"
    validation_header_path = package_root / "include" / "fa_out" / "audio_config_validation.hpp"
    header_text = header_path.read_text(encoding="utf-8")
    validation_text = validation_header_path.read_text(encoding="utf-8")

    assert "std::string backend_name{};" in header_text
    assert "std::string input_topic{};" in header_text
    assert "std::string input_stream_id{};" in header_text
    assert "std::string playback_done_topic{};" in header_text
    assert "std::string playback_control_service{};" in header_text
    assert "std::string device_id{};" in header_text
    assert "std::string file_path{};" in header_text
    assert "std::string endpoint_uri{};" in header_text
    assert "std::string transport_identity{};" in header_text
    assert "std::string encoding{};" in header_text
    assert "uint32_t sample_rate{0};" in header_text
    assert "uint32_t channels{0};" in header_text
    assert "uint32_t bit_depth{0};" in header_text
    assert "size_t max_queue_frames{0};" in header_text
    assert "uint32_t chunk_duration_ms{0};" in header_text
    assert "size_t playback_chunk_frames{0};" in header_text
    assert "size_t playback_chunk_bytes{0};" in header_text
    assert "size_t alsa_buffer_frames{0};" in header_text
    assert "size_t alsa_period_frames{0};" in header_text
    assert "size_t qos_depth{0};" in header_text
    assert "size_t lifecycle_qos_depth{0};" in header_text
    assert "bool overwrite_enabled{false};" in header_text
    assert "size_t network_max_packet_bytes{0};" in header_text

    source_path = package_root / "src" / "fa_out_node.cpp"
    source_text = source_path.read_text(encoding="utf-8")
    assert 'declare_parameter<std::string>("input_topic")' in source_text
    assert 'declare_parameter<std::string>("input_stream_id")' in source_text
    assert 'declare_parameter<std::string>("playback_done_topic")' in source_text
    assert 'declare_parameter<std::string>("playback_control_service")' in source_text
    assert 'declare_parameter("file.path", rclcpp::ParameterValue{}, dynamic_parameter)' in source_text
    assert 'declare_parameter("overwrite.enabled", rclcpp::ParameterValue{}, dynamic_parameter)' in source_text
    assert 'declare_parameter("endpoint.uri", rclcpp::ParameterValue{}, dynamic_parameter)' in source_text
    assert 'declare_parameter("transport.identity", rclcpp::ParameterValue{}, dynamic_parameter)' in source_text
    assert 'declare_parameter("network.max_packet_bytes", rclcpp::ParameterValue{}, dynamic_parameter)' in source_text
    assert "readRequiredString(*this, \"input_topic\")" in source_text
    assert "requirePositiveUint32" in source_text
    assert "requirePositiveSize" in source_text
    assert "std::max<size_t>(1" not in source_text
    assert "validation::bytesPerFrame" in source_text
    assert "validation::bytesForFrames" in source_text
    assert (
        "audio.sample_rate * audio.chunk_duration_ms must produce an integer playback chunk"
        in validation_text
    )
    assert "audio.chunk_duration_ms produces zero playback frames" in validation_text
    assert "audio.channels * audio.bit_depth exceeds size_t range" in validation_text
    assert (package_root / "docs" / "backends" / "pcm_file_writer.md").is_file()
    assert (package_root / "docs" / "backends" / "network_pcm_sender.md").is_file()


def test_alsa_sink_rejects_plugin_pcm_devices() -> None:
    backend_source_path = (
        Path(__file__).parents[2] / "src" / "backends" / "alsa_playback_backend.cpp"
    )
    backend_source = backend_source_path.read_text(encoding="utf-8")
    validation_header = (
        Path(__file__).parents[2] / "include" / "fa_out" / "audio_config_validation.hpp"
    ).read_text(encoding="utf-8")

    assert "requireRawAlsaHardwareSink" in backend_source
    assert "requireRawAlsaHardwareSink" in validation_header
    assert 'rfind("hw:", 0)' in validation_header
    assert "audio.device_id must be an ALSA raw hardware id" in validation_header


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert 'declare_parameter<std::string>("input_topic")' in source
    assert 'declare_parameter<std::string>("input_stream_id")' in source
    assert 'declare_parameter<std::string>("playback_done_topic")' in source
    assert 'declare_parameter<std::string>("playback_control_service")' in source
    assert 'declare_parameter<std::string>("audio.encoding")' in source
    assert 'declare_parameter<int>("audio.sample_rate")' in source
    assert 'declare_parameter<int>("audio.channels")' in source
    assert 'declare_parameter<int>("audio.bit_depth")' in source
    assert 'declare_parameter("audio.alsa.buffer_frames", rclcpp::ParameterValue{}, dynamic_parameter)' in source
    assert 'declare_parameter("audio.alsa.period_frames", rclcpp::ParameterValue{}, dynamic_parameter)' in source
    assert 'declare_parameter<int>("queue.max_frames")' in source
    assert 'declare_parameter<int>("audio.chunk_duration_ms")' in source
    assert 'declare_parameter<int>("audio.qos.depth")' in source
    assert 'declare_parameter<bool>("audio.qos.reliable")' in source
    assert 'declare_parameter<int>("lifecycle.qos.depth")' in source
    assert 'declare_parameter<bool>("lifecycle.qos.reliable")' in source


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
    assert "msg.stream_id != config_.input_stream_id" in source
    assert "AudioFrame stream_id mismatch" in source
    assert "Unsupported audio layout" in source


def test_file_writer_backend_is_explicit_and_does_not_hide_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_out_node.cpp").read_text(encoding="utf-8")
    factory_source = (package_root / "src" / "backends" / "factory.cpp").read_text(
        encoding="utf-8"
    )
    backend_source = (
        package_root / "src" / "backends" / "pcm_file_writer_backend.cpp"
    ).read_text(encoding="utf-8")

    assert "kBackendNetworkPcmSender" in source
    assert "backend_settings.pcm_file_writer.file_path = config_.file_path;" in source
    assert "std::make_unique<PcmFileWriterBackend>" in factory_source
    assert "file.path is required for backend.name=pcm_file_writer" in source
    assert "audio.encoding/audio.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32" in source
    assert "codec" not in backend_source
    assert "resample" not in backend_source
    assert "gain" not in backend_source
    assert "stream_.write" in backend_source


def test_network_sender_backend_is_explicit_and_preserves_packet_boundary() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_out_node.cpp").read_text(encoding="utf-8")
    factory_source = (package_root / "src" / "backends" / "factory.cpp").read_text(
        encoding="utf-8"
    )
    backend_source = (
        package_root / "src" / "backends" / "network_pcm_sender_backend.cpp"
    ).read_text(encoding="utf-8")

    assert "backend_settings.network_pcm_sender.endpoint_uri = config_.endpoint_uri;" in source
    assert "std::make_unique<NetworkPcmSenderBackend>" in factory_source
    assert "endpoint.uri is required for backend.name=network_pcm_sender" in source
    assert "network.max_packet_bytes must be divisible by expected frame byte size" in source
    assert "config_.backend_name == kBackendNetworkPcmSender ? total_bytes" in source
    assert "sendto" in backend_source
    assert "codec" not in backend_source
    assert "resample" not in backend_source
    assert "gain" not in backend_source


def test_alsa_playback_backend_rejects_negotiated_timing_changes() -> None:
    backend_source_path = (
        Path(__file__).parents[2] / "src" / "backends" / "alsa_playback_backend.cpp"
    )
    backend_source = backend_source_path.read_text(encoding="utf-8")
    backend_header_path = (
        Path(__file__).parents[2]
        / "include"
        / "fa_out"
        / "backends"
        / "alsa_playback_backend.hpp"
    )
    backend_header = backend_header_path.read_text(encoding="utf-8")

    assert "size_t buffer_frames{0};" in backend_header
    assert "size_t period_frames{0};" in backend_header
    assert "snd_pcm_hw_params_set_rate_resample(handle, hw_params, 0)" in backend_source
    assert "Failed to disable ALSA software resampling" in backend_source
    assert backend_source.index("const snd_pcm_uframes_t requested_buffer_size") < backend_source.index(
        "snd_pcm_open"
    )
    assert backend_source.index("const snd_pcm_uframes_t requested_period_size") < backend_source.index(
        "snd_pcm_open"
    )
    assert "ALSA buffer size negotiation changed requested playback buffer" in backend_source
    assert "ALSA period size negotiation changed requested playback period" in backend_source
    assert "audio.alsa.period_frames must be <= audio.alsa.buffer_frames" in backend_source
    assert "open_info.warnings.emplace_back" not in backend_source
    assert "std::vector<std::string> warnings" not in backend_header_path.with_name(
        "sink_backend.hpp"
    ).read_text(encoding="utf-8")
    assert "snd_pcm_sw_params_current" in backend_source
    assert "snd_pcm_sw_params_set_start_threshold" in backend_source
    assert "snd_pcm_sw_params_set_avail_min" in backend_source
    assert "throw AlsaPlaybackError(alsaErrorMessage(\"snd_pcm_sw_params\"" in backend_source


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
    assert "writeBackendFrames(" in playback_thread
    assert "sink_backend_->writeFrames(" in source
    assert "rclcpp::shutdown()" in source


def test_queue_overflow_fails_closed_without_dropping_oldest_frame() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    source = source_path.read_text(encoding="utf-8")
    handle_frame = source.split("void FaOutNode::handleFrame")[1].split(
        "bool FaOutNode::validateFrame"
    )[0]

    assert "playback queue exceeded queue.max_frames" in handle_frame
    assert "failClosed(" in handle_frame
    assert "frame_queue_.pop_front()" not in handle_frame
    assert "dropping oldest frame" not in handle_frame


def test_playback_backend_access_is_serialized() -> None:
    package_root = Path(__file__).parents[2]
    header_text = (package_root / "include" / "fa_out" / "fa_out_node.hpp").read_text(
        encoding="utf-8"
    )
    source = (package_root / "src" / "fa_out_node.cpp").read_text(encoding="utf-8")
    playback_thread = source.split("void FaOutNode::playbackThread()")[1].split(
        "}  // namespace fa_out"
    )[0]
    handle_frame = source.split("void FaOutNode::handleFrame")[1].split(
        "bool FaOutNode::validateFrame"
    )[0]

    assert "std::mutex backend_mutex_;" in header_text
    assert "size_t writeBackendFrames(" in header_text
    assert "bool isBackendRunning();" in header_text
    assert "std::lock_guard<std::mutex> lock(backend_mutex_);" in source
    assert "sink_backend_" not in playback_thread
    assert "sink_backend_" not in handle_frame
    assert "writeBackendFrames(" in playback_thread
    assert "isBackendRunning()" in playback_thread


def test_missing_sink_backend_fails_closed_while_running() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_out_node.cpp"
    source = source_path.read_text(encoding="utf-8")
    is_backend_running = source.split("bool FaOutNode::isBackendRunning()")[1].split(
        "void FaOutNode::failClosed"
    )[0]

    assert "required sink backend missing while fa_out is running" in is_backend_running
    assert "failClosed(error_message);" in is_backend_running
    assert "return false;" in is_backend_running


def test_backend_implementation_files_are_ros_free() -> None:
    package_root = Path(__file__).parents[2]
    backend_paths = sorted(
        [
            *(package_root / "include" / "fa_out" / "backends").glob("*.hpp"),
            *(package_root / "src" / "backends").glob("*.cpp"),
        ]
    )
    forbidden_tokens = [
        "rclcpp",
        "fa_interfaces",
        "std_msgs/msg",
        "diagnostic_msgs",
        "rosidl",
    ]

    assert backend_paths
    for path in backend_paths:
        backend_text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in backend_text


def test_backend_builds_as_separate_library() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_library(fa_out_backends" in cmake_text
    assert "src/backends/sink_backend.cpp" in cmake_text
    assert "src/backends/alsa_playback_backend.cpp" in cmake_text
    assert "src/backends/factory.cpp" in cmake_text
    assert "add_library(fa_out_node_core" in cmake_text
    assert "src/fa_out_node.cpp" in cmake_text
    assert "src/main.cpp" in cmake_text
    assert "target_link_libraries(fa_out_node_core fa_out_backends)" in cmake_text
    assert "target_link_libraries(fa_out_node fa_out_node_core)" in cmake_text


def test_node_stores_abstract_sink_backend() -> None:
    package_root = Path(__file__).parents[2]
    header_text = (
        package_root / "include" / "fa_out" / "fa_out_node.hpp"
    ).read_text(encoding="utf-8")
    source_text = (package_root / "src" / "fa_out_node.cpp").read_text(encoding="utf-8")

    assert "#include \"fa_out/backends/sink_backend.hpp\"" in header_text
    assert "std::unique_ptr<backends::SinkBackend> sink_backend_;" in header_text
    assert "std::unique_ptr<backends::AlsaPlaybackBackend>" not in header_text
    assert "std::make_unique<backends::AlsaPlaybackBackend>" not in source_text
    assert "std::make_unique<backends::PcmFileWriterBackend>" not in source_text
    assert "std::make_unique<backends::NetworkPcmSenderBackend>" not in source_text


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
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_audio_config_validation_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_sink_backend_factory_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_node_contract_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_network_pcm_sender_backend_test" in cmake_text
    assert "test/cpp/test_sink_backend_factory.cpp" in cmake_text
    assert "test/cpp/test_network_pcm_sender_backend.cpp" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "find_package(std_msgs REQUIRED)" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "<depend>std_msgs</depend>" in package_xml
    assert "diagnostic_msgs" not in cmake_text
    assert "diagnostic_msgs" not in package_xml


def test_node_uses_ros_free_sink_backend_factory() -> None:
    package_root = Path(__file__).parents[2]
    node_source = (package_root / "src" / "fa_out_node.cpp").read_text(encoding="utf-8")
    header_text = (package_root / "include" / "fa_out" / "fa_out_node.hpp").read_text(
        encoding="utf-8"
    )
    factory_header = (
        package_root / "include" / "fa_out" / "backends" / "factory.hpp"
    ).read_text(encoding="utf-8")
    factory_source = (package_root / "src" / "backends" / "factory.cpp").read_text(
        encoding="utf-8"
    )
    factory_test = (
        package_root / "test" / "cpp" / "test_sink_backend_factory.cpp"
    ).read_text(encoding="utf-8")

    assert '#include "fa_out/backends/factory.hpp"' in node_source
    assert "auto backend = backends::buildSinkBackend(" in node_source
    assert "backends::SinkBackendSettings backend_settings;" in node_source
    assert "backend_settings.name = config_.backend_name;" in node_source
    assert '#include "fa_out/backends/alsa_playback_backend.hpp"' not in node_source
    assert '#include "fa_out/backends/pcm_file_writer_backend.hpp"' not in node_source
    assert '#include "fa_out/backends/network_pcm_sender_backend.hpp"' not in node_source
    assert "using BackendFactory = backends::AlsaPlaybackBackendFactory;" in header_text
    assert "struct SinkBackendSettings" in factory_header
    assert "AlsaPlaybackBackendFactory defaultAlsaPlaybackBackendFactory()" in factory_header
    assert "std::make_unique<AlsaPlaybackBackend>(config)" in factory_source
    assert "std::make_unique<PcmFileWriterBackend>(settings.pcm_file_writer)" in factory_source
    assert (
        "std::make_unique<NetworkPcmSenderBackend>(settings.network_pcm_sender)"
        in factory_source
    )
    assert "BuildsPcmFileWriterBackend" in factory_test
    assert "BuildsNetworkPcmSenderBackend" in factory_test
    assert "RejectsMissingBackendName" in factory_test
    assert "RejectsUnknownBackendName" in factory_test


def test_sink_adapter_exposes_only_playback_lifecycle_control_surface() -> None:
    package_root = Path(__file__).parents[2]
    source_text = (package_root / "src" / "fa_out_node.cpp").read_text(encoding="utf-8")
    header_text = (package_root / "include" / "fa_out" / "fa_out_node.hpp").read_text(
        encoding="utf-8"
    )
    config_text = (package_root / "config" / "default.yaml").read_text(encoding="utf-8")
    combined = "\n".join([source_text, header_text, config_text])

    assert "create_service<fa_interfaces::srv::PlaybackControl>" in source_text
    assert "fa_interfaces::msg::PlaybackDone" in combined
    assert "playback_done_topic" in combined
    assert "playback_control_service" in combined
    assert "active_epoch_" in combined
    assert "paused_" in combined
    assert "last_stop_time_" not in combined
    assert "audio/output/paused" not in combined
    forbidden_tokens = [
        "audio.file",
        "decode",
        "encoder",
        "gain",
        "normalize",
        "limiter",
        "resample",
        "volume",
        "sample_format",
        "channel_convert",
        "bit_depth_convert",
    ]
    for token in forbidden_tokens:
        assert token not in combined


def test_playback_done_and_control_contract_are_explicit() -> None:
    package_root = Path(__file__).parents[2]
    source_text = (package_root / "src" / "fa_out_node.cpp").read_text(encoding="utf-8")
    header_text = (package_root / "include" / "fa_out" / "fa_out_node.hpp").read_text(
        encoding="utf-8"
    )

    assert "fa_interfaces/msg/playback_done.hpp" in header_text
    assert "fa_interfaces/srv/playback_control.hpp" in header_text
    assert "void publishPlaybackDone(const QueuedFrame & queued_frame);" in header_text
    assert "void handlePlaybackControl(" in header_text
    assert "done_msg.header = queued_frame.header;" in source_text
    assert "done_msg.request_id = queued_frame.request_id;" in source_text
    assert "done_msg.epoch = queued_frame.epoch;" in source_text
    assert 'request->command == kPlaybackCommandStop' in source_text
    assert "active_epoch_.store(next_epoch);" in source_text
    assert "frame_queue_.clear();" in source_text
    assert "paused_.store(true);" in source_text
    assert "paused_.store(false);" in source_text
    assert "queued_frame.epoch < active_epoch_.load()" in source_text
    assert "Rejecting stale playback frame epoch" in source_text
    assert "unsupported playback control command" in source_text


def test_sink_backend_exposes_no_session_control_primitive() -> None:
    package_root = Path(__file__).parents[2]
    backend_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            package_root / "include" / "fa_out" / "backends" / "sink_backend.hpp",
            package_root / "include" / "fa_out" / "backends" / "alsa_playback_backend.hpp",
            package_root / "src" / "backends" / "alsa_playback_backend.cpp",
        ]
    )

    assert "discardBuffer" not in backend_text
    assert "snd_pcm_drop" not in backend_text
    assert "snd_pcm_prepare" not in backend_text


def test_backend_open_result_is_info_only_without_warning_continuation() -> None:
    package_root = Path(__file__).parents[2]
    sink_backend_header = (
        package_root / "include" / "fa_out" / "backends" / "sink_backend.hpp"
    ).read_text(encoding="utf-8")
    node_source = (package_root / "src" / "fa_out_node.cpp").read_text(
        encoding="utf-8"
    )
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package_root / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    test_plan = (package_root / "docs" / "テスト設計.md").read_text(
        encoding="utf-8"
    )
    backend_doc = (package_root / "docs" / "backends" / "alsa.md").read_text(
        encoding="utf-8"
    )

    assert "std::vector<std::string> info_messages;" in sink_backend_header
    assert "warnings" not in sink_backend_header
    assert "open_info.info_messages" in node_source
    assert "open_info.warnings" not in node_source
    assert "warning 継続チャネルを持たない" in spec
    assert "warning で継続する経路は持たない" in algorithm
    assert "FA-OUT-SPEC-021" in test_plan
    assert "warning で継続する output は持たない" in backend_doc

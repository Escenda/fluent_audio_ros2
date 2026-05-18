from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_requires_explicit_execution_provider() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_kws"]["ros__parameters"]

    assert params["backend.name"] == "sherpa_onnx_kws"
    assert params["backend.execution_provider"] == ""
    assert params["expected_source_id"] == ""
    assert params["audio_topic"] == "audio/frame"
    assert params["expected_stream_id"] == "audio/raw/mic"
    assert params["audio_topic"] != params["expected_stream_id"]
    assert params["vad.max_age_ms"] == 1000
    assert params["audio.qos.depth"] == 10
    assert params["audio.qos.reliable"] is False
    assert params["vad.qos.depth"] == 20
    assert params["vad.qos.reliable"] is False
    assert params["result.qos.depth"] == 10
    assert params["result.qos.reliable"] is False
    assert "model.provider" not in params


def test_backend_config_has_no_provider_default() -> None:
    header_path = (
        PACKAGE_ROOT
        / "include"
        / "fa_kws"
        / "backends"
        / "sherpa_onnx_kws_backend.hpp"
    )
    interface_path = (
        PACKAGE_ROOT / "include" / "fa_kws" / "backends" / "kws_backend.hpp"
    )

    header_text = header_path.read_text(encoding="utf-8")
    interface_text = interface_path.read_text(encoding="utf-8")

    assert "std::string execution_provider;" in header_text
    assert "std::string execution_provider{};" not in header_text
    assert "class KwsBackend" in interface_text
    assert "struct KwsDetection" in interface_text
    assert "class SherpaOnnxKwsBackend final : public KwsBackend" in header_text
    assert "model_provider" not in header_text
    assert '{"cpu"}' not in header_text
    assert "target_sample_rate{16000}" not in header_text
    assert "model_num_threads{4}" not in header_text
    assert "max_active_paths{4}" not in header_text
    assert "keywords_score{1.0f}" not in header_text
    assert "keywords_threshold{0.25f}" not in header_text
    assert "vad_threshold{0.35f}" not in header_text
    assert "std::chrono::milliseconds{2000}" not in header_text


def test_cmake_accepts_sherpa_prefix_from_environment() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "set(CMAKE_CXX_STANDARD 17)" in cmake_text
    assert "set(CMAKE_CXX_STANDARD_REQUIRED ON)" in cmake_text
    assert 'DEFINED ENV{SHERPA_ONNX_PREFIX}' in cmake_text
    assert '$ENV{SHERPA_ONNX_PREFIX}' in cmake_text


def test_cmake_requires_sherpa_for_runtime_targets() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    node_text = (PACKAGE_ROOT / "src" / "fa_kws_node.cpp").read_text(encoding="utf-8")

    assert 'set(FA_KWS_SHERPA_ONNX "ON"' in cmake_text
    assert 'FA_KWS_SHERPA_ONNX MATCHES "^(ON|OFF)$"' in cmake_text
    assert 'FA_KWS_SHERPA_ONNX STREQUAL "ON"' in cmake_text
    assert "FA_KWS_WITH_SHERPA_ONNX" in cmake_text
    assert "FA_KWS_SHERPA_ONNX=OFF explicitly disables fa_kws runtime targets" in cmake_text
    assert "add_executable(fa_kws_node" in cmake_text
    assert "fa_kws was built without sherpa-onnx support" not in node_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "message(FATAL_ERROR" in cmake_text
    assert "if(FA_KWS_WITH_SHERPA_ONNX)" in cmake_text
    assert "install(TARGETS fa_kws_node fa_kws_wav_tool" in cmake_text


def test_backend_builds_as_shared_runtime_boundary() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_library(fa_kws_backends STATIC" in cmake_text
    assert "src/backends/sherpa_onnx_kws_backend.cpp" in cmake_text
    assert "target_link_libraries(fa_kws_node" in cmake_text
    assert "fa_kws_backends" in cmake_text
    assert "target_link_libraries(fa_kws_wav_tool\n    fa_kws_backends" in cmake_text
    assert "src/backends/sherpa_onnx_kws_backend.cpp" not in cmake_text.split(
        "add_executable(fa_kws_node"
    )[1].split(")")[0]
    assert "src/backends/sherpa_onnx_kws_backend.cpp" not in cmake_text.split(
        "add_executable(fa_kws_wav_tool"
    )[1].split(")")[0]


def test_node_uses_backend_execution_provider_parameter() -> None:
    node_path = PACKAGE_ROOT / "src" / "fa_kws_node.cpp"
    node_text = node_path.read_text(encoding="utf-8")
    identity_text = (PACKAGE_ROOT / "src" / "vad_state_identity.cpp").read_text(
        encoding="utf-8"
    )
    identity_test_text = (
        PACKAGE_ROOT / "test" / "unit" / "vad_state_identity_contract.cpp"
    ).read_text(encoding="utf-8")
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'declare_parameter<std::string>("backend.execution_provider");' in node_text
    assert 'declare_parameter<std::string>("expected_source_id");' in node_text
    assert 'declare_parameter<std::string>("expected_stream_id");' in node_text
    assert 'declare_parameter<std::string>("audio_topic");' in node_text
    assert 'declare_parameter<int>("target_sample_rate");' in node_text
    assert "expected_source_id is required" in node_text
    assert "validateTopicBindingsOrThrow();" in node_text
    assert "frameToCanonicalFloat(*msg, expected_source_id_, expected_stream_id_)" in node_text
    assert 'declare_parameter<double>("vad.probability_gate");' in node_text
    assert "backend.execution_provider is required" in node_text
    assert 'declare_parameter<int>("vad.max_age_ms");' in node_text
    assert 'declare_parameter<int>("cooldown_ms");' in node_text
    assert 'declare_parameter<double>("debug.status_period_sec");' in node_text
    assert 'declare_parameter<int>("audio.qos.depth");' in node_text
    assert 'declare_parameter<bool>("audio.qos.reliable");' in node_text
    assert 'declare_parameter<int>("vad.qos.depth");' in node_text
    assert 'declare_parameter<bool>("vad.qos.reliable");' in node_text
    assert 'declare_parameter<int>("result.qos.depth");' in node_text
    assert 'declare_parameter<bool>("result.qos.reliable");' in node_text
    assert 'declare_parameter<int>("model.num_threads");' in node_text
    assert 'declare_parameter<int>("kws.max_active_paths");' in node_text
    assert 'declare_parameter<int>("kws.num_trailing_blanks");' in node_text
    assert 'declare_parameter<double>("kws.keywords_score");' in node_text
    assert 'declare_parameter<double>("kws.keywords_threshold");' in node_text
    assert 'declare_parameter<std::string>("audio_topic", "audio/frame")' not in node_text
    assert 'declare_parameter<int>("target_sample_rate", 16000)' not in node_text
    assert 'declare_parameter<double>("vad.probability_gate", 0.35)' not in node_text
    assert 'declare_parameter<int>("vad.max_age_ms", 1000)' not in node_text
    assert "rclcpp::KeepLast(10)" not in node_text
    assert "rclcpp::KeepLast(20)" not in node_text
    assert "const rclcpp::QoS qos_audio = makeExplicitQos(audio_qos_depth_, audio_qos_reliable_);" in node_text
    assert "const rclcpp::QoS qos_vad = makeExplicitQos(vad_qos_depth_, vad_qos_reliable_);" in node_text
    assert "const rclcpp::QoS qos_result = makeExplicitQos(result_qos_depth_, result_qos_reliable_);" in node_text
    assert "audio.qos.depth must be greater than zero" in node_text
    assert "vad.qos.depth must be greater than zero" in node_text
    assert "result.qos.depth must be greater than zero" in node_text
    assert "vad.probability_gate must be finite and in (0.0, 1.0]" in node_text
    assert "vad.max_age_ms must be greater than zero" in node_text
    assert "expected_stream_id is required" in node_text
    assert "expected_stream_id must be distinct from ROS audio_topic" in node_text
    assert "expected_stream_id must be distinct from ROS vad_topic" in node_text
    assert "expected_stream_id must be distinct from ROS output_topic" in node_text
    assert "vadStateMatchesAudioBinding(*msg, expected_source_id_, expected_stream_id_)" in node_text
    assert "msg.source_id.empty() || msg.stream_id.empty()" in identity_text
    assert "expected_source_id.empty() || expected_stream_id.empty()" in identity_text
    assert "msg.source_id == expected_source_id && msg.stream_id == expected_stream_id" in identity_text
    assert "Rejecting VadState identity mismatch" in node_text
    assert "Rejecting invalid VadState.probability" in node_text
    assert "last_vad_rx_ns_.store(0" in node_text
    assert "ament_add_gtest(${PROJECT_NAME}_vad_state_identity_test" in cmake_text
    assert "RejectsMissingVadIdentity" in identity_test_text
    assert "RejectsUnexpectedSourceOrStream" in identity_test_text
    assert "isValidVadGateThreshold(probability_gate_)" in node_text
    assert "isValidVadProbability(msg->probability)" in node_text
    assert "passesVadGate(vad_prob, static_cast<float>(probability_gate_))" in node_text
    assert "readFreshVadProbability" in node_text
    assert "model.provider" not in node_text

    gate_index = node_text.index("passesVadGate(vad_prob, static_cast<float>(probability_gate_))")
    process_index = node_text.index("kws_backend_->process")
    assert gate_index < process_index


def test_model_file_validation_rejects_non_regular_or_unreadable_paths() -> None:
    node_text = (PACKAGE_ROOT / "src" / "fa_kws_node.cpp").read_text(encoding="utf-8")
    spec_text = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")

    assert "std::filesystem::is_regular_file(path, ec)" in node_text
    assert "not a regular file" in node_text
    assert "std::ifstream probe(path, std::ios::binary)" in node_text
    assert "not readable" in node_text
    assert "model path missing / not a regular readable file" in spec_text


def test_backend_keeps_vad_gate_mandatory() -> None:
    header_text = (
        PACKAGE_ROOT
        / "include"
        / "fa_kws"
        / "backends"
        / "sherpa_onnx_kws_backend.hpp"
    ).read_text(encoding="utf-8")
    backend_text = (
        PACKAGE_ROOT / "src" / "backends" / "sherpa_onnx_kws_backend.cpp"
    ).read_text(encoding="utf-8")
    tool_text = (PACKAGE_ROOT / "src" / "kws_wav_tool.cpp").read_text(
        encoding="utf-8"
    )

    assert "float vad_threshold;" in header_text
    assert "float vad_threshold{0.35f};" not in header_text
    assert "isValidVadGateThreshold" in backend_text
    assert "isValidVadProbability" in backend_text
    assert "vad_threshold must be finite and in (0.0, 1.0]" in backend_text
    assert "vad_prob must be finite and in [0.0, 1.0]" in backend_text
    assert "passesVadGate(vad_prob, config_.vad_threshold)" in backend_text
    assert "SherpaOnnxResetKeywordStream(spotter_, stream_);" in backend_text
    assert "config_.vad_threshold > 0.0f &&" not in backend_text
    assert 'treat it as "no VAD gating"' not in backend_text
    assert "disable gating" not in tool_text
    assert "cfg.vad_threshold = 1.0f;" in tool_text
    assert "/*vad_prob=*/1.0f" in tool_text
    assert "--assume-vad-speech" in tool_text
    assert "kws_wav_tool requires explicit VAD policy" in tool_text
    assert "vad_policy=assume_speech" in tool_text

    gate_index = backend_text.index("passesVadGate(vad_prob, config_.vad_threshold)")
    accept_index = backend_text.index("SherpaOnnxOnlineStreamAcceptWaveform")
    assert gate_index < accept_index


def test_vad_gate_has_executable_contract_test() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")
    test_text = (PACKAGE_ROOT / "test" / "unit" / "vad_gate_contract.cpp").read_text(
        encoding="utf-8"
    )

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "ament_add_gtest(${PROJECT_NAME}_vad_gate_test" in cmake_text
    assert "test/unit/vad_gate_contract.cpp" in cmake_text
    assert "ProbabilityMustBeFiniteAndNormalized" in test_text
    assert "ThresholdCannotDisableGate" in test_text
    assert "GateIsInclusiveAndFailClosed" in test_text


def test_audio_utils_has_executable_contract_test() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    test_text = (
        PACKAGE_ROOT / "test" / "unit" / "audio_utils_contract.cpp"
    ).read_text(encoding="utf-8")

    assert "ament_add_gtest(${PROJECT_NAME}_audio_utils_test" in cmake_text
    assert "test/unit/audio_utils_contract.cpp" in cmake_text
    assert "src/audio_utils.cpp" in cmake_text
    assert "RejectsPcm32PayloadBeforeFloatInterpretation" in test_text
    assert "RejectsMissingExpectedSourceBinding" in test_text
    assert "RejectsMissingExpectedStreamBinding" in test_text
    assert "RejectsUnexpectedSourceIdentity" in test_text
    assert "RejectsUnexpectedStreamIdentity" in test_text
    assert 'msg.encoding = "PCM32LE"' in test_text


def test_kws_node_rejects_non_canonical_audio_frames() -> None:
    header_text = (PACKAGE_ROOT / "include" / "fa_kws" / "audio_utils.hpp").read_text(
        encoding="utf-8"
    )
    audio_utils_text = (PACKAGE_ROOT / "src" / "audio_utils.cpp").read_text(
        encoding="utf-8"
    )
    node_text = (PACKAGE_ROOT / "src" / "fa_kws_node.cpp").read_text(encoding="utf-8")

    assert "frameToCanonicalFloat" in header_text
    assert "resampleLinear" not in header_text
    assert "reinterpret_cast<const std::int16_t *>" not in audio_utils_text
    assert "AudioFrame channels must be 1" in audio_utils_text
    assert "AudioFrame source_id and stream_id are required" in audio_utils_text
    assert "expected_source_id is required" in audio_utils_text
    assert "expected_stream_id is required" in audio_utils_text
    assert "AudioFrame source_id must match expected_source_id" in audio_utils_text
    assert "AudioFrame stream_id must match expected_stream_id" in audio_utils_text
    assert "AudioFrame layout must be interleaved" in audio_utils_text
    assert "AudioFrame encoding must be FLOAT32LE" in audio_utils_text
    assert "AudioFrame bit_depth must be 32" in audio_utils_text
    assert "AudioFrame data is required" in audio_utils_text
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in audio_utils_text
    assert "sherpa-onnx will resample internally" not in node_text
    assert "Dropping AudioFrame with sample_rate" in node_text
    assert "if (samples.empty())" not in node_text
    assert "dump_audio" not in node_text
    assert "writeWav" not in node_text
    assert "capture_buffer_" not in node_text
    assert "std::int16_t" not in node_text
    assert "WAVE" not in node_text


def test_detection_score_is_owned_by_backend() -> None:
    interface_path = (
        PACKAGE_ROOT / "include" / "fa_kws" / "backends" / "kws_backend.hpp"
    )
    backend_path = PACKAGE_ROOT / "src" / "backends" / "sherpa_onnx_kws_backend.cpp"
    node_path = PACKAGE_ROOT / "src" / "fa_kws_node.cpp"

    assert "float score{1.0f};" in interface_path.read_text(encoding="utf-8")
    assert "det.score = 1.0f;" in backend_path.read_text(encoding="utf-8")
    assert "out.score = detection->score;" in node_path.read_text(encoding="utf-8")
    assert "std::unique_ptr<KwsBackend> kws_backend_;" in node_path.read_text(
        encoding="utf-8"
    )


def test_wav_tool_requires_explicit_provider() -> None:
    tool_path = PACKAGE_ROOT / "src" / "kws_wav_tool.cpp"
    tool_text = tool_path.read_text(encoding="utf-8")

    assert "--provider <provider>" in tool_text
    assert "args.provider.empty()" in tool_text
    assert "cfg.execution_provider = args.provider;" in tool_text


def test_wav_tool_rejects_non_canonical_wav_without_hidden_conversion() -> None:
    tool_path = PACKAGE_ROOT / "src" / "kws_wav_tool.cpp"
    tool_text = tool_path.read_text(encoding="utf-8")

    assert "resample_linear" not in tool_text
    assert "resampled" not in tool_text
    assert "WAV sample_rate must match --sample_rate" in tool_text
    assert "WAV channels must be 1" in tool_text
    assert "WAV bit_depth must be 32" in tool_text
    assert "WAV must be IEEE float format" in tool_text
    assert "WAV float32 data length is not byte-aligned" in tool_text
    assert "WAV samples must be normalized to [-1.0, 1.0]" in tool_text
    assert "Failed to read WAV data chunk" in tool_text
    assert "std::memcpy(&sample" in tool_text
    assert "sum / static_cast<float>(channels)" not in tool_text
    assert "reinterpret_cast<const std::int16_t *>" not in tool_text
    assert "reinterpret_cast<const float *>" not in tool_text


def test_wav_tool_validates_wav_before_backend_initialization() -> None:
    tool_path = PACKAGE_ROOT / "src" / "kws_wav_tool.cpp"
    tool_text = tool_path.read_text(encoding="utf-8")

    batch_validation = tool_text.index("for (const auto &wav_path : wav_files)")
    batch_engine = tool_text.index("fa_kws::SherpaOnnxKwsBackend engine(cfg)", batch_validation)
    single_validation = tool_text.index("validate_wav_contract(wav, args, args.wav_path)")
    single_engine = tool_text.index("fa_kws::SherpaOnnxKwsBackend engine(cfg)", single_validation)

    assert batch_validation < batch_engine
    assert single_validation < single_engine


def test_wav_tool_initializes_every_backend_config_field_explicitly() -> None:
    tool_text = (PACKAGE_ROOT / "src" / "kws_wav_tool.cpp").read_text(encoding="utf-8")

    assert "float keywords_threshold{std::numeric_limits<float>::quiet_NaN()};" in tool_text
    assert "int target_sample_rate{0};" in tool_text
    assert "int model_num_threads{0};" in tool_text
    assert "int max_active_paths{0};" in tool_text
    assert "int num_trailing_blanks{-1};" in tool_text
    assert "float keywords_score{std::numeric_limits<float>::quiet_NaN()};" in tool_text
    assert 'arg == "--num_threads"' in tool_text
    assert 'arg == "--num_trailing_blanks"' in tool_text
    assert 'arg == "--keywords_score"' in tool_text
    assert "num_threads must be positive" in tool_text
    assert "max_active_paths must be positive" in tool_text
    assert "num_trailing_blanks must be zero or positive" in tool_text
    assert "keywords_score must be finite and positive" in tool_text
    assert "threshold must be finite and positive" in tool_text
    assert "build_backend_config(const Args &args)" in tool_text
    for assignment in (
        "cfg.target_sample_rate = args.target_sample_rate;",
        "cfg.model_num_threads = args.model_num_threads;",
        "cfg.execution_provider = args.provider;",
        "cfg.encoder_path = args.encoder;",
        "cfg.decoder_path = args.decoder;",
        "cfg.joiner_path = args.joiner;",
        "cfg.tokens_path = args.tokens;",
        "cfg.keywords_path = args.keywords;",
        "cfg.max_active_paths = args.max_active_paths;",
        "cfg.num_trailing_blanks = args.num_trailing_blanks;",
        "cfg.keywords_score = args.keywords_score;",
        "cfg.keywords_threshold = args.keywords_threshold;",
        "cfg.vad_threshold = 1.0f;",
        "cfg.cooldown = std::chrono::milliseconds{0};",
    ):
        assert assignment in tool_text

    config_index = tool_text.index(
        "const fa_kws::SherpaOnnxKwsBackendConfig cfg = build_backend_config(args);"
    )
    first_engine_index = tool_text.index("fa_kws::SherpaOnnxKwsBackend engine(cfg)")
    assert config_index < first_engine_index


def test_backend_boundary_fails_closed_for_invalid_runtime_state() -> None:
    backend_text = (
        PACKAGE_ROOT / "src" / "backends" / "sherpa_onnx_kws_backend.cpp"
    ).read_text(encoding="utf-8")
    header_text = (
        PACKAGE_ROOT
        / "include"
        / "fa_kws"
        / "backends"
        / "sherpa_onnx_kws_backend.hpp"
    ).read_text(encoding="utf-8")

    assert "void validateConfig() const;" in header_text
    assert "void requireReady(const char *operation) const;" in header_text
    assert "requireReady(\"process\");" in backend_text
    assert "KWS backend samples are required" in backend_text
    assert "sample_rate must match configured target_sample_rate" in backend_text
    assert "SherpaOnnxKwsBackend resetHard requested without keyword spotter" in backend_text
    assert "failed during resetHard" in backend_text
    assert "if (!spotter_ || !stream_ || samples.empty())" not in backend_text
    assert "return std::nullopt;" in backend_text


def test_backend_validates_model_paths_and_numeric_config_before_c_api() -> None:
    backend_text = (
        PACKAGE_ROOT / "src" / "backends" / "sherpa_onnx_kws_backend.cpp"
    ).read_text(encoding="utf-8")

    assert backend_text.index("validateConfig();") < backend_text.index("SherpaOnnxFeatureConfig")
    assert "requireReadableRegularFile(\"backend.encoder\"" in backend_text
    assert "requireReadableRegularFile(\"backend.decoder\"" in backend_text
    assert "requireReadableRegularFile(\"backend.joiner\"" in backend_text
    assert "requireReadableRegularFile(\"backend.tokens\"" in backend_text
    assert "requireReadableRegularFile(\"backend.keywords\"" in backend_text
    assert "backend.target_sample_rate must be > 0" in backend_text
    assert "backend.model_num_threads must be > 0" in backend_text
    assert "backend.max_active_paths must be > 0" in backend_text
    assert "backend.num_trailing_blanks must be >= 0" in backend_text
    assert "backend.keywords_score must be finite and > 0" in backend_text
    assert "backend.keywords_threshold must be finite and > 0" in backend_text

from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_requires_explicit_execution_provider() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_kws"]["ros__parameters"]

    assert params["backend.name"] == "sherpa_onnx_kws"
    assert params["backend.execution_provider"] == ""
    assert params["vad.max_age_ms"] == 1000
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

    assert "std::string execution_provider{};" in header_text
    assert "class KwsBackend" in interface_text
    assert "struct KwsDetection" in interface_text
    assert "class SherpaOnnxKwsBackend final : public KwsBackend" in header_text
    assert "model_provider" not in header_text
    assert '{"cpu"}' not in header_text


def test_cmake_accepts_sherpa_prefix_from_environment() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'DEFINED ENV{SHERPA_ONNX_PREFIX}' in cmake_text
    assert '$ENV{SHERPA_ONNX_PREFIX}' in cmake_text


def test_cmake_has_explicit_test_only_mode_without_sherpa() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "FA_KWS_ALLOW_TEST_ONLY_WITHOUT_SHERPA" in cmake_text
    assert "Runtime targets fa_kws_node/fa_kws_wav_tool will not be built" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "message(FATAL_ERROR" in cmake_text
    assert "if(FA_KWS_HAS_SHERPA_ONNX)" in cmake_text
    assert "add_executable(fa_kws_node" in cmake_text


def test_node_uses_backend_execution_provider_parameter() -> None:
    node_path = PACKAGE_ROOT / "src" / "fa_kws_node.cpp"
    node_text = node_path.read_text(encoding="utf-8")

    assert '"backend.execution_provider", ""' in node_text
    assert "backend.execution_provider is required" in node_text
    assert '"vad.max_age_ms", 1000' in node_text
    assert "vad.max_age_ms must be greater than zero" in node_text
    assert "readFreshVadProbability" in node_text
    assert "model.provider" not in node_text


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
    assert "AudioFrame layout must be interleaved" in audio_utils_text
    assert "AudioFrame bit_depth must be 32" in audio_utils_text
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in audio_utils_text
    assert "sherpa-onnx will resample internally" not in node_text
    assert "Dropping AudioFrame with sample_rate" in node_text


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

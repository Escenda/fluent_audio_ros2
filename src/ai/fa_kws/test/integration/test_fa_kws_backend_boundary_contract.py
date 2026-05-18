from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_sherpa_backend_source_stays_ros_free() -> None:
    backend_files = tuple(
        sorted((PACKAGE_ROOT / "include" / "fa_kws" / "backends").glob("*.[hc]pp"))
    ) + tuple(
        sorted((PACKAGE_ROOT / "src" / "backends").glob("*.[hc]pp"))
    )
    assert backend_files
    forbidden_tokens = (
        "rclcpp",
        "fa_interfaces",
        "AudioFrame",
        "VadState",
        "WakeWordResult",
    )

    for backend_file in backend_files:
        source = backend_file.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source


def test_native_backend_is_explicit_runtime_boundary_not_dummy_fallback() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    docs_text = (
        PACKAGE_ROOT / "docs" / "backends" / "sherpa_onnx_kws.md"
    ).read_text(encoding="utf-8")

    assert 'set(FA_KWS_SHERPA_ONNX "OFF"' in cmake_text
    assert 'FA_KWS_SHERPA_ONNX MATCHES "^(ON|OFF)$"' in cmake_text
    assert 'FA_KWS_SHERPA_ONNX STREQUAL "ON"' in cmake_text
    assert "FA_KWS_SHERPA_ONNX=OFF builds fa_kws runtime targets" in cmake_text
    assert "message(FATAL_ERROR" in cmake_text
    assert "add_library(fa_kws_backends STATIC" in cmake_text
    assert "sherpa_onnx_kws_backend_unavailable.cpp" in cmake_text
    assert "install(TARGETS fa_kws_node fa_kws_wav_tool" in cmake_text
    assert "dummy" not in cmake_text.lower()
    assert "unavailable backend が明示的に起動失敗" in docs_text

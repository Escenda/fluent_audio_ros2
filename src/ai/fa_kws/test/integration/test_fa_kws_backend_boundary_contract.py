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


def test_kws_backend_is_external_worker_boundary_not_native_link() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    docs_text = (
        PACKAGE_ROOT / "docs" / "backends" / "sherpa_onnx_kws.md"
    ).read_text(encoding="utf-8")
    backend_text = (
        PACKAGE_ROOT / "src" / "backends" / "sherpa_onnx_kws_backend.cpp"
    ).read_text(encoding="utf-8")

    assert "FA_KWS_SHERPA_ONNX" not in cmake_text
    assert "SHERPA_ONNX_PREFIX" not in cmake_text
    assert "sherpa-onnx/c-api" not in backend_text
    assert "execvp(command.c_str(), argv.data())" in backend_text
    assert "backend.command" in docs_text
    assert "External worker" in docs_text
    assert "add_library(fa_kws_backends STATIC" in cmake_text
    assert "sherpa_onnx_kws_backend_unavailable.cpp" not in cmake_text
    assert "install(TARGETS fa_kws_node fa_kws_wav_tool" in cmake_text
    assert "dummy" not in cmake_text.lower()

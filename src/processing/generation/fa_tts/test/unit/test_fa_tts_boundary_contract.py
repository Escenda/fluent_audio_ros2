from pathlib import Path

import yaml

from fa_tts_py.backends.factory import build_tts_backend
from fa_tts_py.backends.pyopenjtalk_backend import PyOpenJTalkBackend


def test_default_config_has_no_playback_or_gain_parameters() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_tts"]["ros__parameters"]

    assert params["backend.name"] == "pyopenjtalk"
    assert params["backend.openjtalk_dict_dir"] == ""
    assert params["output_topic"] == "audio/tts/frame"
    assert "playback_topic" not in params
    assert "use_playback_topic" not in params
    assert "stop_topic" not in params
    assert "default_volume_db" not in params


def test_tts_node_does_not_publish_to_playback_topic() -> None:
    source_path = Path(__file__).parents[2] / "fa_tts_py" / "tts_node.py"
    source = source_path.read_text(encoding="utf-8")

    assert "playback_topic" not in source
    assert "use_playback_topic" not in source
    assert "play_pub" not in source
    assert "create_subscription(Empty" not in source
    assert "request.play is not supported by fa_tts" in source
    assert "request.volume_db is not supported by fa_tts" in source
    assert "pyopenjtalk.tts" not in source
    assert "import pyopenjtalk" not in source


def test_tts_publishes_audio_frame_identity_without_analysis_fields() -> None:
    source_path = Path(__file__).parents[2] / "fa_tts_py" / "tts_node.py"
    source = source_path.read_text(encoding="utf-8")
    build_frame = source.split("def build_frame")[1].split("def make_cache_key")[0]

    assert 'frame.source_id = "fa_tts"' in build_frame
    assert "frame.stream_id = self.output_topic" in build_frame
    assert 'frame.layout = "interleaved"' in build_frame
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in build_frame


def test_tts_backend_is_ros_free_and_selected_by_backend_name() -> None:
    package_root = Path(__file__).parents[2]
    node_text = (package_root / "fa_tts_py" / "tts_node.py").read_text(encoding="utf-8")
    backend_text = (
        package_root / "fa_tts_py" / "backends" / "pyopenjtalk_backend.py"
    ).read_text(encoding="utf-8")
    factory_text = (
        package_root / "fa_tts_py" / "backends" / "factory.py"
    ).read_text(encoding="utf-8")

    assert "build_tts_backend(" in node_text
    assert "openjtalk_dict_dir=self.openjtalk_dict_dir" in node_text
    assert "backend.name" in node_text
    assert "import rclpy" not in backend_text
    assert "fa_interfaces" not in backend_text
    assert "AudioFrame" not in backend_text
    assert "pyopenjtalk" in backend_text
    assert "unsupported fa_tts backend.name" in factory_text


def test_tts_backend_factory_rejects_missing_or_unknown_backend() -> None:
    try:
        build_tts_backend("", openjtalk_dict_dir="")
    except RuntimeError as exc:
        assert str(exc) == "backend.name is required"
    else:
        raise AssertionError("missing backend.name was accepted")

    try:
        build_tts_backend("cloud_fallback", openjtalk_dict_dir="")
    except RuntimeError as exc:
        assert str(exc) == "unsupported fa_tts backend.name: cloud_fallback"
    else:
        raise AssertionError("unknown backend.name was accepted")


def test_pyopenjtalk_backend_does_not_import_runtime_until_selected() -> None:
    backend = PyOpenJTalkBackend.__new__(PyOpenJTalkBackend)

    assert backend.name == "pyopenjtalk"


def test_pyopenjtalk_backend_requires_explicit_dictionary_without_home_fallback() -> None:
    package_root = Path(__file__).parents[2]
    backend_text = (
        package_root / "fa_tts_py" / "backends" / "pyopenjtalk_backend.py"
    ).read_text(encoding="utf-8")

    assert "backend.openjtalk_dict_dir is required for pyopenjtalk" in backend_text
    assert "os.environ[\"OPEN_JTALK_DICT_DIR\"]" in backend_text
    assert "setdefault" not in backend_text
    assert "Path.home" not in backend_text

    try:
        PyOpenJTalkBackend(openjtalk_dict_dir="")
    except RuntimeError as exc:
        assert str(exc) == "backend.openjtalk_dict_dir is required for pyopenjtalk"
    else:
        raise AssertionError("missing backend.openjtalk_dict_dir was accepted")


def test_pyopenjtalk_backend_has_no_hidden_scale_guessing_or_clipping() -> None:
    package_root = Path(__file__).parents[2]
    backend_text = (
        package_root / "fa_tts_py" / "backends" / "pyopenjtalk_backend.py"
    ).read_text(encoding="utf-8")
    base_text = (package_root / "fa_tts_py" / "backends" / "base.py").read_text(
        encoding="utf-8"
    )
    node_text = (package_root / "fa_tts_py" / "tts_node.py").read_text(
        encoding="utf-8"
    )

    assert "np.clip" not in backend_text
    assert "32768.0" not in backend_text
    assert "waveform /=" not in backend_text
    assert "astype(np.int16)" not in backend_text
    assert 'encoding="FLOAT32LE"' in backend_text
    assert "bit_depth=32" in backend_text
    assert "pyopenjtalk waveform must be normalized to [-1.0, 1.0]" in backend_text
    assert "encoding: str" in base_text
    assert "encoding_value" in node_text
    assert "frame.encoding = cached.encoding" in node_text
    assert "encoding:{cached.encoding}" in node_text
    assert "cache metadata bit_depth must be 32 for FLOAT32LE" in node_text


def test_tts_cache_key_is_path_safe_and_write_failure_is_not_success() -> None:
    node_text = (Path(__file__).parents[2] / "fa_tts_py" / "tts_node.py").read_text(
        encoding="utf-8"
    )
    spec_text = (Path(__file__).parents[2] / "docs" / "仕様書.md").read_text(
        encoding="utf-8"
    )
    algorithm_text = (
        Path(__file__).parents[2] / "docs" / "アルゴリズム詳細説明書.md"
    ).read_text(encoding="utf-8")

    assert "def validate_cache_key(cache_key: str) -> None:" in node_text
    assert "cache_key must be 40 lowercase hex characters" in node_text
    assert "self.validate_cache_key(cache_key)" in node_text
    assert 'return self.cache_dir / f"{cache_key}.pcm"' in node_text
    assert "raise RuntimeError(f\"failed to write TTS cache file {path}: {exc}\") from exc" in node_text
    assert "response.message = str(exc)" in node_text
    assert "40 lowercase hex" in spec_text
    assert "path traversal" in algorithm_text


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

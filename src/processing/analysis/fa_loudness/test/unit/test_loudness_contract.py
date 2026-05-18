import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from fa_loudness_py.backends.frame_meter import (
    FrameMeterConfig,
    InternalFrameMeterBackend,
)


PACKAGE_ROOT = Path(__file__).parents[2]


def _config() -> FrameMeterConfig:
    return FrameMeterConfig(sample_rate=16000, db_floor=-120.0)


def test_default_config_uses_internal_backend_and_canonical_audio() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_loudness"]["ros__parameters"]

    assert params["backend.name"] == "internal_frame_meter"
    assert params["expected.sample_rate"] == 16000
    assert params["expected.channels"] == 1
    assert params["expected.encoding"] == "FLOAT32LE"
    assert params["expected.bit_depth"] == 32
    assert params["expected.layout"] == "interleaved"
    assert params["meter.db_floor"] == -120.0


def test_backend_measures_rms_peak_dbfs_and_crest_factor() -> None:
    backend = InternalFrameMeterBackend(_config())
    samples = np.asarray([0.0, 0.5, -0.5, 1.0], dtype=np.float32)

    result = backend.measure(samples)

    expected_rms = math.sqrt((0.0 + 0.25 + 0.25 + 1.0) / 4.0)
    assert result.sample_count == 4
    assert result.rms == pytest.approx(expected_rms)
    assert result.peak == pytest.approx(1.0)
    assert result.rms_dbfs == pytest.approx(20.0 * math.log10(expected_rms))
    assert result.peak_dbfs == pytest.approx(0.0)
    assert result.crest_factor == pytest.approx(1.0 / expected_rms)


def test_backend_zero_signal_uses_configured_db_floor() -> None:
    backend = InternalFrameMeterBackend(_config())
    result = backend.measure(np.zeros(8, dtype=np.float32))

    assert result.sample_count == 8
    assert result.rms == 0.0
    assert result.peak == 0.0
    assert result.rms_dbfs == -120.0
    assert result.peak_dbfs == -120.0
    assert result.crest_factor == 0.0


def test_backend_rejects_non_canonical_audio_without_hidden_conversion() -> None:
    backend = InternalFrameMeterBackend(_config())

    with pytest.raises(ValueError, match="float32"):
        backend.measure(np.zeros(8, dtype=np.float64))
    with pytest.raises(ValueError, match="one-dimensional"):
        backend.measure(np.zeros((1, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="required"):
        backend.measure(np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError, match="normalized"):
        backend.measure(np.full(8, 2.0, dtype=np.float32))


def test_backend_config_rejects_invalid_meter_contract() -> None:
    with pytest.raises(RuntimeError, match="meter.sample_rate must be > 0"):
        InternalFrameMeterBackend(FrameMeterConfig(sample_rate=0, db_floor=-120.0))
    with pytest.raises(RuntimeError, match="meter.db_floor must be finite and < 0.0"):
        InternalFrameMeterBackend(FrameMeterConfig(sample_rate=16000, db_floor=0.0))
    with pytest.raises(RuntimeError, match="meter.db_floor must be >= -300.0"):
        InternalFrameMeterBackend(FrameMeterConfig(sample_rate=16000, db_floor=-301.0))


def test_backend_is_ros_free_and_not_ai_runtime() -> None:
    backend_path = PACKAGE_ROOT / "fa_loudness_py" / "backends" / "frame_meter.py"
    node_path = PACKAGE_ROOT / "fa_loudness_py" / "loudness_node.py"
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    backend_text = backend_path.read_text(encoding="utf-8")
    node_text = node_path.read_text(encoding="utf-8")

    assert "import rclpy" not in backend_text
    assert "fa_interfaces" not in backend_text
    assert "LoudnessFrame" not in backend_text
    assert "InternalFrameMeterBackend" in node_text
    assert "VAD / KWS / ASR / Turn Detector" in spec
    assert "非 AI analysis node" in spec
    assert "normalize" in readme
    assert "resample" not in backend_text


def test_node_requires_explicit_ros_parameters_and_binds_stream_identity() -> None:
    node_path = PACKAGE_ROOT / "fa_loudness_py" / "loudness_node.py"
    node_text = node_path.read_text(encoding="utf-8")

    assert "ParameterUninitializedException" in node_text
    assert "Parameter.Type.STRING" in node_text
    assert 'declare_parameter("input_topic", "audio/features/input")' not in node_text
    assert 'declare_parameter("meter.db_floor", -120.0)' not in node_text
    assert "msg.stream_id != self.input_topic" in node_text
    assert "except ValueError as exc:" in node_text
    assert "except RuntimeError as exc:" in node_text
    assert "raise" in node_text


def test_package_layout_matches_required_analysis_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_frame_meter.md",
        "config/default.yaml",
        "launch/fa_loudness.launch.py",
        "fa_loudness_py/loudness_node.py",
        "fa_loudness_py/backends/frame_meter.py",
        "test/unit/test_loudness_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    missing = [relative for relative in required_paths if not (PACKAGE_ROOT / relative).exists()]

    assert missing == []

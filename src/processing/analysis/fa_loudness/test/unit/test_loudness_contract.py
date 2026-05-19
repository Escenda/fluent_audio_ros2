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
    assert params["expected.stream_id"]
    assert params["output.stream_id"]
    assert params["expected.stream_id"] != params["input_topic"]
    assert params["output.stream_id"] != params["output_topic"]
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

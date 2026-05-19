from pathlib import Path

import numpy as np
import pytest
import yaml

from fa_stft_py.backends.stft import InternalStftBackend, StftConfig

PACKAGE_ROOT = Path(__file__).parents[2]

def _config() -> StftConfig:
    return StftConfig(sample_rate=16000, n_fft=8, hop_length=4, window="rectangular")

def test_default_config_uses_internal_backend_and_canonical_audio() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_stft"]["ros__parameters"]

    assert params["backend.name"] == "internal_stft"
    assert params["expected.sample_rate"] == 16000
    assert params["expected.channels"] == 1
    assert params["expected.encoding"] == "FLOAT32LE"
    assert params["expected.bit_depth"] == 32
    assert params["expected.layout"] == "interleaved"
    assert params["expected.stream_id"]
    assert params["output.stream_id"]
    assert params["expected.stream_id"] != params["input_topic"]
    assert params["output.stream_id"] != params["output_topic"]
    assert params["feature.n_fft"] == 400
    assert params["feature.hop_length"] == 160
    assert params["feature.window"] == "hann"

def test_backend_computes_finite_complex_matrix() -> None:
    backend = InternalStftBackend(_config())
    samples = np.sin(np.linspace(0.0, 1.0, 16, dtype=np.float32))

    result = backend.compute(samples)

    assert result.frame_count == 3
    assert result.bin_count == 5
    assert result.real.shape == (3, 5)
    assert result.imag.shape == (3, 5)
    assert result.real.dtype == np.float32
    assert result.imag.dtype == np.float32
    assert np.all(np.isfinite(result.real))
    assert np.all(np.isfinite(result.imag))

def test_backend_zero_signal_matches_zero_complex_spectrum() -> None:
    backend = InternalStftBackend(_config())
    result = backend.compute(np.zeros(8, dtype=np.float32))

    assert result.frame_count == 1
    assert result.bin_count == 5
    assert np.allclose(result.real, 0.0, rtol=0.0, atol=0.0)
    assert np.allclose(result.imag, 0.0, rtol=0.0, atol=0.0)

def test_backend_rejects_non_canonical_audio_without_hidden_conversion() -> None:
    backend = InternalStftBackend(_config())

    with pytest.raises(ValueError, match="float32"):
        backend.compute(np.zeros(8, dtype=np.float64))
    with pytest.raises(ValueError, match="one-dimensional"):
        backend.compute(np.zeros((1, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="normalized"):
        backend.compute(np.full(8, 2.0, dtype=np.float32))
    with pytest.raises(ValueError, match="feature.n_fft"):
        backend.compute(np.zeros(7, dtype=np.float32))
    with pytest.raises(ValueError, match="align"):
        backend.compute(np.zeros(9, dtype=np.float32))

def test_backend_config_rejects_invalid_feature_contract() -> None:
    with pytest.raises(RuntimeError, match="feature.hop_length must be <= feature.n_fft"):
        InternalStftBackend(StftConfig(sample_rate=16000, n_fft=8, hop_length=9, window="hann"))
    with pytest.raises(RuntimeError, match="feature.window must be hann or rectangular"):
        InternalStftBackend(StftConfig(sample_rate=16000, n_fft=8, hop_length=4, window="blackman"))

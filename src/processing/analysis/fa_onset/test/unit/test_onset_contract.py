from pathlib import Path

import numpy as np
import pytest
import yaml

from fa_onset_py.backends.spectral_flux import InternalSpectralFluxBackend, OnsetConfig


PACKAGE_ROOT = Path(__file__).parents[2]


def _config() -> OnsetConfig:
    return OnsetConfig(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        method="spectral_flux",
        threshold=0.1,
        min_interval_sec=0.05,
    )


def test_default_config_uses_internal_backend_and_canonical_audio() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_onset"]["ros__parameters"]

    assert params["backend.name"] == "internal_spectral_flux"
    assert params["expected.sample_rate"] == 16000
    assert params["expected.channels"] == 1
    assert params["expected.encoding"] == "FLOAT32LE"
    assert params["expected.bit_depth"] == 32
    assert params["expected.layout"] == "interleaved"
    assert params["feature.n_fft"] == 400
    assert params["feature.hop_length"] == 160
    assert params["feature.method"] == "spectral_flux"
    assert params["detector.threshold"] == 0.1
    assert params["detector.min_interval_sec"] == 0.05


def test_backend_computes_finite_onset_sequences() -> None:
    backend = InternalSpectralFluxBackend(_config())
    samples = np.zeros(880, dtype=np.float32)
    samples[420:520] = np.hanning(100).astype(np.float32)

    result = backend.compute(samples)

    assert result.frame_count == 4
    assert result.frame_times_sec.shape == (4,)
    assert result.strengths.shape == (4,)
    assert result.detected.shape == (4,)
    assert result.strengths.dtype == np.float32
    assert result.detected.dtype == np.bool_
    assert np.all(np.isfinite(result.strengths))
    assert np.count_nonzero(result.detected) == 1


def test_backend_zero_signal_has_zero_strength_and_no_detection() -> None:
    backend = InternalSpectralFluxBackend(_config())
    result = backend.compute(np.zeros(400, dtype=np.float32))

    assert result.frame_count == 1
    assert np.allclose(result.strengths, 0.0, rtol=0.0, atol=0.0)
    assert result.detected.tolist() == [False]
    assert np.allclose(result.frame_times_sec, [0.0125], rtol=0.0, atol=1.0e-7)


def test_backend_rejects_non_canonical_audio_without_hidden_conversion() -> None:
    backend = InternalSpectralFluxBackend(_config())

    with pytest.raises(ValueError, match="float32"):
        backend.compute(np.zeros(400, dtype=np.float64))
    with pytest.raises(ValueError, match="one-dimensional"):
        backend.compute(np.zeros((1, 400), dtype=np.float32))
    with pytest.raises(ValueError, match="normalized"):
        backend.compute(np.full(400, 2.0, dtype=np.float32))
    with pytest.raises(ValueError, match="feature.n_fft"):
        backend.compute(np.zeros(399, dtype=np.float32))
    with pytest.raises(ValueError, match="align"):
        backend.compute(np.zeros(401, dtype=np.float32))


def test_backend_config_rejects_invalid_feature_contract() -> None:
    with pytest.raises(RuntimeError, match="feature.method must be spectral_flux"):
        InternalSpectralFluxBackend(
            OnsetConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="energy",
                threshold=0.1,
                min_interval_sec=0.05,
            )
        )
    with pytest.raises(RuntimeError, match="detector.threshold must be finite and > 0.0"):
        InternalSpectralFluxBackend(
            OnsetConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="spectral_flux",
                threshold=0.0,
                min_interval_sec=0.05,
            )
        )
    with pytest.raises(RuntimeError, match="detector.min_interval_sec"):
        InternalSpectralFluxBackend(
            OnsetConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="spectral_flux",
                threshold=0.1,
                min_interval_sec=-0.1,
            )
        )


def test_backend_is_ros_free_and_not_ai_runtime() -> None:
    backend_path = PACKAGE_ROOT / "fa_onset_py" / "backends" / "spectral_flux.py"
    node_path = PACKAGE_ROOT / "fa_onset_py" / "onset_node.py"
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    backend_text = backend_path.read_text(encoding="utf-8")
    node_text = node_path.read_text(encoding="utf-8")

    assert "import rclpy" not in backend_text
    assert "fa_interfaces" not in backend_text
    assert "OnsetFrame" not in backend_text
    assert "InternalSpectralFluxBackend" in node_text
    assert "VAD / KWS / ASR / Turn Detector" in spec
    assert "非 AI analysis node" in spec
    assert "model runtime は持たず" in readme
    assert "resample" not in backend_text


def test_node_requires_explicit_ros_parameters_and_binds_stream_identity() -> None:
    node_path = PACKAGE_ROOT / "fa_onset_py" / "onset_node.py"
    node_text = node_path.read_text(encoding="utf-8")

    assert "ParameterUninitializedException" in node_text
    assert "Parameter.Type.STRING" in node_text
    assert 'declare_parameter("input_topic", "audio/features/input")' not in node_text
    assert 'declare_parameter("feature.n_fft", 400)' not in node_text
    assert "msg.stream_id != self.input_topic" in node_text
    assert "except ValueError as exc:" in node_text
    assert "except RuntimeError as exc:" in node_text
    assert "raise" in node_text


def test_system_sample_aligns_onset_window_with_default_mic_chunk() -> None:
    sample_path = (
        PACKAGE_ROOT.parents[2]
        / "system"
        / "fluent_audio_system"
        / "config"
        / "fluent_audio_system.sample.yaml"
    )
    config = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
    analysis_group = next(group for group in config["groups"] if group["id"] == "analysis")
    onset_node = next(node for node in analysis_group["nodes"] if node["id"] == "fa_onset")

    assert onset_node["parameters"]["input_topic"] == "audio/resample16k/mic"
    assert onset_node["parameters"]["feature.n_fft"] == 320
    assert onset_node["parameters"]["feature.hop_length"] == 160
    assert onset_node["parameters"]["detector.threshold"] == 0.1

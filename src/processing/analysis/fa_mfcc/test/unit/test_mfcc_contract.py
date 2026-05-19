from pathlib import Path

import numpy as np
import pytest
import yaml

from fa_mfcc_py.backends.mfcc import InternalMfccBackend, MfccConfig


PACKAGE_ROOT = Path(__file__).parents[2]


def _config() -> MfccConfig:
    return MfccConfig(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=40,
        n_mfcc=13,
        f_min_hz=0.0,
        f_max_hz=8000.0,
        log_floor=1.0e-10,
        dct_type="dct2",
        normalization="ortho",
    )


def test_default_config_uses_internal_backend_and_canonical_audio() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_mfcc"]["ros__parameters"]

    assert params["backend.name"] == "internal_mfcc"
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
    assert params["feature.n_mels"] == 40
    assert params["feature.n_mfcc"] == 13
    assert params["feature.dct_type"] == "dct2"
    assert params["feature.normalization"] == "ortho"


def test_backend_computes_finite_frames_by_coefficients_matrix() -> None:
    backend = InternalMfccBackend(_config())
    samples = np.sin(np.linspace(0.0, 1.0, 1520, dtype=np.float32)).astype(np.float32)

    result = backend.compute(samples)

    assert result.frame_count == 8
    assert result.values.shape == (8, 13)
    assert result.values.dtype == np.float32
    assert np.all(np.isfinite(result.values))


def test_backend_zero_signal_matches_dct_ortho_golden_vector() -> None:
    backend = InternalMfccBackend(_config())
    result = backend.compute(np.zeros(400, dtype=np.float32))

    expected_c0 = np.sqrt(40.0) * np.log(1.0e-10)

    assert result.frame_count == 1
    assert result.values.shape == (1, 13)
    assert np.allclose(result.values[:, 0], expected_c0, rtol=0.0, atol=1.0e-5)
    assert np.allclose(result.values[:, 1:], 0.0, rtol=0.0, atol=1.0e-5)


def test_backend_rejects_non_canonical_audio_without_hidden_conversion() -> None:
    backend = InternalMfccBackend(_config())

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
    with pytest.raises(RuntimeError, match="feature.n_mfcc must be <= feature.n_mels"):
        InternalMfccBackend(
            MfccConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                n_mels=12,
                n_mfcc=13,
                f_min_hz=0.0,
                f_max_hz=8000.0,
                log_floor=1.0e-10,
                dct_type="dct2",
                normalization="ortho",
            )
        )
    with pytest.raises(RuntimeError, match="feature.f_max_hz must be <= Nyquist"):
        InternalMfccBackend(
            MfccConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                n_mels=40,
                n_mfcc=13,
                f_min_hz=0.0,
                f_max_hz=9000.0,
                log_floor=1.0e-10,
                dct_type="dct2",
                normalization="ortho",
            )
        )
    with pytest.raises(RuntimeError, match="feature.dct_type must be dct2"):
        InternalMfccBackend(
            MfccConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                n_mels=40,
                n_mfcc=13,
                f_min_hz=0.0,
                f_max_hz=8000.0,
                log_floor=1.0e-10,
                dct_type="dct3",
                normalization="ortho",
            )
        )
    with pytest.raises(RuntimeError, match="feature.normalization must be ortho"):
        InternalMfccBackend(
            MfccConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                n_mels=40,
                n_mfcc=13,
                f_min_hz=0.0,
                f_max_hz=8000.0,
                log_floor=1.0e-10,
                dct_type="dct2",
                normalization="none",
            )
        )


def test_backend_is_ros_free_and_not_ai_runtime() -> None:
    backend_path = PACKAGE_ROOT / "fa_mfcc_py" / "backends" / "mfcc.py"
    node_path = PACKAGE_ROOT / "fa_mfcc_py" / "mfcc_node.py"
    backend_text = backend_path.read_text(encoding="utf-8")
    node_text = node_path.read_text(encoding="utf-8")

    assert "import rclpy" not in backend_text
    assert "fa_interfaces" not in backend_text
    assert "MfccFrame" not in backend_text
    assert "InternalMfccBackend" in node_text
    assert "resample" not in backend_text


def test_node_requires_explicit_ros_parameters_and_binds_stream_identity() -> None:
    node_path = PACKAGE_ROOT / "fa_mfcc_py" / "mfcc_node.py"
    node_text = node_path.read_text(encoding="utf-8")

    assert "ParameterUninitializedException" in node_text
    assert "Parameter.Type.STRING" in node_text
    assert "must be a string parameter" in node_text
    assert "get_parameter(name).get_parameter_value().string_value" not in node_text
    assert "expected.stream_id" in node_text
    assert "output.stream_id" in node_text
    assert "out.stream_id = self.output_stream_id" in node_text
    assert "@staticmethod" in node_text
    assert "def _same_identity(left: str, right: str) -> bool:" in node_text
    assert 'declare_parameter("input_topic", "audio/features/input")' not in node_text
    assert 'declare_parameter("feature.n_fft", 400)' not in node_text
    assert "msg.stream_id != self.expected_stream_id" in node_text
    assert "except ValueError as exc:" in node_text
    assert "except RuntimeError as exc:" in node_text
    assert "raise" in node_text


def test_system_sample_aligns_mfcc_window_with_default_mic_chunk() -> None:
    sample_path = (
        PACKAGE_ROOT.parents[2]
        / "system"
        / "fluent_audio_system"
        / "config"
        / "fluent_audio_system.sample.yaml"
    )
    config = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
    analysis_group = next(group for group in config["groups"] if group["id"] == "analysis")
    mfcc_node = next(node for node in analysis_group["nodes"] if node["id"] == "fa_mfcc")

    assert mfcc_node["parameters"]["input_topic"] == "audio/resample16k/mic"
    assert mfcc_node["parameters"]["feature.n_fft"] == 320
    assert mfcc_node["parameters"]["feature.hop_length"] == 160
    assert mfcc_node["parameters"]["feature.n_mfcc"] == 13

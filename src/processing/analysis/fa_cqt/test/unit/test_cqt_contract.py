from pathlib import Path

import numpy as np
import pytest
import yaml

from fa_cqt_py.backends.constant_q import CqtConfig, InternalCqtBackend


PACKAGE_ROOT = Path(__file__).parents[2]


def _config() -> CqtConfig:
    return CqtConfig(
        sample_rate=16000,
        frame_length=4096,
        hop_length=512,
        bin_count=48,
        bins_per_octave=12,
        f_min_hz=130.8128,
        window="hann",
        normalization="l2",
    )


def test_default_config_uses_internal_backend_and_canonical_audio() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_cqt"]["ros__parameters"]

    assert params["backend.name"] == "internal_cqt"
    assert params["expected.sample_rate"] == 16000
    assert params["expected.channels"] == 1
    assert params["expected.encoding"] == "FLOAT32LE"
    assert params["expected.bit_depth"] == 32
    assert params["expected.layout"] == "interleaved"
    assert params["expected.stream_id"]
    assert params["output.stream_id"]
    assert params["expected.stream_id"] != params["input_topic"]
    assert params["output.stream_id"] != params["output_topic"]
    assert params["feature.frame_length"] == 4096
    assert params["feature.hop_length"] == 512
    assert params["feature.bin_count"] == 48
    assert params["feature.bins_per_octave"] == 12
    assert params["feature.f_min_hz"] == 130.8128
    assert params["feature.window"] == "hann"
    assert params["feature.normalization"] == "l2"


def test_backend_computes_finite_complex_frames_by_bins_matrix() -> None:
    backend = InternalCqtBackend(_config())
    time = np.arange(5120, dtype=np.float32) / np.float32(16000.0)
    samples = (0.4 * np.sin(np.float32(2.0 * np.pi * 130.8128) * time)).astype(np.float32)

    result = backend.compute(samples)
    magnitude = np.sqrt(result.real**2 + result.imag**2)

    assert result.frame_count == 3
    assert result.real.shape == (3, 48)
    assert result.imag.shape == (3, 48)
    assert result.real.dtype == np.float32
    assert result.imag.dtype == np.float32
    assert np.all(np.isfinite(result.real))
    assert np.all(np.isfinite(result.imag))
    assert np.all(np.argmax(magnitude, axis=1) <= 1)
    assert backend.center_frequencies_hz.shape == (48,)
    assert backend.window_lengths.shape == (48,)


def test_backend_zero_signal_is_zero_complex_vector() -> None:
    backend = InternalCqtBackend(_config())
    result = backend.compute(np.zeros(4096, dtype=np.float32))

    assert result.frame_count == 1
    assert result.real.shape == (1, 48)
    assert result.imag.shape == (1, 48)
    assert np.allclose(result.real, 0.0, rtol=0.0, atol=0.0)
    assert np.allclose(result.imag, 0.0, rtol=0.0, atol=0.0)


def test_backend_rejects_non_canonical_audio_without_hidden_conversion() -> None:
    backend = InternalCqtBackend(_config())

    with pytest.raises(ValueError, match="float32"):
        backend.compute(np.zeros(4096, dtype=np.float64))
    with pytest.raises(ValueError, match="one-dimensional"):
        backend.compute(np.zeros((1, 4096), dtype=np.float32))
    with pytest.raises(ValueError, match="normalized"):
        backend.compute(np.full(4096, 2.0, dtype=np.float32))
    with pytest.raises(ValueError, match="feature.frame_length"):
        backend.compute(np.zeros(4095, dtype=np.float32))
    with pytest.raises(ValueError, match="align"):
        backend.compute(np.zeros(4097, dtype=np.float32))


def test_backend_config_rejects_invalid_feature_contract() -> None:
    with pytest.raises(RuntimeError, match="feature.window must be hann"):
        InternalCqtBackend(
            CqtConfig(
                sample_rate=16000,
                frame_length=4096,
                hop_length=512,
                bin_count=48,
                bins_per_octave=12,
                f_min_hz=130.8128,
                window="rect",
                normalization="l2",
            )
        )
    with pytest.raises(RuntimeError, match="feature.normalization must be l2"):
        InternalCqtBackend(
            CqtConfig(
                sample_rate=16000,
                frame_length=4096,
                hop_length=512,
                bin_count=48,
                bins_per_octave=12,
                f_min_hz=130.8128,
                window="hann",
                normalization="l1",
            )
        )
    with pytest.raises(RuntimeError, match="feature.bin_count exceeds Nyquist"):
        InternalCqtBackend(
            CqtConfig(
                sample_rate=16000,
                frame_length=4096,
                hop_length=512,
                bin_count=96,
                bins_per_octave=12,
                f_min_hz=130.8128,
                window="hann",
                normalization="l2",
            )
        )
    with pytest.raises(RuntimeError, match="feature.f_min_hz requires feature.frame_length"):
        InternalCqtBackend(
            CqtConfig(
                sample_rate=16000,
                frame_length=1024,
                hop_length=512,
                bin_count=48,
                bins_per_octave=12,
                f_min_hz=130.8128,
                window="hann",
                normalization="l2",
            )
        )


def test_backend_is_ros_free_and_not_ai_runtime() -> None:
    backend_path = PACKAGE_ROOT / "fa_cqt_py" / "backends" / "constant_q.py"
    node_path = PACKAGE_ROOT / "fa_cqt_py" / "cqt_node.py"
    backend_text = backend_path.read_text(encoding="utf-8")
    node_text = node_path.read_text(encoding="utf-8")

    assert "import rclpy" not in backend_text
    assert "fa_interfaces" not in backend_text
    assert "CqtFrame" not in backend_text
    assert "InternalCqtBackend" in node_text
    assert "resample" not in backend_text


def test_node_requires_explicit_ros_parameters_and_binds_stream_identity() -> None:
    node_path = PACKAGE_ROOT / "fa_cqt_py" / "cqt_node.py"
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
    assert 'declare_parameter("feature.frame_length", 4096)' not in node_text
    assert "msg.stream_id != self.expected_stream_id" in node_text
    assert "except ValueError as exc:" in node_text
    assert "except RuntimeError as exc:" in node_text
    assert "raise" in node_text


def test_system_sample_documents_cqt_long_window_input() -> None:
    sample_path = (
        PACKAGE_ROOT.parents[2]
        / "system"
        / "fluent_audio_system"
        / "config"
        / "fluent_audio_system.sample.yaml"
    )
    config = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
    analysis_group = next(group for group in config["groups"] if group["id"] == "analysis")
    cqt_node = next(node for node in analysis_group["nodes"] if node["id"] == "fa_cqt")

    assert cqt_node["parameters"]["input_topic"] == "audio/frame_buffer/cqt"
    assert cqt_node["parameters"]["feature.frame_length"] == 4096
    assert cqt_node["parameters"]["feature.hop_length"] == 512

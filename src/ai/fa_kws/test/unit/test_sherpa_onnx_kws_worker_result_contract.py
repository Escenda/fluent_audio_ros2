from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(KeywordSpotter=object),
    )
    sys.modules.pop("fa_kws_py.backends.sherpa_onnx_kws_worker", None)
    return importlib.import_module("fa_kws_py.backends.sherpa_onnx_kws_worker")


def _detect_config(worker_module, tmp_path: Path):
    files = {}
    for name in ("encoder", "decoder", "joiner", "tokens", "keywords"):
        path = tmp_path / f"{name}.txt"
        path.write_text(name, encoding="utf-8")
        files[name] = path
    audio = tmp_path / "audio.f32"
    np.zeros(160, dtype="<f4").tofile(audio)
    return worker_module.DetectConfig(
        worker=worker_module.WorkerConfig(
            encoder=files["encoder"],
            decoder=files["decoder"],
            joiner=files["joiner"],
            tokens=files["tokens"],
            keywords=files["keywords"],
            provider="cpu",
            sample_rate=16000,
            num_threads=1,
            max_active_paths=4,
            num_trailing_blanks=1,
            keywords_score=1.0,
            keywords_threshold=0.05,
        ),
        audio_path=audio,
    )


class _FakeStream:
    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self.sample_rate = sample_rate
        self.samples = samples


class _FakeSpotter:
    def __init__(self, result: object) -> None:
        self.result = result

    def create_stream(self) -> _FakeStream:
        return _FakeStream()

    def is_ready(self, stream: _FakeStream) -> bool:
        return False

    def decode_stream(self, stream: _FakeStream) -> None:
        raise AssertionError("decode_stream should not be called")

    def get_result(self, stream: _FakeStream) -> object:
        return self.result


def test_detect_accepts_string_result_from_current_sherpa_onnx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker(monkeypatch)
    monkeypatch.setattr(worker, "create_spotter", lambda config: _FakeSpotter("  hey_aspa  "))

    result = worker.detect(_detect_config(worker, tmp_path))

    assert result == "DETECTED\they_aspa\t1.00000000\t0.00000000"


def test_detect_accepts_object_result_from_older_sherpa_onnx(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker(monkeypatch)
    result_object = SimpleNamespace(keyword="hey_aspa", start_time=0.25)
    monkeypatch.setattr(worker, "create_spotter", lambda config: _FakeSpotter(result_object))

    result = worker.detect(_detect_config(worker, tmp_path))

    assert result == "DETECTED\they_aspa\t1.00000000\t0.25000000"


def test_detect_returns_no_detection_for_empty_string_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker(monkeypatch)
    monkeypatch.setattr(worker, "create_spotter", lambda config: _FakeSpotter(" "))

    result = worker.detect(_detect_config(worker, tmp_path))

    assert result == "NO_DETECTION"


def test_detect_rejects_invalid_object_start_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker(monkeypatch)
    result_object = SimpleNamespace(keyword="hey_aspa", start_time=-0.1)
    monkeypatch.setattr(worker, "create_spotter", lambda config: _FakeSpotter(result_object))

    with pytest.raises(RuntimeError, match="start_time"):
        worker.detect(_detect_config(worker, tmp_path))

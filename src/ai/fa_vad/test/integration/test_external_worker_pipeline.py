from pathlib import Path
import sys

import numpy as np

from fa_vad_py.backends.base import Float32MonoWindow
from fa_vad_py.backends.silero import SileroVAD


def _backend(
    *,
    tmp_path: Path,
    model_dir: Path,
    threshold_start: float = 0.5,
    threshold_end: float = 0.4,
    hangover_ms: int = 20,
) -> SileroVAD:
    worker = Path(__file__).parents[1] / "fixtures" / "fake_vad_worker.py"
    return SileroVAD(
        sample_rate=16000,
        frame_ms=20,
        hangover_ms=hangover_ms,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        model_path=str(model_dir),
        execution_provider="cpu",
        command=sys.executable,
        args=(
            str(worker),
            "--audio",
            "{audio}",
            "--model",
            "{model}",
            "--provider",
            "{provider}",
            "--sample-rate",
            "{sample_rate}",
        ),
        timeout_sec=1.0,
        workspace_dir=str(tmp_path / "workspace"),
        cleanup_audio_files=True,
    )


def test_external_worker_pipeline_reports_speech_start(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "probability.txt").write_text("0.75", encoding="utf-8")
    backend = _backend(tmp_path=tmp_path, model_dir=model_dir)

    result = backend.update(
        Float32MonoWindow(
            sample_rate=16000,
            data=np.zeros(512, dtype="<f4").tobytes(),
        )
    )

    assert result is not None
    assert result.probability == 0.75
    assert result.is_speech is True
    assert result.start is True
    assert result.end is False
    assert list((tmp_path / "workspace").iterdir()) == []


def test_external_worker_pipeline_reports_speech_end_after_hangover(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "probability.txt").write_text("0.75", encoding="utf-8")
    backend = _backend(tmp_path=tmp_path, model_dir=model_dir, hangover_ms=20)

    start = backend.update(
        Float32MonoWindow(
            sample_rate=16000,
            data=np.zeros(512, dtype="<f4").tobytes(),
        )
    )
    assert start is not None
    assert start.start is True

    (model_dir / "probability.txt").write_text("0.10", encoding="utf-8")
    end = backend.update(
        Float32MonoWindow(
            sample_rate=16000,
            data=np.zeros(512, dtype="<f4").tobytes(),
        )
    )

    assert end is not None
    assert end.probability == 0.1
    assert end.is_speech is False
    assert end.start is False
    assert end.end is True

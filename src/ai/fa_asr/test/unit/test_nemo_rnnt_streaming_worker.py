import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader

import pytest


PACKAGE_ROOT = Path(__file__).parents[2]
WORKER = PACKAGE_ROOT / "scripts" / "nemo_rnnt_streaming_worker"


def _load_worker_module():
    loader = SourceFileLoader("nemo_rnnt_streaming_worker_test", str(WORKER))
    spec = spec_from_loader(loader.name, loader)
    assert spec is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


def _write_model(path: Path) -> Path:
    path.write_bytes(b"fake nemo archive")
    return path


def _fake_modules_dir(tmp_path: Path, *, model_kind: str = "cache_aware") -> Path:
    root = tmp_path / "fake_nemo"
    models_dir = root / "nemo" / "collections" / "asr" / "models"
    utils_dir = root / "nemo" / "collections" / "asr" / "parts" / "utils"
    if model_kind == "offline":
        class_name = "FakeOfflineModel"
    elif model_kind == "noisy_cache_aware":
        class_name = "FakeNoisyCacheAwareRNNTModel"
    elif model_kind == "runtime_cache_aware":
        class_name = "FakeRuntimeCacheAwareRNNTModel"
    elif model_kind == "cache_aware_no_model_reset":
        class_name = "FakeCacheAwareRNNTModelWithoutReset"
    elif model_kind == "cache_aware_empty_transcript":
        class_name = "FakeCacheAwareRNNTModelWithEmptyTranscript"
    elif model_kind == "cache_aware":
        class_name = "FakeCacheAwareRNNTModel"
    else:
        class_name = "FakeRNNTModel"
    cache_aware = "False" if model_kind in ("offline", "runtime_cache_aware") else "True"
    decoder_kind = "ctc" if model_kind == "offline" else "rnnt"
    restore_noise = (
        'print("fake nemo restore log")'
        if model_kind == "noisy_cache_aware"
        else ""
    )
    models_dir.mkdir(parents=True)
    utils_dir.mkdir(parents=True)
    for package_dir in (
        root / "nemo",
        root / "nemo" / "collections",
        root / "nemo" / "collections" / "asr",
        root / "nemo" / "collections" / "asr" / "parts",
        utils_dir,
        models_dir,
    ):
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
    root.joinpath("numpy.py").write_text(
        """
float32 = "float32"


def asarray(samples, dtype=None):
    del dtype
    return tuple(samples)
""",
        encoding="utf-8",
    )
    root.joinpath("torch.py").write_text(
        """
class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def inference_mode():
    return _InferenceMode()
""",
        encoding="utf-8",
    )
    utils_dir.joinpath("streaming_utils.py").write_text(
        """
class CacheAwareStreamingAudioBuffer:
    def __init__(self, model, online_normalization=False):
        self.model = model
        self.items = []
        self.cursor = 0
        self.streams_length = None

    def append_audio(self, audio, stream_id=-1):
        self.model.append_stream_ids.append(stream_id)
        if self.streams_length is None:
            if stream_id >= 0:
                raise RuntimeError("first append must create a new stream")
            self.items.append(audio)
            self.streams_length = [len(audio)]
            return None, None, -1
        if stream_id != 0:
            raise RuntimeError("subsequent append must target stream 0")
        self.items.append(audio)
        self.streams_length[0] += len(audio)
        return None, None, stream_id

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor >= len(self.items):
            raise StopIteration
        item = self.items[self.cursor]
        self.cursor += 1
        return item, len(item)

    def is_buffer_empty(self):
        return self.cursor >= len(self.items)

    def reset_buffer(self):
        self.items = []
        self.cursor = 0
        self.streams_length = None
""",
        encoding="utf-8",
    )
    reset_methods = (
        ""
        if model_kind == "cache_aware_no_model_reset"
        else """
    def reset_stream(self, reason):
        self.reset_reasons.append(reason)
"""
    )
    models_dir.joinpath("__init__.py").write_text(
        f"""
class FakeStreamingCfg:
    drop_extra_pre_encoded = 0


class FakeEncoder:
    cache_aware_streaming = True

    def __init__(self):
        self.streaming_cfg = FakeStreamingCfg()

    def get_initial_cache_state(self, batch_size):
        return ("cache_channel", "cache_time", "cache_len")


class {class_name}:
    def __init__(self):
        self.cfg = {{
            "preprocessor": {{"sample_rate": 16000}},
            "encoder": {{
                "cache_aware_streaming": {cache_aware},
                "chunk_size": 1600,
                "max_partial_interval_ms": 300,
            }},
            "decoder": {{"kind": "{decoder_kind}"}},
            "joint": {{"jointnet": "present"}},
            "languages": ["ja", "en"],
        }}
        self.samples = []
        self.append_stream_ids = []
        self.reset_reasons = []
        self.calls = 0
        self.encoder = FakeEncoder()

    def start_stream(self, session_id):
        self.session_id = session_id

    def accept_audio(self, samples):
        self.samples.extend(samples)
        return {{"text": "partial-" + str(len(self.samples))}}

    def poll_partial(self):
        return "partial-" + str(len(self.samples))

    def finish_stream(self):
        return "final-" + str(len(self.samples))
{reset_methods}

    def conformer_stream_step(self, **kwargs):
        if "{model_kind}" not in ("cache_aware", "runtime_cache_aware", "cache_aware_no_model_reset", "cache_aware_empty_transcript"):
            raise RuntimeError("unexpected conformer_stream_step call")
        self.calls += 1
        text = "" if "{model_kind}" == "cache_aware_empty_transcript" else "cache-" + str(self.calls)
        return (
            "pred-" + str(self.calls),
            [{{"text": text}}],
            kwargs["cache_last_channel"],
            kwargs["cache_last_time"],
            kwargs["cache_last_channel_len"],
            ["hyp-" + str(self.calls)],
        )


class ASRModel:
    @staticmethod
    def restore_from(path):
        {restore_noise}
        return {class_name}()
""",
        encoding="utf-8",
    )
    return root


def _run_worker(*, python_path: Path, stdin_text: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_path)
    return subprocess.run(
        (sys.executable, "-S", str(WORKER)),
        input=stdin_text,
        capture_output=True,
        text=True,
        env=env,
        timeout=10.0,
        check=False,
    )


def _json_line(payload: dict[str, str | int | bool]) -> str:
    return json.dumps(payload, separators=(",", ":")) + "\n"


def _health_message(model_path: Path) -> dict[str, str | int | bool]:
    return {
        "type": "health",
        "model_path": str(model_path),
        "language": "ja",
        "sample_rate_hz": 16000,
        "channels": 1,
        "audio_encoding": "FLOAT32LE",
        "streaming": True,
        "emit_partial": True,
        "chunk_size_samples": 1600,
        "max_partial_interval_ms": 300,
    }


def _start_message(model_path: Path) -> dict[str, str | int | bool]:
    message = _health_message(model_path)
    message["type"] = "start"
    message["session_id"] = "s1"
    message["user_turn_id"] = 7
    return message


def _audio_b64(samples: tuple[float, ...]) -> str:
    payload = b"".join(struct.pack("<f", sample) for sample in samples)
    return base64.b64encode(payload).decode("ascii")


def test_health_accepts_fake_cache_aware_rnnt_model(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path, model_kind="cache_aware")

    completed = _run_worker(
        python_path=fake_root,
        stdin_text=_json_line(_health_message(model_path)),
    )

    assert completed.returncode == 0
    response = json.loads(completed.stdout)
    assert response["type"] == "health_ok"
    assert response["model_class"] == "FakeCacheAwareRNNTModel"
    assert response["cache_aware_streaming"] is True
    assert response["sample_rate_hz"] == 16000
    assert completed.stderr == ""


def test_health_accepts_runtime_encoder_cache_aware_model_when_cfg_encoder_lacks_it(
    tmp_path: Path,
) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path, model_kind="runtime_cache_aware")

    completed = _run_worker(
        python_path=fake_root,
        stdin_text=_json_line(_health_message(model_path)),
    )

    assert completed.returncode == 0
    response = json.loads(completed.stdout)
    assert response["type"] == "health_ok"
    assert response["model_class"] == "FakeRuntimeCacheAwareRNNTModel"
    assert response["cache_aware_streaming"] is True
    assert completed.stderr == ""


def test_health_keeps_jsonl_stdout_clean_when_model_logs_to_stdout(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path, model_kind="noisy_cache_aware")

    completed = _run_worker(
        python_path=fake_root,
        stdin_text=_json_line(_health_message(model_path)),
    )

    assert completed.returncode == 0
    response = json.loads(completed.stdout)
    assert response["type"] == "health_ok"
    assert response["model_class"] == "FakeNoisyCacheAwareRNNTModel"
    assert completed.stdout.strip().startswith("{")
    assert "fake nemo restore log" in completed.stderr


def test_health_rejects_missing_model_without_importing_nemo(tmp_path: Path) -> None:
    fake_root = _fake_modules_dir(tmp_path)

    completed = _run_worker(
        python_path=fake_root,
        stdin_text=_json_line(_health_message(tmp_path / "missing.nemo")),
    )

    assert completed.returncode == 1
    assert "model_path file does not exist" in completed.stderr


def test_health_rejects_non_rnnt_or_offline_model(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path, model_kind="offline")

    completed = _run_worker(
        python_path=fake_root,
        stdin_text=_json_line(_health_message(model_path)),
    )

    assert completed.returncode == 1
    assert "RNNT/Transducer" in completed.stderr


def test_health_fails_closed_when_nemo_module_is_unavailable(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "model.nemo")

    completed = _run_worker(
        python_path=tmp_path,
        stdin_text=_json_line(_health_message(model_path)),
    )

    assert completed.returncode == 1
    assert "NeMo ASR module is unavailable" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_capability_rejects_sample_rate_and_chunk_mismatch(tmp_path: Path) -> None:
    worker = _load_worker_module()
    capability = worker.ModelCapability(
        model_class="EncDecRNNTBPEModel",
        sample_rate_hz=48000,
        languages=("ja",),
        rnnt=True,
        cache_aware_streaming=True,
        chunk_size_samples=1600,
        max_partial_interval_ms=300,
        supports_partials=True,
    )
    config = worker.WorkerConfig(
        model_path=_write_model(tmp_path / "model.nemo"),
        language="ja",
        sample_rate_hz=16000,
        channels=1,
        audio_encoding="FLOAT32LE",
        emit_partial=True,
        chunk_size_samples=1600,
        max_partial_interval_ms=300,
    )

    with pytest.raises(worker.WorkerError, match="sample_rate"):
        worker.validate_model_capability(capability, config)


def test_jsonl_protocol_accepts_health_start_audio_drain_and_finish(
    tmp_path: Path,
) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path)
    stdin_text = "".join(
        (
            _json_line(_health_message(model_path)),
            _json_line(_start_message(model_path)),
            _json_line(
                {
                    "type": "audio",
                    "session_id": "s1",
                    "encoding": "base64_float32le",
                    "sample_count": 2,
                    "data": _audio_b64((0.0, 0.25)),
                }
            ),
            _json_line({"type": "drain", "session_id": "s1"}),
            _json_line({"type": "finish", "session_id": "s1"}),
        )
    )

    completed = _run_worker(python_path=fake_root, stdin_text=stdin_text)

    assert completed.returncode == 0
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["type"] for response in responses] == [
        "health_ok",
        "stream_started",
        "partial",
        "audio_accepted",
        "partial",
        "drained",
        "final",
        "finished",
    ]
    assert responses[-2]["text"] == "cache-1"
    assert completed.stderr == ""


def test_jsonl_protocol_uses_cache_aware_conformer_stream_step_when_available(
    tmp_path: Path,
) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path, model_kind="cache_aware")
    stdin_text = "".join(
        (
            _json_line(_health_message(model_path)),
            _json_line(_start_message(model_path)),
            _json_line(
                {
                    "type": "audio",
                    "session_id": "s1",
                    "encoding": "base64_float32le",
                    "sample_count": 2,
                    "data": _audio_b64((0.0, 0.25)),
                }
            ),
            _json_line(
                {
                    "type": "audio",
                    "session_id": "s1",
                    "encoding": "base64_float32le",
                    "sample_count": 2,
                    "data": _audio_b64((0.5, 0.75)),
                }
            ),
            _json_line({"type": "finish", "session_id": "s1"}),
        )
    )

    completed = _run_worker(python_path=fake_root, stdin_text=stdin_text)

    assert completed.returncode == 0
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["type"] for response in responses] == [
        "health_ok",
        "stream_started",
        "partial",
        "audio_accepted",
        "partial",
        "audio_accepted",
        "final",
        "finished",
    ]
    assert responses[2]["text"] == "cache-1"
    assert responses[4]["text"] == "cache-2"
    assert responses[6]["text"] == "cache-2"
    assert completed.stderr == ""


def test_cache_aware_cancel_uses_runner_reset_without_model_reset_method(
    tmp_path: Path,
) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path, model_kind="cache_aware_no_model_reset")
    stdin_text = "".join(
        (
            _json_line(_health_message(model_path)),
            _json_line(_start_message(model_path)),
            _json_line(
                {
                    "type": "audio",
                    "session_id": "s1",
                    "encoding": "base64_float32le",
                    "sample_count": 2,
                    "data": _audio_b64((0.0, 0.25)),
                }
            ),
            _json_line({"type": "cancel", "session_id": "s1"}),
        )
    )

    completed = _run_worker(python_path=fake_root, stdin_text=stdin_text)

    assert completed.returncode == 0
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["type"] for response in responses] == [
        "health_ok",
        "stream_started",
        "partial",
        "audio_accepted",
        "cancelled",
    ]
    assert completed.stderr == ""


def test_cache_aware_finish_allows_empty_final_transcript(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path, model_kind="cache_aware_empty_transcript")
    stdin_text = "".join(
        (
            _json_line(_health_message(model_path)),
            _json_line(_start_message(model_path)),
            _json_line(
                {
                    "type": "audio",
                    "session_id": "s1",
                    "encoding": "base64_float32le",
                    "sample_count": 2,
                    "data": _audio_b64((0.0, 0.25)),
                }
            ),
            _json_line({"type": "finish", "session_id": "s1"}),
        )
    )

    completed = _run_worker(python_path=fake_root, stdin_text=stdin_text)

    assert completed.returncode == 0
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["type"] for response in responses] == [
        "health_ok",
        "stream_started",
        "audio_accepted",
        "final",
        "finished",
    ]
    assert responses[3]["text"] == ""
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("not json\n", "malformed JSON"),
        (_json_line({"type": "unknown"}), "unknown message type"),
    ],
)
def test_streaming_fails_closed_on_malformed_control_messages(
    tmp_path: Path,
    line: str,
    expected: str,
) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path)
    completed = _run_worker(
        python_path=fake_root,
        stdin_text=_json_line(_health_message(model_path)) + line,
    )

    assert completed.returncode == 1
    assert expected in completed.stderr


def test_start_contract_mismatch_fails_closed(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path)
    start = _start_message(model_path)
    start["sample_rate_hz"] = 48000

    completed = _run_worker(
        python_path=fake_root,
        stdin_text=_json_line(_health_message(model_path)) + _json_line(start),
    )

    assert completed.returncode == 1
    assert "sample_rate" in completed.stderr or "contract" in completed.stderr


@pytest.mark.parametrize(
    ("audio_update", "expected"),
    [
        ({"data": "%%%not-base64%%%"}, "not valid base64"),
        ({"sample_count": 3}, "sample_count mismatch"),
        ({"encoding": "S16LE"}, "audio encoding"),
        ({"data": _audio_b64((1.5,))}, "normalized float32 range"),
    ],
)
def test_streaming_fails_closed_on_invalid_audio_messages(
    tmp_path: Path,
    audio_update: dict[str, str | int],
    expected: str,
) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path)
    audio_message = {
        "type": "audio",
        "session_id": "s1",
        "encoding": "base64_float32le",
        "sample_count": 1,
        "data": _audio_b64((0.0,)),
    }
    audio_message.update(audio_update)
    stdin_text = "".join(
        (
            _json_line(_health_message(model_path)),
            _json_line(_start_message(model_path)),
            _json_line(audio_message),
        )
    )

    completed = _run_worker(python_path=fake_root, stdin_text=stdin_text)

    assert completed.returncode == 1
    assert expected in completed.stderr


def test_streaming_rejects_finish_without_audio(tmp_path: Path) -> None:
    model_path = _write_model(tmp_path / "model.nemo")
    fake_root = _fake_modules_dir(tmp_path)
    stdin_text = "".join(
        (
            _json_line(_health_message(model_path)),
            _json_line(_start_message(model_path)),
            _json_line({"type": "finish", "session_id": "s1"}),
        )
    )

    completed = _run_worker(python_path=fake_root, stdin_text=stdin_text)

    assert completed.returncode == 1
    assert "requires at least one accepted audio" in completed.stderr


def test_streaming_rejects_session_mismatch_before_forwarding_audio(tmp_path: Path) -> None:
    worker = _load_worker_module()
    config = worker.WorkerConfig(
        model_path=_write_model(tmp_path / "model.nemo"),
        language="ja",
        sample_rate_hz=16000,
        channels=1,
        audio_encoding="FLOAT32LE",
        emit_partial=True,
        chunk_size_samples=1600,
        max_partial_interval_ms=300,
    )

    class FakeModel:
        def __init__(self) -> None:
            self.samples: list[float] = []

        def accept_audio(self, samples: tuple[float, ...]) -> str:
            self.samples.extend(samples)
            return "partial"

        def finish_stream(self) -> str:
            return "final"

        def reset_stream(self, reason: str) -> None:
            self.reason = reason

    model = FakeModel()
    runtime = worker.StreamingRuntime(config, worker.NemoStreamingModelRunner(model))
    runtime.start(_start_message(config.model_path))

    with pytest.raises(worker.WorkerError, match="session_id"):
        runtime.accept_audio(
            {
                "type": "audio",
                "session_id": "other",
                "encoding": "base64_float32le",
                "sample_count": 1,
                "data": _audio_b64((0.5,)),
            }
        )

    assert model.samples == []

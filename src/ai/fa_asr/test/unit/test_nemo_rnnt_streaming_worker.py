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


def _worker_config(worker, model_path: Path):
    return worker.WorkerConfig(
        model_path=model_path,
        language="ja",
        sample_rate_hz=16000,
        channels=1,
        audio_encoding="FLOAT32LE",
        emit_partial=True,
        chunk_size_samples=1600,
        max_partial_interval_ms=300,
    )


def _fake_modules_dir(
    tmp_path: Path,
    *,
    model_kind: str = "cache_aware",
    streaming_cache_size: int = 10000,
) -> Path:
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


class FakeAudio(tuple):
    def size(self, dimension):
        if dimension != -1:
            raise RuntimeError("fake audio only exposes the final dimension")
        return len(self)


def asarray(samples, dtype=None):
    del dtype
    return FakeAudio(samples)
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
    eval_method = (
        ""
        if model_kind == "cache_aware_missing_eval"
        else """
    def eval(self):
        self.events.append("eval")
        self.eval_called = True
        return self
"""
    )
    encoder_set_max_audio_length = (
        ""
        if model_kind == "cache_aware_missing_set_max_audio_length"
        else """
    def set_max_audio_length(self, max_audio_length):
        self.model.events.append("set_max_audio_length:" + str(max_audio_length))
        self.max_audio_length_calls.append(max_audio_length)
"""
    )
    last_channel_cache_size = (
        '"invalid"'
        if model_kind == "cache_aware_invalid_cache_size"
        else str(streaming_cache_size)
    )
    models_dir.joinpath("__init__.py").write_text(
        f"""
class FakeStreamingCfg:
    drop_extra_pre_encoded = 0
    last_channel_cache_size = {last_channel_cache_size}


class FakeEncoder:
    cache_aware_streaming = True

    def __init__(self, model):
        self.model = model
        self.att_context_size = [-1, -1]
        self.streaming_cfg = FakeStreamingCfg()
        self.max_audio_length_calls = []

    def get_initial_cache_state(self, batch_size):
        return ("cache_channel", "cache_time", "cache_len")

    def set_default_att_context_size(self, context):
        self.model.events.append("set_default_att_context_size:" + str(context))
        self.att_context_size = context

    def setup_streaming_params(self, chunk_size, shift_size, left_chunks):
        if not self.model.eval_called:
            raise RuntimeError("model eval must be called before streaming setup")
        self.model.events.append(
            "setup_streaming_params:"
            + str(chunk_size)
            + ":"
            + str(shift_size)
            + ":"
            + str(left_chunks)
        )
{encoder_set_max_audio_length}


class {class_name}:
    def __init__(self):
        self.cfg = {{
            "preprocessor": {{"sample_rate": 16000, "window_stride": 0.01}},
            "encoder": {{
                "cache_aware_streaming": {cache_aware},
                "chunk_size": 1600,
                "max_partial_interval_ms": 300,
                "subsampling_factor": 8,
                "att_context_size": [-1, -1],
            }},
            "decoder": {{"kind": "{decoder_kind}"}},
            "joint": {{"jointnet": "present"}},
            "languages": ["ja", "en"],
        }}
        self.samples = []
        self.append_stream_ids = []
        self.reset_reasons = []
        self.calls = 0
        self.events = []
        self.eval_called = False
        self.encoder = FakeEncoder(self)
{eval_method}

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
        required_length = self.encoder.streaming_cfg.last_channel_cache_size + kwargs["processed_signal"].size(-1)
        expected_event = "set_max_audio_length:" + str(required_length)
        if len(self.events) == 0 or self.events[-1] != expected_event:
            raise RuntimeError("set_max_audio_length was not called before conformer_stream_step")
        self.events.append("conformer_stream_step")
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


class _DirectFakeAudio(tuple):
    def size(self, dimension: int) -> int:
        if dimension != -1:
            raise RuntimeError("fake audio only exposes the final dimension")
        return len(self)


class _DirectFakeNumpy:
    float32 = "float32"

    @staticmethod
    def asarray(samples: tuple[float, ...], dtype=None) -> _DirectFakeAudio:
        del dtype
        return _DirectFakeAudio(samples)


class _DirectFakeInferenceMode:
    def __enter__(self) -> "_DirectFakeInferenceMode":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        return False


class _DirectFakeTorch:
    @staticmethod
    def inference_mode() -> _DirectFakeInferenceMode:
        return _DirectFakeInferenceMode()


class _DirectFakeStreamingAudioBuffer:
    def __init__(self, model, online_normalization: bool = False) -> None:
        del online_normalization
        self._items: list[_DirectFakeAudio] = []
        self._cursor = 0
        self.streams_length: list[int] | None = None

    def append_audio(self, audio: _DirectFakeAudio, stream_id: int = -1):
        if self.streams_length is None:
            if stream_id >= 0:
                raise RuntimeError("first append must create a new stream")
            self._items.append(audio)
            self.streams_length = [audio.size(-1)]
            return None, None, -1
        if stream_id != 0:
            raise RuntimeError("subsequent append must target stream 0")
        self._items.append(audio)
        self.streams_length[0] += audio.size(-1)
        return None, None, stream_id

    def __iter__(self) -> "_DirectFakeStreamingAudioBuffer":
        return self

    def __next__(self) -> tuple[_DirectFakeAudio, int]:
        if self._cursor >= len(self._items):
            raise StopIteration
        item = self._items[self._cursor]
        self._cursor += 1
        return item, item.size(-1)

    def is_buffer_empty(self) -> bool:
        return self._cursor >= len(self._items)

    def reset_buffer(self) -> None:
        self._items = []
        self._cursor = 0
        self.streams_length = None


class _DirectFakeStreamingUtils:
    CacheAwareStreamingAudioBuffer = _DirectFakeStreamingAudioBuffer


class _DirectFakeStreamingCfg:
    drop_extra_pre_encoded = 0

    def __init__(self, last_channel_cache_size) -> None:
        self.last_channel_cache_size = last_channel_cache_size


class _DirectFakeEncoder:
    def __init__(self, last_channel_cache_size) -> None:
        self.att_context_size = [-1, -1]
        self.streaming_cfg = _DirectFakeStreamingCfg(last_channel_cache_size)
        self.max_audio_length_calls: list[int] = []
        self.model = None

    def get_initial_cache_state(self, batch_size: int) -> tuple[str, str, str]:
        del batch_size
        return ("cache_channel", "cache_time", "cache_len")

    def set_default_att_context_size(self, context: list[int]) -> None:
        self.att_context_size = context
        if self.model is not None:
            self.model.events.append(f"set_default_att_context_size:{context}")

    def setup_streaming_params(
        self,
        *,
        chunk_size: int,
        shift_size: int,
        left_chunks: int,
    ) -> None:
        if self.model is not None:
            self.model.events.append(
                f"setup_streaming_params:{chunk_size}:{shift_size}:{left_chunks}"
            )

    def set_max_audio_length(self, max_audio_length: int) -> None:
        self.max_audio_length_calls.append(max_audio_length)
        if self.model is not None:
            self.model.events.append(f"set_max_audio_length:{max_audio_length}")


class _DirectFakeEncoderWithoutSetMaxAudioLength:
    def __init__(self, last_channel_cache_size: int) -> None:
        self.att_context_size = [70, 1]
        self.streaming_cfg = _DirectFakeStreamingCfg(last_channel_cache_size)
        self.model = None

    def get_initial_cache_state(self, batch_size: int) -> tuple[str, str, str]:
        del batch_size
        return ("cache_channel", "cache_time", "cache_len")

    def setup_streaming_params(
        self,
        *,
        chunk_size: int,
        shift_size: int,
        left_chunks: int,
    ) -> None:
        if self.model is not None:
            self.model.events.append(
                f"setup_streaming_params:{chunk_size}:{shift_size}:{left_chunks}"
            )


class _DirectFakeModel:
    def __init__(self, encoder) -> None:
        self.cfg = {
            "preprocessor": {"sample_rate": 16000, "window_stride": 0.01},
            "encoder": {
                "subsampling_factor": 8,
                "att_context_size": encoder.att_context_size,
            },
        }
        self.encoder = encoder
        self.events: list[str] = []
        self.calls = 0
        if hasattr(encoder, "model"):
            encoder.model = self

    def conformer_stream_step(self, **kwargs):
        required_length = (
            self.encoder.streaming_cfg.last_channel_cache_size
            + kwargs["processed_signal"].size(-1)
        )
        expected_event = f"set_max_audio_length:{required_length}"
        if not self.events or self.events[-1] != expected_event:
            raise RuntimeError("set_max_audio_length was not called before model step")
        self.events.append("conformer_stream_step")
        self.calls += 1
        return (
            f"pred-{self.calls}",
            [{"text": f"cache-{self.calls}"}],
            kwargs["cache_last_channel"],
            kwargs["cache_last_time"],
            kwargs["cache_last_channel_len"],
            [f"hyp-{self.calls}"],
        )


def _install_direct_cache_aware_modules(worker, monkeypatch) -> None:
    def import_fake_module(module_name: str):
        if module_name == "numpy":
            return _DirectFakeNumpy
        if module_name == "torch":
            return _DirectFakeTorch
        if module_name == "nemo.collections.asr.parts.utils.streaming_utils":
            return _DirectFakeStreamingUtils
        raise AssertionError(f"unexpected fake module import: {module_name}")

    monkeypatch.setattr(worker, "_require_module", lambda module_name: None)
    monkeypatch.setattr(worker.importlib, "import_module", import_fake_module)


class _EvalFakeModel:
    def __init__(self) -> None:
        self.eval_calls = 0

    def eval(self) -> "_EvalFakeModel":
        self.eval_calls += 1
        return self


class _EvalFakeASRModel:
    restored_model = _EvalFakeModel()

    @staticmethod
    def restore_from(path: str) -> _EvalFakeModel:
        del path
        return _EvalFakeASRModel.restored_model


class _EvalFakeModelsModule:
    ASRModel = _EvalFakeASRModel


class _MissingEvalFakeModel:
    pass


class _MissingEvalFakeASRModel:
    @staticmethod
    def restore_from(path: str) -> _MissingEvalFakeModel:
        del path
        return _MissingEvalFakeModel()


class _MissingEvalFakeModelsModule:
    ASRModel = _MissingEvalFakeASRModel


def _install_direct_nemo_models_module(worker, monkeypatch, models_module) -> None:
    def find_fake_spec(module_name: str):
        if module_name in (
            "nemo",
            "nemo.collections",
            "nemo.collections.asr",
            "nemo.collections.asr.models",
        ):
            return True
        return None

    def import_fake_module(module_name: str):
        if module_name == "nemo.collections.asr.models":
            return models_module
        raise AssertionError(f"unexpected fake module import: {module_name}")

    monkeypatch.setattr(worker.importlib.util, "find_spec", find_fake_spec)
    monkeypatch.setattr(worker.importlib, "import_module", import_fake_module)


def test_load_nemo_model_calls_eval_after_restore(monkeypatch) -> None:
    worker = _load_worker_module()
    model = _EvalFakeModel()
    _EvalFakeASRModel.restored_model = model
    _install_direct_nemo_models_module(worker, monkeypatch, _EvalFakeModelsModule)

    restored = worker.load_nemo_model(Path("model.nemo"))

    assert restored is model
    assert model.eval_calls == 1


def test_load_nemo_model_fails_closed_when_eval_is_missing(monkeypatch) -> None:
    worker = _load_worker_module()
    _install_direct_nemo_models_module(
        worker,
        monkeypatch,
        _MissingEvalFakeModelsModule,
    )

    with pytest.raises(worker.WorkerError, match="eval"):
        worker.load_nemo_model(Path("model.nemo"))


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


def test_cache_aware_step_sets_required_max_audio_length_before_model_step(
    monkeypatch,
) -> None:
    worker = _load_worker_module()
    _install_direct_cache_aware_modules(worker, monkeypatch)
    encoder = _DirectFakeEncoder(last_channel_cache_size=10000)
    model = _DirectFakeModel(encoder)
    runner = worker.CacheAwareConformerStreamingRunner(
        model,
        _worker_config(worker, Path("model.nemo")),
    )

    runner.start_stream()
    text = runner.accept_audio(samples=(0.0, 0.25, 0.5))

    assert text == "cache-1"
    assert encoder.max_audio_length_calls == [10003]
    assert model.events == [
        "set_default_att_context_size:[70, 1]",
        "setup_streaming_params:10:5:2",
        "set_max_audio_length:10003",
        "conformer_stream_step",
    ]


def test_cache_aware_step_fails_closed_without_set_max_audio_length(
    monkeypatch,
) -> None:
    worker = _load_worker_module()
    _install_direct_cache_aware_modules(worker, monkeypatch)
    model = _DirectFakeModel(
        _DirectFakeEncoderWithoutSetMaxAudioLength(last_channel_cache_size=10000)
    )
    runner = worker.CacheAwareConformerStreamingRunner(
        model,
        _worker_config(worker, Path("model.nemo")),
    )

    runner.start_stream()

    with pytest.raises(worker.WorkerError, match="set_max_audio_length"):
        runner.accept_audio(samples=(0.0, 0.25))

    assert model.events == ["setup_streaming_params:10:5:2"]


@pytest.mark.parametrize("last_channel_cache_size", (0, -1, "invalid"))
def test_cache_aware_step_fails_closed_on_invalid_last_channel_cache_size(
    monkeypatch,
    last_channel_cache_size,
) -> None:
    worker = _load_worker_module()
    _install_direct_cache_aware_modules(worker, monkeypatch)
    encoder = _DirectFakeEncoder(last_channel_cache_size=last_channel_cache_size)
    model = _DirectFakeModel(encoder)

    with pytest.raises(worker.WorkerError, match="last_channel_cache_size"):
        worker.CacheAwareConformerStreamingRunner(
            model,
            _worker_config(worker, Path("model.nemo")),
        )

    assert encoder.max_audio_length_calls == []
    assert model.events == [
        "set_default_att_context_size:[70, 1]",
        "setup_streaming_params:10:5:2",
    ]


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
    runtime = worker.StreamingRuntime(
        config,
        worker.NemoStreamingModelRunner(model, config),
    )
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

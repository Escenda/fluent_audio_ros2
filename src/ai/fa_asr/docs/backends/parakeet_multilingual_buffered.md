# parakeet_multilingual_buffered Backend

## Position

`parakeet_multilingual_buffered` is the standard FluentAudio ASR backend for multilingual Parakeet 1.1B.

It is not a NIM, Riva, gRPC, Whisper, OpenAI, or stdin/stdout JSONL worker backend.
The `fa_asr` backend object owns the local NeMo / Parakeet runner in-process.
The streaming session owns its rolling audio buffer and emits partial hypotheses from repeated full-context decodes.

## Required Config

- `backend.name`: `parakeet_multilingual_buffered`
- `backend.model_path`: readable local multilingual Parakeet 1.1B `.nemo`, or `backend.model` identifying a multilingual Parakeet 1.1B model
- `backend.language`: empty string
- `backend.language_policy`: `auto_detect`
- `backend.sample_rate_hz`: `16000`
- `backend.channels`: `1`
- `backend.chunk_size_samples` or `backend.chunk_ms`: re-decode cadence
- `backend.emit_partial`: whether to publish uncommitted partial hypotheses
- `backend.max_buffer_sec`: retained rolling-buffer duration
- `backend.speech_energy_threshold`: RMS threshold used to reject empty final transcript for speech audio

Set only one of `backend.model` and `backend.model_path`.
The backend rejects model identifiers and paths that do not identify multilingual Parakeet 1.1B.

## Audio Contract

The backend accepts only:

- `FLOAT32LE`
- 16 kHz
- mono
- finite normalized samples
- non-empty payloads

It does not resample, downmix, convert PCM16, clamp, normalize, denoise, or fill missing audio.
Those operations belong to upstream FluentAudio processing nodes such as `fa_sample_format`, `fa_resample`, and correction nodes.

## Streaming Policy

The multilingual Parakeet `.nemo` currently used by FluentAudio is treated as a full-context model.
It is not treated as a cache-aware streaming model.

The backend policy is therefore explicit:

1. Keep a rolling buffer of accepted ASR-ready `FLOAT32LE` samples.
2. Re-decode the retained buffer at chunk boundaries.
3. Emit changed partial hypotheses as uncommitted `STATUS_PARTIAL` results.
4. Do not commit partial text as final text.
5. On VAD / turn-detector / timeout close, run a final decode through `finish()`.
6. Commit only the final result returned by `finish()`.

This is not fallback from a failed cache-aware streaming backend.
It is the declared policy for full-context multilingual Parakeet use in FluentAudio.

## Fail-Closed Behavior

The backend fails closed when:

- `backend.model` and `backend.model_path` are both empty
- both `backend.model` and `backend.model_path` are set
- model id / path does not identify multilingual Parakeet 1.1B
- `backend.language_policy` is not `auto_detect`
- `backend.language` is non-empty
- sample rate is not 16 kHz
- channels is not 1
- chunk / buffer / energy threshold config is invalid
- pushed payload does not match `FLOAT32LE` / 16 kHz / mono
- samples are empty, non-finite, or outside the backend payload contract
- final transcript is empty while the buffered speech energy is at or above `backend.speech_energy_threshold`

It does not fallback to unsupported models, Whisper, OpenAI, NIM, Riva, gRPC, or legacy JSONL workers.

## Verification Evidence

Current representative evidence:

- Unit/config tests for config validation, internal runner ownership, rolling-buffer re-decode, duplicate partial suppression, retained-window behavior, cancel/finish fail-closed behavior, and empty speech final rejection.
- Targeted pytest:
  `src/ai/fa_asr/test/unit/test_parakeet_multilingual_buffered_backend.py`
- Direct real-model smoke in `fluent-audio-runtime`: local Parakeet 1.1B `.nemo` restored and produced final Japanese text from a 16 kHz mono `FLOAT32LE` fixture.
- File-source ROS graph smoke in `fluent-audio-runtime`: `file_ja_voice_frontend` uses `parakeet_multilingual_buffered` without an ASR worker command and produced the expected final ASR result.

Remaining non-goals for this backend evidence:

- Generic live microphone accuracy.
- Long-form accuracy benchmarking.
- Riva / NIM / gRPC serving readiness.
- Whisper or OpenAI fallback behavior.

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from fa_asr_py.backends.parakeet_rnnt_stream_processor import (
    ContextSize,
    DecodeStepResult,
    EncodedAudio,
    ParakeetStreamConfig,
)


class NemoStreamingAudioBuffer:
    def __init__(self, buffer: Any) -> None:
        self._buffer = buffer

    @property
    def context_size(self) -> Any:
        return self._buffer.context_size

    @property
    def context_size_batch(self) -> Any:
        return self._buffer.context_size_batch

    @property
    def samples(self) -> Any:
        return self._buffer.samples

    def add_audio_batch(
        self,
        samples: np.ndarray,
        *,
        audio_lengths: Any,
        is_last_chunk: bool,
        is_last_chunk_batch: Any,
    ) -> None:
        import torch

        audio = torch.from_numpy(samples).to(device=self._buffer.samples.device).unsqueeze(0)
        self._buffer.add_audio_batch_(
            audio,
            audio_lengths=audio_lengths,
            is_last_chunk=is_last_chunk,
            is_last_chunk_batch=is_last_chunk_batch,
        )


class NemoParakeetRnntRuntime:
    def __init__(self, config: ParakeetStreamConfig) -> None:
        self.config = config
        self.model = self._load_model(config)
        self.device = self.model.device
        self.sample_rate = int(self.model._cfg.preprocessor["sample_rate"])
        self.feature_stride_sec = float(self.model._cfg.preprocessor["window_stride"])
        self.encoder_subsampling_factor = int(self.model.encoder.subsampling_factor)
        self.uses_chunked_limited_attention_with_right_context = (
            getattr(self.model.cfg.encoder, "att_context_style", "")
            == "chunked_limited_with_rc"
        )

    def configure_for_streaming(self) -> None:
        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0
        self.model.eval()

    def set_default_attention_context(self, context_encoder_frames: ContextSize) -> None:
        self.model.encoder.set_default_att_context_size(
            att_context_size=[
                context_encoder_frames.left,
                context_encoder_frames.chunk,
                context_encoder_frames.right,
            ]
        )

    def create_buffer(self, context_samples: ContextSize) -> NemoStreamingAudioBuffer:
        from nemo.collections.asr.parts.utils.streaming_utils import (
            ContextSize as NemoContextSize,
            StreamingBatchedAudioBuffer,
        )

        buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=NemoContextSize(
                left=context_samples.left,
                chunk=context_samples.chunk,
                right=context_samples.right,
            ),
            dtype=self._torch_dtype("float32"),
            device=self.device,
        )
        return NemoStreamingAudioBuffer(buffer)

    def make_audio_lengths(self, sample_count: int) -> Any:
        import torch

        return torch.tensor([sample_count], dtype=torch.long, device=self.device)

    def make_last_chunk_batch(self, is_last_chunk: bool) -> Any:
        import torch

        return torch.tensor([is_last_chunk], dtype=torch.bool, device=self.device)

    def encode(self, buffer: NemoStreamingAudioBuffer) -> EncodedAudio:
        encoder_output, encoder_output_len = self.model(
            input_signal=buffer.samples,
            input_signal_length=buffer.context_size_batch.total(),
        )
        return EncodedAudio(
            output=encoder_output.transpose(1, 2),
            output_lengths=encoder_output_len,
        )

    def subsample_context(self, context: Any, *, factor: int) -> Any:
        return context.subsample(factor=factor)

    def trim_left_context(self, encoded_output: Any, *, left_frames: int) -> Any:
        return encoded_output[:, left_frames:]

    def output_length(self, encoded: EncodedAudio, context_batch: Any, *, is_last_chunk: bool) -> Any:
        import torch

        last_chunk_batch = self.make_last_chunk_batch(is_last_chunk)
        return torch.where(
            last_chunk_batch,
            encoded.output_lengths - context_batch.left,
            context_batch.chunk,
        )

    def decode(
        self,
        encoded_output: Any,
        output_lengths: Any,
        *,
        previous_state: Any,
    ) -> DecodeStepResult:
        decoding_computer = self.model.decoding.decoding.decoding_computer
        chunk_hyps, _, decoder_state = decoding_computer(
            x=encoded_output,
            out_len=output_lengths,
            prev_batched_state=previous_state,
        )
        return DecodeStepResult(
            chunk_hypotheses=chunk_hyps,
            decoder_state=decoder_state,
        )

    def merge_hypotheses(self, current_hypotheses: Any, chunk_hypotheses: Any) -> Any:
        if current_hypotheses is None:
            return chunk_hypotheses
        current_hypotheses.merge_(chunk_hypotheses)
        return current_hypotheses

    def hypotheses_to_text(self, hypotheses: Any) -> str:
        from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses

        hyp = batched_hyps_to_hypotheses(hypotheses, None, batch_size=1)[0]
        return self.model.tokenizer.ids_to_text(hyp.y_sequence.tolist())

    def _load_model(self, config: ParakeetStreamConfig) -> Any:
        import nemo.collections.asr as nemo_asr
        import torch
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
        from omegaconf import OmegaConf, open_dict

        if config.model_path:
            model = nemo_asr.models.ASRModel.restore_from(config.model_path)
        elif config.model_name:
            model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=config.model_name
            )
        else:
            raise ValueError("model_name or model_path is required")

        model_cfg = copy.deepcopy(model._cfg)
        OmegaConf.set_struct(model_cfg.preprocessor, False)
        model_cfg.preprocessor.dither = 0.0
        model_cfg.preprocessor.pad_to = 0
        OmegaConf.set_struct(model_cfg.preprocessor, True)

        decoding = OmegaConf.structured(RNNTDecodingConfig)
        with open_dict(decoding):
            decoding.strategy = "greedy_batch"
            decoding.greedy.loop_labels = True
            decoding.greedy.preserve_alignments = False
            decoding.fused_batch_size = -1
            decoding.beam.return_best_hypothesis = True

        if hasattr(model, "change_decoding_strategy"):
            model.change_decoding_strategy(decoding)

        model.freeze()
        model = model.to(torch.device(config.device))
        model = model.to(self._torch_dtype(config.compute_dtype))
        return model

    @staticmethod
    def _torch_dtype(name: str) -> Any:
        import torch

        match name:
            case "float32":
                return torch.float32
            case "float16":
                return torch.float16
            case "bfloat16":
                return torch.bfloat16
            case _:
                raise ValueError(f"unsupported compute_dtype: {name}")

from pathlib import Path


def test_audio_frame_is_waveform_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "AudioFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "string encoding",
        "uint32 sample_rate",
        "uint32 channels",
        "uint32 bit_depth",
        "string layout",
        "uint8[] data",
        "uint32 epoch",
    ]
    assert all("rms" not in field for field in fields)
    assert all("peak" not in field for field in fields)
    assert all("vad" not in field for field in fields)


def test_encoded_audio_chunk_is_codec_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "EncodedAudioChunk.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "string codec",
        "string container",
        "string payload_format",
        "uint32 sample_rate",
        "uint32 channels",
        "uint64 sequence",
        "uint64 media_time_ns",
        "uint64 duration_ns",
        "uint32 epoch",
        "uint8[] data",
    ]
    assert all("encoding" not in field for field in fields)
    assert all("bit_depth" not in field for field in fields)
    assert all("layout" not in field for field in fields)
    assert all("rms" not in field for field in fields)
    assert all("vad" not in field for field in fields)


def test_encoded_audio_chunk_is_registered_for_rosidl_generation() -> None:
    cmake_path = Path(__file__).parents[2] / "CMakeLists.txt"
    cmake_text = cmake_path.read_text(encoding="utf-8")

    assert '"msg/EncodedAudioChunk.msg"' in cmake_text


def test_audio_embedding_frame_is_embedding_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "AudioEmbeddingFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "string model_id",
        "uint32 sample_rate",
        "uint32 input_sample_count",
        "uint32 dimension",
        "string payload_encoding",
        "float32[] embedding",
    ]
    assert all("text" not in field for field in fields)
    assert all("probability" not in field for field in fields)
    assert all("detected" not in field for field in fields)
    assert all("zero" not in field for field in fields)


def test_audio_embedding_frame_is_registered_for_rosidl_generation() -> None:
    cmake_path = Path(__file__).parents[2] / "CMakeLists.txt"
    cmake_text = cmake_path.read_text(encoding="utf-8")

    assert '"msg/AudioEmbeddingFrame.msg"' in cmake_text


def test_log_mel_frame_is_feature_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "LogMelFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "uint32 sample_rate",
        "uint32 input_sample_count",
        "uint32 n_fft",
        "uint32 hop_length",
        "uint32 n_mels",
        "uint32 frame_count",
        "float32 f_min_hz",
        "float32 f_max_hz",
        "float32 log_floor",
        "string layout",
        "float32[] values",
    ]
    assert all("probability" not in field for field in fields)
    assert all("detected" not in field for field in fields)
    assert all("text" not in field for field in fields)


def test_cqt_frame_is_feature_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "CqtFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "uint32 sample_rate",
        "uint32 input_sample_count",
        "uint32 hop_length",
        "uint32 frame_count",
        "uint32 bin_count",
        "uint32 bins_per_octave",
        "float32 f_min_hz",
        "float32 f_max_hz",
        "string window",
        "string normalization",
        "string layout",
        "string value_format",
        "float32[] center_frequencies_hz",
        "uint32[] window_lengths",
        "float32[] real",
        "float32[] imag",
    ]
    assert all("probability" not in field for field in fields)
    assert all("detected" not in field for field in fields)
    assert all("text" not in field for field in fields)


def test_mfcc_frame_is_feature_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "MfccFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "uint32 sample_rate",
        "uint32 input_sample_count",
        "uint32 n_fft",
        "uint32 hop_length",
        "uint32 n_mels",
        "uint32 n_mfcc",
        "uint32 frame_count",
        "float32 f_min_hz",
        "float32 f_max_hz",
        "float32 log_floor",
        "string dct_type",
        "string normalization",
        "string layout",
        "float32[] values",
    ]
    assert all("probability" not in field for field in fields)
    assert all("detected" not in field for field in fields)
    assert all("text" not in field for field in fields)


def test_loudness_frame_is_measurement_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "LoudnessFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "uint32 sample_rate",
        "uint32 input_sample_count",
        "float32 rms",
        "float32 peak",
        "float32 rms_dbfs",
        "float32 peak_dbfs",
        "float32 crest_factor",
        "float32 db_floor",
    ]
    assert all("vad" not in field for field in fields)
    assert all("detected" not in field for field in fields)
    assert all("text" not in field for field in fields)


def test_onset_frame_is_measurement_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "OnsetFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "uint32 sample_rate",
        "uint32 input_sample_count",
        "uint32 n_fft",
        "uint32 hop_length",
        "uint32 frame_count",
        "float32 threshold",
        "float32 min_interval_sec",
        "string method",
        "string layout",
        "float32[] frame_times_sec",
        "float32[] strengths",
        "bool[] detected",
    ]
    assert all("probability" not in field for field in fields)
    assert all("text" not in field for field in fields)


def test_pitch_frame_is_measurement_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "PitchFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "uint32 sample_rate",
        "uint32 input_sample_count",
        "uint32 n_fft",
        "uint32 hop_length",
        "uint32 frame_count",
        "float32 f_min_hz",
        "float32 f_max_hz",
        "float32 confidence_threshold",
        "string method",
        "string layout",
        "float32[] frame_times_sec",
        "float32[] frequencies_hz",
        "float32[] confidence",
        "bool[] voiced",
    ]
    assert all("probability" not in field for field in fields)
    assert all("text" not in field for field in fields)


def test_tempo_frame_is_measurement_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "TempoFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "uint32 sample_rate",
        "uint32 input_sample_count",
        "uint32 n_fft",
        "uint32 hop_length",
        "uint32 frame_count",
        "float32 bpm_min",
        "float32 bpm_max",
        "float32 tempo_bpm",
        "float32 confidence",
        "float32 confidence_threshold",
        "uint32 beat_period_frames",
        "bool tempo_detected",
        "string method",
        "string layout",
        "float32[] frame_times_sec",
        "float32[] onset_envelope",
        "bool[] beats",
    ]
    assert all("probability" not in field for field in fields)
    assert all("text" not in field for field in fields)


def test_stft_frame_is_feature_payload_only() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "StftFrame.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "uint32 sample_rate",
        "uint32 input_sample_count",
        "uint32 n_fft",
        "uint32 hop_length",
        "uint32 frame_count",
        "uint32 bin_count",
        "string window",
        "string layout",
        "string value_format",
        "float32[] real",
        "float32[] imag",
    ]
    assert all("probability" not in field for field in fields)
    assert all("detected" not in field for field in fields)
    assert all("text" not in field for field in fields)


def test_vad_state_carries_audio_stream_identity() -> None:
    msg_path = Path(__file__).parents[2] / "msg" / "VadState.msg"
    fields = [
        line.strip()
        for line in msg_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert fields == [
        "std_msgs/Header header",
        "string source_id",
        "string stream_id",
        "float32 probability",
        "bool is_speech",
        "bool start",
        "bool end",
    ]

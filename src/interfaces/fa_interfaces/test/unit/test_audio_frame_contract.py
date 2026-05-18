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

from pathlib import Path


def test_resample_preserves_source_and_publishes_stream_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_resample_node.cpp").read_text(encoding="utf-8")

    assert "out.source_id = in.source_id;" in source
    assert "out.stream_id = output_stream_id;" in source
    assert "out.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source


def test_resample_rejects_non_interleaved_or_unidentified_frames() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_resample_node.cpp").read_text(encoding="utf-8")

    assert "in.source_id.empty() || in.stream_id.empty()" in source
    assert "msg.layout != kInterleavedLayout" in source

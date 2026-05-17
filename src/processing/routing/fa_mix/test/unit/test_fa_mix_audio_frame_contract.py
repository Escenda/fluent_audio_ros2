from pathlib import Path


def test_mix_outputs_waveform_frame_identity_without_analysis_metadata() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_mix_node.cpp").read_text(encoding="utf-8")

    assert "msg.source_id.empty() || msg.stream_id.empty()" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "out.source_id = base.source_id;" in source
    assert "out.stream_id = config_.output_topic;" in source
    assert "out.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source

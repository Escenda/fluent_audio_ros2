from pathlib import Path


def test_mix_outputs_waveform_frame_identity_without_analysis_metadata() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_mix_node.cpp").read_text(encoding="utf-8")

    assert "msg.source_id.empty()" in source
    assert "msg.stream_id != expected_stream_id" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "out.source_id = base.source_id;" in source
    assert "out.stream_id = config_.output_topic;" in source
    assert "out.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source


def test_mix_drops_overflow_instead_of_hidden_limiter_clamp() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_mix_node.cpp").read_text(encoding="utf-8")
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package_root / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )

    assert "std::clamp" not in source
    assert "mixed sample out of normalized PCM16 range" in source
    assert "Dropping mixed frame" in source
    assert "hidden range clamp" in spec
    assert "hidden limiter" in algorithm


def test_mix_requires_all_configured_inputs_before_publish() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_mix_node.cpp").read_text(encoding="utf-8")
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")

    assert "input %zu has no valid frame" in source
    assert "input %zu is stale" in source
    assert "hasValidFrameStamp" in source
    assert "frameStamp(base)" in source
    assert "latest_frames_time_" not in source
    assert "base_time - other_time" in source
    assert "sample count differs from master" in source
    assert "std::min" not in source
    assert "partial mix" in spec


def test_mix_rejects_implicit_zero_db_gain_fallback() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_mix_node.cpp").read_text(encoding="utf-8")
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")

    assert "config_.input_gains_db.empty()" in source
    assert "input_gains_db must be size 1 or match input_topics length" in source
    assert "implicit 0 dB" in spec

from fa_dialogue_py.wake_ack_tone import WakeAckToneConfig, synthesize_wake_ack_pcm16


def test_wake_ack_tone_is_pcm16_mono_with_expected_duration() -> None:
    config = WakeAckToneConfig(sample_rate=48000, duration_ms=260, gain=0.18)

    audio = synthesize_wake_ack_pcm16(config)

    assert len(audio) == 48000 * 260 // 1000 * 2
    assert any(byte != 0 for byte in audio)


def test_wake_ack_tone_config_rejects_invalid_values() -> None:
    invalid = [
        {"sample_rate": 0},
        {"channels": 2},
        {"duration_ms": 0},
        {"fade_ms": -1},
        {"fade_ms": 200, "duration_ms": 260},
        {"gain": 0.0},
        {"base_hz": 0.0},
    ]
    for overrides in invalid:
        values = {
            "sample_rate": 48000,
            "channels": 1,
            "duration_ms": 260,
            "fade_ms": 34,
            "gain": 0.18,
            "base_hz": 660.0,
            "lift_hz": 260.0,
            "shimmer_hz": 1320.0,
        }
        values.update(overrides)
        try:
            WakeAckToneConfig(**values)
        except ValueError:
            continue
        raise AssertionError(f"invalid config was accepted: {overrides}")

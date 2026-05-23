from __future__ import annotations

from fluent_audio_system.config_schema import AudioNodeSpec, ParamValue
from fluent_audio_system.site_binding import SiteBindingOverrides


SOURCE_BOUND_AUDIO_AI_PACKAGES = frozenset(
    ("fa_asr", "fa_audio_embedding", "fa_kws", "fa_turn_detector", "fa_vad")
)
SOURCE_BOUND_CONTROL_PACKAGES = frozenset(("fa_kws", "fa_turn_detector"))
SOURCE_BOUND_STREAMING_PACKAGES = frozenset(("fa_audio_window",))


def node_enabled_by_site_binding(
    node: AudioNodeSpec,
    overrides: SiteBindingOverrides,
) -> bool:
    if node.package == "fa_in" and node.backend_name == "alsa_capture":
        return overrides.fa_in_enabled
    if node.package == "fa_out" and node.backend_name == "alsa_playback":
        return overrides.fa_out_enabled
    return True


def node_launch_parameters(
    node: AudioNodeSpec,
    overrides: SiteBindingOverrides,
) -> list[str | dict[str, ParamValue]]:
    parameters = node.launch_parameters()
    override_params: dict[str, ParamValue] = {}
    if (
        node.package == "fa_in"
        and node.backend_name == "alsa_capture"
        and overrides.fa_in_source_id
    ):
        override_params["audio.device_selector.mode"] = "id"
        override_params["audio.device_selector.identifier"] = overrides.fa_in_source_id
    if node.package in SOURCE_BOUND_AUDIO_AI_PACKAGES and overrides.fa_in_source_id:
        override_params["expected_source_id"] = overrides.fa_in_source_id
    if node.package in SOURCE_BOUND_CONTROL_PACKAGES and overrides.fa_in_source_id:
        for control_id in _control_input_ids(node):
            override_params[f"control.{control_id}.source_id"] = overrides.fa_in_source_id
    if node.package in SOURCE_BOUND_STREAMING_PACKAGES and overrides.fa_in_source_id:
        override_params["input.source_id"] = overrides.fa_in_source_id
    if (
        node.package == "fa_out"
        and node.backend_name == "alsa_playback"
        and overrides.fa_out_sink_id
    ):
        override_params["audio.device_id"] = overrides.fa_out_sink_id
    if override_params:
        parameters.append(override_params)
    return parameters


def _control_input_ids(node: AudioNodeSpec) -> list[str]:
    value = node.parameters.get(
        "control.inputs",
        node.params_file_parameters.get("control.inputs"),
    )
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return []
    return value

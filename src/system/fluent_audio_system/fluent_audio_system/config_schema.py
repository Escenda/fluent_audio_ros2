from __future__ import annotations

import os
import re
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import TypeAlias, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
import yaml


ScalarParam = str | int | float | bool
ParamValue = ScalarParam | list[str] | list[int] | list[float] | list[bool]
ConfigScalar: TypeAlias = str | int | float | bool | None
ConfigMapping: TypeAlias = dict[str, "ConfigValue"]
ConfigSequence: TypeAlias = list["ConfigValue"]
ConfigValue: TypeAlias = ConfigScalar | ConfigMapping | ConfigSequence
_INLINE_SHARE_RE = re.compile(r"\$\{share:([A-Za-z0-9_]+)\}")
_INLINE_ENV_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
_AI_PACKAGE_NAMES = (
    "fa_audio_embedding",
    "fa_kws",
    "fa_turn_detector",
)
_STREAMING_PACKAGE_NAMES = (
    "fa_audio_window",
    "fa_chunk_overlap",
    "fa_clock_drift",
    "fa_frame_buffer",
    "fa_jitter_buffer",
    "fa_latency_compensation",
    "fa_overlap_add",
    "fa_packet_loss_concealment",
    "fa_time_alignment",
)
_PACKAGE_CATEGORIES = {
    "fa_in": frozenset(("io",)),
    "fa_out": frozenset(("io",)),
    "fa_record": frozenset(("io",)),
    "fa_stream": frozenset(("io",)),
    "fa_resample": frozenset(("format",)),
    "fa_bit_depth": frozenset(("format",)),
    "fa_channel_convert": frozenset(("format",)),
    "fa_interleave": frozenset(("format",)),
    "fa_sample_format": frozenset(("format",)),
    "fa_encode": frozenset(("format",)),
    "fa_decode": frozenset(("format",)),
    "fa_gain": frozenset(("dynamics",)),
    "fa_normalize": frozenset(("dynamics",)),
    "fa_compressor": frozenset(("dynamics",)),
    "fa_limiter": frozenset(("dynamics",)),
    "fa_expander": frozenset(("dynamics",)),
    "fa_noise_gate": frozenset(("dynamics",)),
    "fa_agc": frozenset(("dynamics",)),
    "fa_eq": frozenset(("frequency",)),
    "fa_low_pass": frozenset(("frequency",)),
    "fa_high_pass": frozenset(("frequency",)),
    "fa_band_pass": frozenset(("frequency",)),
    "fa_notch": frozenset(("frequency",)),
    "fa_deesser": frozenset(("frequency",)),
    "fa_trim": frozenset(("temporal",)),
    "fa_silence_removal": frozenset(("temporal",)),
    "fa_delay": frozenset(("temporal",)),
    "fa_echo": frozenset(("temporal",)),
    "fa_reverb": frozenset(("temporal",)),
    "fa_crossfade": frozenset(("temporal",)),
    "fa_fade": frozenset(("temporal",)),
    "fa_window": frozenset(("temporal",)),
    "fa_denoise": frozenset(("correction",)),
    "fa_aec_linear": frozenset(("correction",)),
    "fa_aec_nn": frozenset(("correction",)),
    "fa_declick": frozenset(("correction",)),
    "fa_hum": frozenset(("correction",)),
    "fa_dc_offset_removal": frozenset(("correction",)),
    "fa_pan": frozenset(("spatial",)),
    "fa_stereo_widening": frozenset(("spatial",)),
    "fa_downmix": frozenset(("spatial",)),
    "fa_upmix": frozenset(("spatial",)),
    "fa_beamforming": frozenset(("spatial",)),
    "fa_onset": frozenset(("analysis",)),
    "fa_pitch": frozenset(("analysis",)),
    "fa_tempo": frozenset(("analysis",)),
    "fa_stft": frozenset(("analysis",)),
    "fa_log_mel": frozenset(("analysis",)),
    "fa_mfcc": frozenset(("analysis",)),
    "fa_cqt": frozenset(("analysis",)),
    "fa_loudness": frozenset(("analysis",)),
    "fa_tts": frozenset(("generation",)),
    "fa_mix": frozenset(("routing",)),
    "fa_bus_router": frozenset(("routing",)),
    "fa_sidechain": frozenset(("routing",)),
    "fa_ducking": frozenset(("routing",)),
    "fa_monitor_mix": frozenset(("routing",)),
    "fa_loopback": frozenset(("routing",)),
    "fa_patchbay": frozenset(("routing",)),
    "fa_kws": frozenset(("ai",)),
    "fa_turn_detector": frozenset(("ai",)),
    "fa_audio_embedding": frozenset(("ai",)),
    "fa_audio_mcp": frozenset(("apps",)),
    "fa_dialogue": frozenset(("apps",)),
    "fa_frame_buffer": frozenset(("streaming",)),
    "fa_jitter_buffer": frozenset(("streaming",)),
    "fa_clock_drift": frozenset(("streaming",)),
    "fa_packet_loss_concealment": frozenset(("streaming",)),
    "fa_latency_compensation": frozenset(("streaming",)),
    "fa_time_alignment": frozenset(("streaming",)),
    "fa_chunk_overlap": frozenset(("streaming",)),
    "fa_overlap_add": frozenset(("streaming",)),
    "fa_audio_window": frozenset(("streaming",)),
    "fa_voice_command_router": frozenset(("apps",)),
}
_GROUP_CATEGORY_ALIASES = {
    "app": "apps",
    "apps": "apps",
    "application": "apps",
    "applications": "apps",
    "source": "io",
    "sources": "io",
    "sink": "io",
    "sinks": "io",
}
_GROUP_CATEGORY_TOKENS = frozenset((
    "io",
    "format",
    "dynamics",
    "frequency",
    "temporal",
    "correction",
    "spatial",
    "analysis",
    "generation",
    "routing",
    "ai",
    "streaming",
    "apps",
))
_GROUP_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BASE_REQUIRED_PACKAGES = ("fa_interfaces", "fluent_audio_system")
_BACKEND_NAME_REQUIRED_PACKAGES = frozenset((
    "fa_in",
    "fa_out",
    "fa_encode",
    "fa_decode",
    "fa_aec_nn",
    "fa_denoise",
    "fa_cqt",
    "fa_log_mel",
    "fa_loudness",
    "fa_mfcc",
    "fa_onset",
    "fa_pitch",
    "fa_stft",
    "fa_tempo",
    "fa_kws",
    "fa_turn_detector",
    "fa_audio_embedding",
    "fa_tts",
))


class _TimingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    default_start_delay: float | int
    inter_group_delay: float | int

    @model_validator(mode="after")
    def _validate_delays(self) -> "_TimingConfig":
        if isinstance(self.default_start_delay, bool) or self.default_start_delay < 0.0:
            raise ValueError("system.default_start_delay must be >= 0")
        if isinstance(self.inter_group_delay, bool) or self.inter_group_delay < 0.0:
            raise ValueError("system.inter_group_delay must be >= 0")
        return self


RemappingConfigValue: TypeAlias = dict[str, str]


class _NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str | None = None
    enable: bool | None = None
    package: str | None = None
    executable: str | None = Field(default=None, alias="exec")
    node_name: str | None = None
    namespace: str | None = None
    output: str | None = None
    params_file: str | None = None
    parameters: dict[str, ParamValue] | None = None
    remappings: RemappingConfigValue | None = None
    env: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "_NodeConfig":
        node_id = _required_model_text(self.id, "node id")
        if self.enable is None:
            raise ValueError(f"node {node_id}.enable is required")
        if not self.enable:
            return self
        _required_model_text(self.package, f"node {node_id}.package")
        _required_model_text(self.executable, f"node {node_id}.exec")
        _required_model_text(self.node_name, f"node {node_id}.node_name")
        _required_model_text(self.params_file, f"node {node_id}.params_file")
        return self


class _GroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str | None = None
    enable: bool | None = None
    nodes: list[_NodeConfig] | None = None

    @model_validator(mode="after")
    def _validate_required_fields(self) -> "_GroupConfig":
        group_id = _required_model_text(self.id, "group id")
        if self.enable is None:
            raise ValueError(f"group {group_id}.enable is required")
        if self.enable and self.nodes is None:
            raise ValueError(f"group {group_id}.nodes is required")
        return self


class _AudioSystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    system: _TimingConfig
    groups: list[_GroupConfig]


@dataclass(frozen=True)
class ParameterIdentityContract:
    topic_keys: frozenset[str]
    stream_identity_keys: frozenset[str]


@dataclass(frozen=True)
class RemappingSpec:
    source: str
    target: str


@dataclass(frozen=True)
class AudioNodeSpec:
    id: str
    package: str
    executable: str
    node_name: str
    namespace: str
    output: str
    params_file: str
    parameters: dict[str, ParamValue]
    remappings: list[RemappingSpec]
    backend_name: str | None
    params_file_parameters: dict[str, ParamValue] = dataclass_field(default_factory=dict)
    env: dict[str, str] = dataclass_field(default_factory=dict)

    def launch_parameters(self) -> list[str | dict[str, ParamValue]]:
        sources: list[str | dict[str, ParamValue]] = []
        if self.params_file:
            sources.append(self.params_file)
        launchable_params_file_parameters = {
            key: value
            for key, value in self.params_file_parameters.items()
            # launch_ros cannot transport empty arrays in dict parameter sources.
            if not (isinstance(value, list) and not value)
        }
        if launchable_params_file_parameters:
            sources.append(launchable_params_file_parameters)
        if self.parameters:
            sources.append(self.parameters)
        return sources

    def launch_remappings(self) -> list[tuple[str, str]]:
        return [(item.source, item.target) for item in self.remappings]


@dataclass(frozen=True)
class AudioGroupSpec:
    id: str
    nodes: list[AudioNodeSpec]


@dataclass(frozen=True)
class AudioSystemSpec:
    default_start_delay: float
    inter_group_delay: float
    groups: list[AudioGroupSpec]


_SOURCE_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("output_topic",)),
    stream_identity_keys=frozenset(("audio.stream_id", "output.stream_id")),
)
_SINK_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("input_topic",)),
    stream_identity_keys=frozenset(("input_stream_id",)),
)
_FA_OUT_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("input_topic", "playback_done_topic")),
    stream_identity_keys=frozenset(("input_stream_id",)),
)
_SIMPLE_PROCESSING_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("input_topic", "output_topic")),
    stream_identity_keys=frozenset(("input_stream_id", "output.stream_id")),
)
_FEATURE_ANALYSIS_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("input_topic", "output_topic")),
    stream_identity_keys=frozenset(("expected.stream_id", "output.stream_id")),
)
_RESAMPLE_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset((
        "mic.input_topic",
        "mic.output_topic",
        "ref.input_topic",
        "ref.output_topic",
    )),
    stream_identity_keys=frozenset((
        "mic.input_stream_id",
        "mic.output.stream_id",
        "ref.input_stream_id",
        "ref.output.stream_id",
    )),
)
_AI_AUDIO_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset((
        "audio_topic",
        "input_topic",
        "output_topic",
        "turn_context_topic",
    )),
    stream_identity_keys=frozenset(("expected_stream_id", "input_stream_id")),
)
_KWS_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset((
        "audio_topic",
        "output_topic",
    )),
    stream_identity_keys=frozenset(("expected_stream_id",)),
)
_TURN_DETECTOR_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset((
        "audio_topic",
        "turn_context_topic",
        "output_topic",
    )),
    stream_identity_keys=frozenset(("expected_stream_id",)),
)
_AUDIO_WINDOW_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("input_topic",)),
    stream_identity_keys=frozenset(("input.stream_id",)),
)
_MIX_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("input_topics", "output_topic")),
    stream_identity_keys=frozenset(("input_stream_ids", "output.stream_id")),
)
_BUS_ROUTER_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("input_topic", "output_topics")),
    stream_identity_keys=frozenset(("input_stream_id", "output.stream_ids")),
)
_PATCHBAY_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("input_topics", "output_topics")),
    stream_identity_keys=frozenset(("input_stream_ids", "output_stream_ids")),
)
_AEC_LINEAR_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("mic_topic", "ref_topic", "output_topic")),
    stream_identity_keys=frozenset(("mic_stream_id", "ref_stream_id", "output.stream_id")),
)
_CROSSFADE_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("input_a_topic", "input_b_topic", "output_topic")),
    stream_identity_keys=frozenset((
        "input_a_stream_id",
        "input_b_stream_id",
        "output.stream_id",
    )),
)
_DUCKING_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("program_topic", "sidechain_topic", "output_topic")),
    stream_identity_keys=frozenset((
        "program_stream_id",
        "sidechain_stream_id",
        "output.stream_id",
    )),
)
_SIDECHAIN_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("sidechain_topic", "control_topic")),
    stream_identity_keys=frozenset(("sidechain_stream_id", "output.stream_id")),
)
_TTS_CONTRACT = ParameterIdentityContract(
    topic_keys=frozenset(("output_topic",)),
    stream_identity_keys=frozenset(("output.stream_id",)),
)
_SIMPLE_PROCESSING_PACKAGES = (
    "fa_agc",
    "fa_band_pass",
    "fa_beamforming",
    "fa_bit_depth",
    "fa_channel_convert",
    "fa_chunk_overlap",
    "fa_clock_drift",
    "fa_compressor",
    "fa_dc_offset_removal",
    "fa_declick",
    "fa_deesser",
    "fa_delay",
    "fa_denoise",
    "fa_downmix",
    "fa_echo",
    "fa_eq",
    "fa_expander",
    "fa_fade",
    "fa_gain",
    "fa_high_pass",
    "fa_hum",
    "fa_interleave",
    "fa_jitter_buffer",
    "fa_latency_compensation",
    "fa_limiter",
    "fa_loopback",
    "fa_low_pass",
    "fa_noise_gate",
    "fa_normalize",
    "fa_notch",
    "fa_overlap_add",
    "fa_packet_loss_concealment",
    "fa_pan",
    "fa_reverb",
    "fa_sample_format",
    "fa_silence_removal",
    "fa_stereo_widening",
    "fa_time_alignment",
    "fa_trim",
    "fa_upmix",
    "fa_window",
)
_FEATURE_ANALYSIS_PACKAGES = (
    "fa_cqt",
    "fa_log_mel",
    "fa_loudness",
    "fa_mfcc",
    "fa_onset",
    "fa_pitch",
    "fa_stft",
    "fa_tempo",
)
_PARAMETER_IDENTITY_CONTRACTS: dict[str, ParameterIdentityContract] = {
    "fa_in": _SOURCE_CONTRACT,
    "fa_out": _FA_OUT_CONTRACT,
    "fa_resample": _RESAMPLE_CONTRACT,
    "fa_encode": _SIMPLE_PROCESSING_CONTRACT,
    "fa_decode": _SIMPLE_PROCESSING_CONTRACT,
    "fa_aec_linear": _AEC_LINEAR_CONTRACT,
    "fa_aec_nn": _SIMPLE_PROCESSING_CONTRACT,
    "fa_crossfade": _CROSSFADE_CONTRACT,
    "fa_ducking": _DUCKING_CONTRACT,
    "fa_sidechain": _SIDECHAIN_CONTRACT,
    "fa_mix": _MIX_CONTRACT,
    "fa_monitor_mix": _MIX_CONTRACT,
    "fa_bus_router": _BUS_ROUTER_CONTRACT,
    "fa_patchbay": _PATCHBAY_CONTRACT,
    "fa_kws": _KWS_CONTRACT,
    "fa_turn_detector": _TURN_DETECTOR_CONTRACT,
    "fa_audio_embedding": _AI_AUDIO_CONTRACT,
    "fa_audio_window": _AUDIO_WINDOW_CONTRACT,
    "fa_tts": _TTS_CONTRACT,
    **{
        package_name: _SIMPLE_PROCESSING_CONTRACT
        for package_name in _SIMPLE_PROCESSING_PACKAGES
    },
    **{
        package_name: _FEATURE_ANALYSIS_CONTRACT
        for package_name in _FEATURE_ANALYSIS_PACKAGES
    },
}
_CONTROL_TOP_LEVEL_KEYS = frozenset((
    "control.inputs",
    "control.default_enabled",
))
_KWS_CONTROL_SUFFIXES = frozenset((
    "action",
    "topic",
    "msg_type",
    "source_id",
    "stream_id",
    "probability_field",
    "probability_gate",
    "max_age_ms",
    "qos.depth",
    "qos.reliable",
))
_TURN_DETECTOR_CONTROL_SUFFIXES = frozenset((
    "action",
    "topic",
    "msg_type",
    "source_id",
    "stream_id",
    "active_field",
    "start_field",
    "end_field",
    "close_on",
    "qos.depth",
    "qos.reliable",
))
_CONTROL_SUFFIXES_BY_PACKAGE: dict[str, frozenset[str]] = {
    "fa_kws": _KWS_CONTROL_SUFFIXES,
    "fa_turn_detector": _TURN_DETECTOR_CONTROL_SUFFIXES,
}
_DISALLOWED_PARAMETER_KEYS_BY_PACKAGE: dict[str, frozenset[str]] = {}
_BACKEND_REQUIRED_PARAMETERS: dict[tuple[str, str], frozenset[str]] = {}


def load_system_config(path: str) -> AudioSystemSpec:
    roots = _load_system_config_roots(path)
    _validate_composed_roots(roots)
    specs = [_parse_validated_system_config(root) for root in roots]
    return _compose_system_specs(specs)


def load_required_packages(path: str) -> list[str]:
    roots = _load_system_config_roots(path)
    _validate_composed_roots(roots)
    return _required_packages_for_validated_configs(roots)


def required_packages_for_system(spec: AudioSystemSpec) -> list[str]:
    packages = list(_BASE_REQUIRED_PACKAGES)
    seen = set(packages)
    for group in spec.groups:
        for node in group.nodes:
            if node.package in seen:
                continue
            packages.append(node.package)
            seen.add(node.package)
    return packages


def _required_packages_for_validated_config(root: _AudioSystemConfig) -> list[str]:
    packages = list(_BASE_REQUIRED_PACKAGES)
    seen = set(packages)
    _append_required_packages_for_validated_config(root, packages, seen)
    return packages


def _required_packages_for_validated_configs(
    roots: list[_AudioSystemConfig],
) -> list[str]:
    packages = list(_BASE_REQUIRED_PACKAGES)
    seen = set(packages)
    for root in roots:
        _append_required_packages_for_validated_config(root, packages, seen)
    return packages


def _append_required_packages_for_validated_config(
    root: _AudioSystemConfig,
    packages: list[str],
    seen: set[str],
) -> None:
    for group in root.groups:
        group_id = _required_model_text(group.id, "group id")
        _validate_group_taxonomy(group, group_id)
        if not group.enable:
            continue
        for node in group.nodes or []:
            if not node.enable:
                continue
            package = _required_model_text(node.package, f"node {node.id}.package")
            if package in seen:
                continue
            packages.append(package)
            seen.add(package)


def parse_system_config(raw: ConfigValue) -> AudioSystemSpec:
    root = _validate_system_config(raw)
    return _parse_validated_system_config(root)


def _parse_validated_system_config(root: _AudioSystemConfig) -> AudioSystemSpec:
    default_start_delay = float(root.system.default_start_delay)
    inter_group_delay = float(root.system.inter_group_delay)

    groups: list[AudioGroupSpec] = []
    for group in root.groups:
        group_id = _required_model_text(group.id, "group id")
        _validate_group_taxonomy(group, group_id)
        if not group.enable:
            continue
        nodes: list[AudioNodeSpec] = []
        for node in group.nodes or []:
            if not node.enable:
                continue
            nodes.append(_parse_node(node))
        groups.append(AudioGroupSpec(id=group_id, nodes=nodes))

    return AudioSystemSpec(
        default_start_delay=default_start_delay,
        inter_group_delay=inter_group_delay,
        groups=groups,
    )


def _load_system_config_roots(path: str) -> list[_AudioSystemConfig]:
    return [
        _load_system_config_root(config_path)
        for config_path in _split_config_argument(path)
    ]


def _split_config_argument(path: str) -> list[str]:
    if not path or not path.strip():
        raise RuntimeError("config launch argument is required")

    config_paths: list[str] = []
    for raw_segment in path.split(","):
        segment = raw_segment.strip()
        if not segment:
            raise RuntimeError("config launch argument contains an empty path segment")
        config_paths.append(segment)
    return config_paths


def _load_system_config_root(path: str) -> _AudioSystemConfig:
    resolved_path = _resolve_config_refs(path)
    if not os.path.isfile(resolved_path):
        raise RuntimeError(f"fluent_audio_system config not found: {resolved_path}")
    try:
        with open(resolved_path, "r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise RuntimeError(
            f"fluent_audio_system config invalid YAML: {resolved_path}: {exc}"
        ) from exc
    return _validate_system_config(raw)


def _validate_composed_roots(roots: list[_AudioSystemConfig]) -> None:
    if not roots:
        raise RuntimeError("config launch argument is required")
    first = roots[0].system
    expected_default_start_delay = float(first.default_start_delay)
    expected_inter_group_delay = float(first.inter_group_delay)
    group_ids: set[str] = set()
    enabled_node_ids: set[str] = set()

    for root in roots:
        default_start_delay = float(root.system.default_start_delay)
        inter_group_delay = float(root.system.inter_group_delay)
        if default_start_delay != expected_default_start_delay:
            raise RuntimeError(
                "system.default_start_delay must match across composed config files"
            )
        if inter_group_delay != expected_inter_group_delay:
            raise RuntimeError(
                "system.inter_group_delay must match across composed config files"
            )
        for group in root.groups:
            group_id = _required_model_text(group.id, "group id")
            if group_id in group_ids:
                raise RuntimeError(
                    f"duplicate group id across composed config files: {group_id}"
                )
            group_ids.add(group_id)
            if not group.enable:
                continue
            for node in group.nodes or []:
                if not node.enable:
                    continue
                node_id = _required_model_text(node.id, "node id")
                if node_id in enabled_node_ids:
                    raise RuntimeError(
                        "duplicate enabled node id across composed config files: "
                        f"{node_id}"
                    )
                enabled_node_ids.add(node_id)


def _compose_system_specs(specs: list[AudioSystemSpec]) -> AudioSystemSpec:
    if not specs:
        raise RuntimeError("config launch argument is required")
    first = specs[0]
    groups: list[AudioGroupSpec] = []
    for spec in specs:
        groups.extend(spec.groups)
    return AudioSystemSpec(
        default_start_delay=first.default_start_delay,
        inter_group_delay=first.inter_group_delay,
        groups=groups,
    )


def _validate_system_config(raw: ConfigValue) -> _AudioSystemConfig:
    try:
        return _AudioSystemConfig.model_validate(raw)
    except ValidationError as exc:
        raise RuntimeError(_format_validation_error(exc)) from exc
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _parse_node(node: _NodeConfig) -> AudioNodeSpec:
    node_id = _required_model_text(node.id, "node id")
    package = _required_model_text(node.package, f"node {node_id}.package")
    executable = _required_model_text(node.executable, f"node {node_id}.exec")
    node_name = _required_model_text(node.node_name, f"node {node_id}.node_name")
    namespace = _optional_model_text(node.namespace, "")
    if namespace == "/":
        namespace = ""
    output = _optional_model_text(node.output, "screen")
    params_file = _resolve_config_refs(
        _required_model_text(node.params_file, f"node {node_id}.params_file")
    )
    if not os.path.isfile(params_file):
        raise RuntimeError(f"params_file not found: {params_file}")
    params_file_parameters = _params_file_parameters(params_file, package, node_name, node_id)
    _validate_disallowed_parameter_keys(package, node_id, params_file_parameters)
    _validate_parameter_identity_contract(package, node_id, params_file_parameters)
    parameters = _optional_parameters(node.parameters, node_id)
    _validate_disallowed_parameter_keys(package, node_id, parameters)
    _validate_parameter_identity_contract(package, node_id, parameters)
    remappings = _optional_remappings(node.remappings, node_id)
    env = _optional_env(node.env, node_id)
    backend_name = _effective_backend_name(params_file_parameters, parameters, node_id)
    if package in _BACKEND_NAME_REQUIRED_PACKAGES and backend_name is None:
        raise RuntimeError(f"node {node_id}.backend.name is required for {package}")
    if backend_name is not None:
        _validate_backend_required_parameters(
            package,
            node_id,
            backend_name,
            parameters,
        )
    return AudioNodeSpec(
        id=node_id,
        package=package,
        executable=executable,
        node_name=node_name,
        namespace=namespace,
        output=output,
        params_file=params_file,
        parameters=parameters,
        remappings=remappings,
        env=env,
        backend_name=backend_name,
        params_file_parameters=params_file_parameters,
    )


def _validate_group_taxonomy(group: _GroupConfig, group_id: str) -> None:
    group_categories = _group_categories(group_id)
    for node in group.nodes or []:
        if node.package is None:
            continue
        package = node.package.strip()
        package_categories = _PACKAGE_CATEGORIES.get(package)
        if package_categories is None:
            raise RuntimeError(f"group {group_id} contains unsupported FluentAudio package {package}")
        if "analysis" in group_categories and package in _AI_PACKAGE_NAMES:
            raise RuntimeError(
                f"group {group_id} must not contain AI package {package}; "
                "use an ai or voice_frontend group"
            )
        if "streaming" not in group_categories and package in _STREAMING_PACKAGE_NAMES:
            raise RuntimeError(
                f"group {group_id} must not contain streaming package {package}; "
                "use a streaming group"
            )
        if group_categories and package_categories is not None and not (
            group_categories & package_categories
        ):
            raise RuntimeError(
                f"group {group_id} must not contain {package}; "
                f"package category is {_format_categories(package_categories)}"
            )


def _group_categories(group_id: str) -> frozenset[str]:
    group_id_normalized = group_id.strip().lower()
    if group_id_normalized == "voice_frontend":
        return frozenset(("ai",))

    categories = set()
    for token in _GROUP_TOKEN_RE.findall(group_id_normalized):
        category = _GROUP_CATEGORY_ALIASES.get(token, token)
        if category in _GROUP_CATEGORY_TOKENS:
            categories.add(category)
    return frozenset(categories)


def _format_categories(categories: frozenset[str]) -> str:
    return "/".join(sorted(categories))


def _required_model_text(value: str | None, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _optional_model_text(value: str | None, default_value: str = "") -> str:
    if value is None:
        return default_value
    if not isinstance(value, str):
        raise RuntimeError("optional text must be a string")
    return value.strip()


def _format_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors()[0]
    location = ".".join(str(part) for part in first_error["loc"])
    message = str(first_error["msg"])
    if "parameters" in first_error["loc"]:
        return "unsupported value type"
    if first_error["type"] == "missing":
        return f"{location} is required"
    if message.startswith("Value error, "):
        return message.removeprefix("Value error, ")
    return f"{location}: {message}"


def _optional_parameters(
    value: dict[str, ParamValue] | None,
    node_id: str,
) -> dict[str, ParamValue]:
    if value is None:
        return {}
    params: dict[str, ParamValue] = {}
    for key, param_value in value.items():
        if not isinstance(key, str) or not key.strip():
            raise RuntimeError(f"node {node_id} parameter keys must be non-empty strings")
        expanded_value = _expand_param_value(param_value)
        if not _is_param_value(expanded_value):
            raise RuntimeError(f"node {node_id} parameter '{key}' has unsupported value type")
        params[key.strip()] = expanded_value
    return params


def _effective_backend_name(
    params_file_parameters: dict[str, ParamValue],
    inline_parameters: dict[str, ParamValue],
    node_id: str,
) -> str | None:
    value = inline_parameters.get("backend.name", params_file_parameters.get("backend.name"))
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"node {node_id}.backend.name must be a non-empty string")
    return value.strip()


def _validate_backend_required_parameters(
    package: str,
    node_id: str,
    backend_name: str,
    inline_parameters: dict[str, ParamValue],
) -> None:
    required_parameters = _BACKEND_REQUIRED_PARAMETERS.get((package, backend_name))
    if required_parameters is None:
        return
    for key in sorted(required_parameters):
        value = inline_parameters.get(key)
        if value is None:
            raise RuntimeError(f"node {node_id}.{key} is required for {backend_name}")
        _validate_backend_required_parameter_value(node_id, backend_name, key, value)


def _validate_backend_required_parameter_value(
    node_id: str,
    backend_name: str,
    key: str,
    value: ParamValue,
) -> None:
    if key in _NEMO_RNNT_STREAMING_REQUIRED_TEXT_PARAMETERS:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"node {node_id}.{key} is required for {backend_name}")
        return
    if key in _NEMO_RNNT_STREAMING_REQUIRED_INT_PARAMETERS:
        if type(value) is not int:
            raise RuntimeError(
                f"node {node_id}.{key} must be an integer for {backend_name}"
            )
        return
    if key in _NEMO_RNNT_STREAMING_REQUIRED_NUMBER_PARAMETERS:
        if type(value) not in (float, int):
            raise RuntimeError(
                f"node {node_id}.{key} must be a number for {backend_name}"
            )
        return
    if key in _NEMO_RNNT_STREAMING_REQUIRED_BOOL_PARAMETERS:
        if type(value) is not bool:
            raise RuntimeError(
                f"node {node_id}.{key} must be a bool for {backend_name}"
            )
        return


def _params_file_parameters(
    params_file: str,
    package: str,
    node_name: str,
    node_id: str,
) -> dict[str, ParamValue]:
    with open(params_file, "r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if raw is None:
        raise RuntimeError(
            f"params_file {params_file} must define ros__parameters for node {node_id}"
        )
    if not isinstance(raw, dict):
        raise RuntimeError(f"params_file {params_file} root must be a mapping")

    for root_key in (node_name, package, "/**"):
        if root_key not in raw:
            continue
        candidate = raw[root_key]
        if not isinstance(candidate, dict):
            raise RuntimeError(f"params_file {params_file}.{root_key} must be a mapping")
        if "ros__parameters" not in candidate:
            continue
        parameters = candidate["ros__parameters"]
        if not isinstance(parameters, dict):
            raise RuntimeError(
                f"params_file {params_file}.{root_key}.ros__parameters must be a mapping"
            )
        flattened: dict[str, ParamValue] = {}
        _flatten_params_file_parameters(
            parameters,
            "",
            flattened,
            f"params_file {params_file}.{root_key}.ros__parameters",
            node_id,
        )
        return flattened
    raise RuntimeError(f"params_file {params_file} must define ros__parameters for node {node_id}")


def _flatten_params_file_parameters(
    mapping: ConfigMapping,
    prefix: str,
    flattened: dict[str, ParamValue],
    label: str,
    node_id: str,
) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise RuntimeError(f"{label} keys for node {node_id} must be non-empty strings")
        param_key = f"{prefix}.{key.strip()}" if prefix else key.strip()
        if isinstance(value, dict):
            _flatten_params_file_parameters(value, param_key, flattened, label, node_id)
            continue
        if not _is_param_value(value):
            raise RuntimeError(
                f"{label}.{param_key} for node {node_id} has unsupported value type"
            )
        flattened[param_key] = value


def _validate_parameter_identity_contract(
    package: str,
    node_id: str,
    parameters: dict[str, ParamValue],
) -> None:
    contract = _PARAMETER_IDENTITY_CONTRACTS.get(package)
    if contract is None or not parameters:
        return
    if package in _CONTROL_SUFFIXES_BY_PACKAGE:
        contract = _control_identity_contract(package, contract, parameters, node_id)

    topic_values = _contract_values(parameters, contract.topic_keys, node_id)
    if not topic_values:
        return

    topic_identities = {
        _identity_value(value)
        for values in topic_values.values()
        for value in values
        if _identity_value(value)
    }
    if not topic_identities:
        return

    stream_values = _contract_values(parameters, contract.stream_identity_keys, node_id)
    for stream_key, values in stream_values.items():
        for value in values:
            identity = _identity_value(value)
            if identity and identity in topic_identities:
                raise RuntimeError(
                    f"node {node_id} parameter '{stream_key}' must not reuse ROS topic "
                    f"value '{value}'"
                )


def _validate_disallowed_parameter_keys(
    package: str,
    node_id: str,
    parameters: dict[str, ParamValue],
) -> None:
    disallowed_keys = _DISALLOWED_PARAMETER_KEYS_BY_PACKAGE.get(package)
    if disallowed_keys is None:
        return
    for key in sorted(disallowed_keys):
        if key in parameters:
            raise RuntimeError(
                f"node {node_id} parameter '{key}' is not supported; "
                "use control.inputs and control.<id>.* parameters"
            )


def _control_identity_contract(
    package: str,
    base_contract: ParameterIdentityContract,
    parameters: dict[str, ParamValue],
    node_id: str,
) -> ParameterIdentityContract:
    control_parameter_ids = _control_parameter_ids(package, parameters)
    if "control.inputs" not in parameters:
        if control_parameter_ids:
            raise RuntimeError(
                f"node {node_id}.control.inputs is required when control parameters are set"
            )
        return base_contract

    control_ids = _control_inputs(parameters["control.inputs"], node_id)
    for control_parameter_id in control_parameter_ids:
        if control_parameter_id not in control_ids:
            raise RuntimeError(
                f"node {node_id} control parameter '{control_parameter_id}' "
                "must be listed in control.inputs"
            )

    control_topic_keys = {
        f"control.{control_id}.topic"
        for control_id in control_ids
    }
    control_stream_identity_keys = {
        f"control.{control_id}.stream_id"
        for control_id in control_ids
    }
    for control_key in sorted(control_topic_keys | control_stream_identity_keys):
        if control_key not in parameters:
            raise RuntimeError(f"node {node_id} parameter '{control_key}' is required")

    return ParameterIdentityContract(
        topic_keys=base_contract.topic_keys | frozenset(control_topic_keys),
        stream_identity_keys=(
            base_contract.stream_identity_keys | frozenset(control_stream_identity_keys)
        ),
    )


def _control_inputs(value: ParamValue, node_id: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeError(
            f"node {node_id} parameter 'control.inputs' must be a list of strings"
        )

    control_ids: list[str] = []
    seen: set[str] = set()
    for control_id in value:
        if not control_id:
            raise RuntimeError("control.inputs must not contain empty IDs")
        if control_id != control_id.strip():
            raise RuntimeError("control.inputs IDs must not contain surrounding whitespace")
        if control_id in seen:
            raise RuntimeError("control.inputs must not contain duplicate IDs")
        seen.add(control_id)
        control_ids.append(control_id)
    return control_ids


def _control_parameter_ids(package: str, parameters: dict[str, ParamValue]) -> set[str]:
    supported_suffixes = _CONTROL_SUFFIXES_BY_PACKAGE[package]
    control_parameter_ids: set[str] = set()
    for key in parameters:
        if not key.startswith("control."):
            continue
        if key in _CONTROL_TOP_LEVEL_KEYS:
            continue
        segments = key.split(".")
        if len(segments) < 3 or not segments[1]:
            raise RuntimeError(f"unsupported {package} control parameter '{key}'")
        suffix = ".".join(segments[2:])
        if suffix not in supported_suffixes:
            raise RuntimeError(f"unsupported {package} control parameter '{key}'")
        control_parameter_ids.add(segments[1])
    return control_parameter_ids


def _contract_values(
    parameters: dict[str, ParamValue],
    keys: frozenset[str],
    node_id: str,
) -> dict[str, list[str]]:
    values_by_key: dict[str, list[str]] = {}
    for key in keys:
        if key not in parameters:
            continue
        values_by_key[key] = _string_values_for_contract_key(
            parameters[key], f"node {node_id} parameter '{key}'"
        )
    return values_by_key


def _string_values_for_contract_key(value: ParamValue, label: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise RuntimeError(f"{label} must be a string or a list of strings")


def _identity_value(value: str) -> str:
    return value.strip().lstrip("/")


def _optional_remappings(
    value: RemappingConfigValue | None,
    node_id: str,
) -> list[RemappingSpec]:
    if value is None:
        return []
    remappings: list[RemappingSpec] = []
    for source, target in value.items():
        if not isinstance(source, str) or not source.strip():
            raise RuntimeError(f"node {node_id} remapping source must be a non-empty string")
        if not isinstance(target, str) or not target.strip():
            raise RuntimeError(f"node {node_id} remapping target must be a non-empty string")
        source = source.strip()
        target = target.strip()
        if not source:
            raise RuntimeError(f"node {node_id} remapping source must be a non-empty string")
        if not target:
            raise RuntimeError(f"node {node_id} remapping target must be a non-empty string")
        remappings.append(RemappingSpec(source=source, target=target))
    return remappings


def _optional_env(
    value: dict[str, str] | None,
    node_id: str,
) -> dict[str, str]:
    if value is None:
        return {}
    env: dict[str, str] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str) or not key.strip():
            raise RuntimeError(f"node {node_id} env keys must be non-empty strings")
        if not isinstance(raw_value, str):
            raise RuntimeError(f"node {node_id} env '{key}' must be a string")
        env[key.strip()] = _resolve_config_refs(raw_value)
    return env


def _resolve_config_refs(value: str) -> str:
    def _replace_share(match_obj: re.Match[str]) -> str:
        return _get_package_share_directory(match_obj.group(1))

    def _replace_env(match_obj: re.Match[str]) -> str:
        variable_name = match_obj.group(1)
        variable_value = os.environ.get(variable_name)
        if variable_value is None or not variable_value.strip():
            raise RuntimeError(f"environment variable {variable_name} is required")
        return variable_value.strip()

    return _INLINE_ENV_RE.sub(_replace_env, _INLINE_SHARE_RE.sub(_replace_share, value))


def _expand_param_value(value: ParamValue) -> ParamValue:
    if isinstance(value, str):
        return _resolve_config_refs(value)
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return [_resolve_config_refs(item) for item in value]
        return value
    return value


def _get_package_share_directory(package_name: str) -> str:
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        candidate = Path(prefix) / "share" / package_name
        if candidate.is_dir():
            return str(candidate)
    raise RuntimeError(f"package share directory not found: {package_name}")


def _is_param_value(value: ConfigValue) -> TypeGuard[ParamValue]:
    if isinstance(value, bool):
        return True
    if isinstance(value, (str, int, float)):
        return True
    if not isinstance(value, list):
        return False
    if not value:
        return True
    if all(isinstance(item, bool) for item in value):
        return True
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return True
    if all(isinstance(item, float) for item in value):
        return True
    if all(isinstance(item, str) for item in value):
        return True
    return False

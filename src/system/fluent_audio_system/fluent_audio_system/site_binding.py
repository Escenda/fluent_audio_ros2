from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SiteBindingOverrides:
    fa_in_enabled: bool
    fa_out_enabled: bool
    fa_in_source_id: str
    fa_out_sink_id: str


def parse_bool_launch_arg_value(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be true or false")


def build_site_binding_overrides(
    *,
    fa_in_enabled: bool,
    fa_out_enabled: bool,
    fa_in_source_id: str,
    fa_out_sink_id: str,
) -> SiteBindingOverrides:
    source_id = fa_in_source_id.strip()
    sink_id = fa_out_sink_id.strip()
    if fa_in_enabled and not source_id:
        raise RuntimeError("fa_in_source_id is required when fa_in_enabled is true")
    if fa_out_enabled and not sink_id:
        raise RuntimeError("fa_out_sink_id is required when fa_out_enabled is true")
    return SiteBindingOverrides(
        fa_in_enabled=fa_in_enabled,
        fa_out_enabled=fa_out_enabled,
        fa_in_source_id=source_id,
        fa_out_sink_id=sink_id,
    )

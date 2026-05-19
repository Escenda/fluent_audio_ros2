from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from fluent_audio_system.config_schema import load_system_config
from fluent_audio_system.site_binding import (
    SiteBindingOverrides,
    build_site_binding_overrides,
    parse_bool_launch_arg_value,
)
from fluent_audio_system.site_binding_launch import (
    node_enabled_by_site_binding,
    node_launch_parameters,
)


def _required_bool_launch_arg(context, name: str) -> bool:
    value = LaunchConfiguration(name).perform(context)
    return parse_bool_launch_arg_value(name, value)


def _site_binding_overrides(context) -> SiteBindingOverrides:
    return build_site_binding_overrides(
        fa_in_enabled=_required_bool_launch_arg(context, "fa_in_enabled"),
        fa_out_enabled=_required_bool_launch_arg(context, "fa_out_enabled"),
        fa_in_source_id=LaunchConfiguration("fa_in_source_id").perform(context).strip(),
        fa_out_sink_id=LaunchConfiguration("fa_out_sink_id").perform(context).strip(),
    )


def _launch_setup(context):
    config_path = LaunchConfiguration("config").perform(context)
    config = load_system_config(config_path)
    overrides = _site_binding_overrides(context)

    actions = [LogInfo(msg=f"[fluent_audio_system] config={config_path}")]
    delay = 0.0

    for group_index, group in enumerate(config.groups):
        for node in group.nodes:
            if not node_enabled_by_site_binding(node, overrides):
                continue
            node_action = Node(
                package=node.package,
                executable=node.executable,
                name=node.node_name,
                namespace=node.namespace,
                output=node.output,
                parameters=node_launch_parameters(node, overrides),
                remappings=node.launch_remappings(),
                additional_env=node.env,
            )
            actions.append(
                LogInfo(
                    msg=(
                        "[fluent_audio_system] launch "
                        f"{node.package}:{node.executable} "
                        f"name={node.node_name} ns={node.namespace}"
                    )
                )
            )
            actions.append(TimerAction(period=delay, actions=[node_action]))
            delay += config.default_start_delay

        if group_index < len(config.groups) - 1:
            delay += config.inter_group_delay

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                description=(
                    "Explicit comma-separated fluent_audio_system YAML config path list."
                ),
            ),
            DeclareLaunchArgument(
                "fa_in_enabled",
                description="Explicit true/false site binding for profile-declared fa_in.",
            ),
            DeclareLaunchArgument(
                "fa_out_enabled",
                description="Explicit true/false site binding for profile-declared fa_out.",
            ),
            DeclareLaunchArgument(
                "fa_in_source_id",
                description="Explicit raw ALSA capture source id for fa_in.",
            ),
            DeclareLaunchArgument(
                "fa_out_sink_id",
                description="Explicit raw ALSA playback sink id for fa_out.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )

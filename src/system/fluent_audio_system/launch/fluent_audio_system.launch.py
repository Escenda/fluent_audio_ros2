from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from fluent_audio_system.config_schema import load_system_config


def _launch_setup(context):
    config_path = LaunchConfiguration("config").perform(context)
    config = load_system_config(config_path)

    actions = [LogInfo(msg=f"[fluent_audio_system] config={config_path}")]
    delay = 0.0

    for group_index, group in enumerate(config.groups):
        for node in group.nodes:
            node_action = Node(
                package=node.package,
                executable=node.executable,
                name=node.node_name,
                namespace=node.namespace,
                output=node.output,
                parameters=node.launch_parameters(),
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
                default_value="/config/fluent_audio_system.yaml",
                description="Absolute path to fluent_audio_system yaml.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )

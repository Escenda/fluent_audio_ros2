import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _setup(context):
    pkg_share = get_package_share_directory("fluent_audio_system")
    config_path = LaunchConfiguration("config").perform(context)
    fa_in_enabled = LaunchConfiguration("fa_in_enabled").perform(context)
    fa_out_enabled = LaunchConfiguration("fa_out_enabled").perform(context)
    fa_in_source_id = LaunchConfiguration("fa_in_source_id").perform(context)
    fa_out_sink_id = LaunchConfiguration("fa_out_sink_id").perform(context)
    launch_path = os.path.join(pkg_share, "launch", "fluent_audio_system.launch.py")
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_path),
            launch_arguments={
                "config": config_path,
                "fa_in_enabled": fa_in_enabled,
                "fa_out_enabled": fa_out_enabled,
                "fa_in_source_id": fa_in_source_id,
                "fa_out_sink_id": fa_out_sink_id,
            }.items(),
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value="/config/fluent_audio_system.yaml",
                description="Absolute path to fluent_audio_system yaml.",
            ),
            DeclareLaunchArgument(
                "fa_in_enabled",
                default_value="true",
                description="Enable profile-declared fa_in unless site profile disables it.",
            ),
            DeclareLaunchArgument(
                "fa_out_enabled",
                default_value="true",
                description="Enable profile-declared fa_out unless site profile disables it.",
            ),
            DeclareLaunchArgument(
                "fa_in_source_id",
                default_value="",
                description="Raw ALSA capture source id for fa_in.",
            ),
            DeclareLaunchArgument(
                "fa_out_sink_id",
                default_value="",
                description="Raw ALSA playback sink id for fa_out.",
            ),
            OpaqueFunction(function=_setup),
        ]
    )

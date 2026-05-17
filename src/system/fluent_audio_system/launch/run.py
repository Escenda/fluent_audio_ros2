import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _setup(context):
    pkg_share = get_package_share_directory("fluent_audio_system")
    config_path = LaunchConfiguration("config").perform(context)
    launch_path = os.path.join(pkg_share, "launch", "fluent_audio_system.launch.py")
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_path),
            launch_arguments={"config": config_path}.items(),
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
            OpaqueFunction(function=_setup),
        ]
    )

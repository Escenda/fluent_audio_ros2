from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("fa_kws"), "config", "default.yaml"
    )
    config = LaunchConfiguration("config", default=default_config)
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=default_config,
                description="Path to fa_kws config file.",
            ),
            Node(
                package="fa_kws",
                executable="fa_kws_node",
                name="fa_kws",
                output="screen",
                parameters=[config],
            ),
        ]
    )

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                description="Path to fa_onset parameter YAML.",
            ),
            Node(
                package="fa_onset",
                executable="fa_onset_node",
                name="fa_onset",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )

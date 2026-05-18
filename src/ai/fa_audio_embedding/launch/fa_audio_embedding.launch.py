from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fa_audio_embedding"), "config", "default.yaml"]
                ),
                description="Path to fa_audio_embedding parameter YAML.",
            ),
            Node(
                package="fa_audio_embedding",
                executable="fa_audio_embedding_node",
                name="fa_audio_embedding",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )

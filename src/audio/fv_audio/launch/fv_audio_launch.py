from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    node_name = LaunchConfiguration("node_name")
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "node_name",
            default_value="fv_audio_node",
            description="ノード名"
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value="config/default_audio.yaml",
            description="設定ファイルへのパス"
        ),
        Node(
            package="fv_audio",
            executable="fv_audio_node",
            name=node_name,
            output="screen",
            parameters=[config_file]
        )
    ])

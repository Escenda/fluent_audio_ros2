from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config_path = PathJoinSubstitution([
        FindPackageShare("fa_audio_window"),
        "config",
        "default.yaml",
    ])

    return LaunchDescription([
        Node(
            package="fa_audio_window",
            executable="fa_audio_window_node",
            name="fa_audio_window",
            parameters=[config_path],
            output="screen",
        )
    ])

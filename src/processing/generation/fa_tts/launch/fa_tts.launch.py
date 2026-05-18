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
            description="Required ROS node name.",
        ),
        DeclareLaunchArgument(
            "config_file",
            description="Required absolute path to a fa_tts parameter YAML file.",
        ),
        Node(
            package="fa_tts",
            executable="fa_tts_node",
            name=node_name,
            output="screen",
            parameters=[config_file],
        )
    ])

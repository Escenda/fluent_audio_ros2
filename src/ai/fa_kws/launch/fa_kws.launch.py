from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    node_name = LaunchConfiguration("node_name")
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "node_name",
                description="fa_kws node name.",
            ),
            DeclareLaunchArgument(
                "config_file",
                description="Path to fa_kws config file.",
            ),
            Node(
                package="fa_kws",
                executable="fa_kws_node",
                name=node_name,
                output="screen",
                parameters=[config_file],
            ),
        ]
    )

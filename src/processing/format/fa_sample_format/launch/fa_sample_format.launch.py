from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    node_name = LaunchConfiguration("node_name")
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "node_name",
            default_value="fa_sample_format",
            description="ノード名",
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("fa_sample_format"), "config", "default.yaml"]
            ),
            description="設定ファイルへのパス",
        ),
        Node(
            package="fa_sample_format",
            executable="fa_sample_format_node",
            name=node_name,
            output="screen",
            parameters=[config_file],
        ),
    ])

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
            default_value="fa_time_alignment",
            description="ノード名",
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("fa_time_alignment"), "config", "default.yaml"]
            ),
            description="設定ファイルへのパス",
        ),
        Node(
            package="fa_time_alignment",
            executable="fa_time_alignment_node",
            name=node_name,
            output="screen",
            parameters=[config_file],
        ),
    ])

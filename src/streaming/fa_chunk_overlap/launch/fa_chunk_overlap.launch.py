from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    node_name = LaunchConfiguration("node_name")
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "node_name",
                default_value="fa_chunk_overlap",
                description="ノード名",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("fa_chunk_overlap"), "config", "default.yaml"]
                ),
            ),
            Node(
                package="fa_chunk_overlap",
                executable="fa_chunk_overlap_node",
                name=node_name,
                output="screen",
                parameters=[config_file],
            ),
        ]
    )

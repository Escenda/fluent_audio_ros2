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
                default_value="fa_packet_loss_concealment_node",
                description="ノード名",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("fa_packet_loss_concealment"),
                        "config",
                        "default.yaml",
                    ]
                ),
            ),
            Node(
                package="fa_packet_loss_concealment",
                executable="fa_packet_loss_concealment_node",
                name=node_name,
                output="screen",
                parameters=[config_file],
            ),
        ]
    )

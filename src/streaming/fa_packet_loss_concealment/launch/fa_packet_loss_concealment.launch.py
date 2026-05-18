from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    node_name = LaunchConfiguration("node_name")
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "node_name",
                description="ノード名",
            ),
            DeclareLaunchArgument(
                "config_file",
                description="設定ファイルへのパス",
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

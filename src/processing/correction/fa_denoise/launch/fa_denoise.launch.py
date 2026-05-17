from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    node_name = LaunchConfiguration("node_name")
    config_file = LaunchConfiguration("config_file")
    model_1_path = LaunchConfiguration("model_1_path")
    model_2_path = LaunchConfiguration("model_2_path")

    return LaunchDescription([
        DeclareLaunchArgument(
            "node_name",
            default_value="fa_denoise",
            description="ノード名",
        ),
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("fa_denoise"), "config", "default.yaml"]
            ),
            description="設定ファイルへのパス",
        ),
        DeclareLaunchArgument(
            "model_1_path",
            default_value="",
            description="DTLN model 1 path",
        ),
        DeclareLaunchArgument(
            "model_2_path",
            default_value="",
            description="DTLN model 2 path",
        ),
        Node(
            package="fa_denoise",
            executable="fa_denoise_node",
            name=node_name,
            output="screen",
            parameters=[
                config_file,
                {
                    "dtln.model_1_path": model_1_path,
                    "dtln.model_2_path": model_2_path,
                },
            ],
        ),
    ])

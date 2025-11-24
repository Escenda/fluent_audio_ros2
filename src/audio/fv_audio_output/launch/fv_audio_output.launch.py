from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fv_audio_output',
            executable='fv_audio_output_node',
            name='fv_audio_output',
            output='screen',
            parameters=[{
                'audio.device_id': 'default',
                'audio.sample_rate': 48000,
                'audio.channels': 1,
                'audio.bit_depth': 16,
                'queue.max_frames': 8,
            }]
        )
    ])

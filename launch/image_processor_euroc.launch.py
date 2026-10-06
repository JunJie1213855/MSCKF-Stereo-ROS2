"""Launch only the image processor on EuRoC data.

Useful when the feature tracks are wanted on their own, or when the filter is
run separately. See msckf_vio_euroc.launch.py for the topic list.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare('msckf_vio'), 'config',
        'msckf_vio_euroc_params.yaml'])

    imu_topic = LaunchConfiguration('imu_topic')
    cam0_topic = LaunchConfiguration('cam0_topic')
    cam1_topic = LaunchConfiguration('cam1_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'imu_topic', default_value='/imu0',
            description='IMU topic.'),
        DeclareLaunchArgument(
            'cam0_topic', default_value='/cam0/image_raw',
            description='Left camera image topic.'),
        DeclareLaunchArgument(
            'cam1_topic', default_value='/cam1/image_raw',
            description='Right camera image topic.'),

        Node(
            package='msckf_vio',
            executable='image_processor_node',
            name='image_processor',
            output='screen',
            parameters=[params_file],
            remappings=[
                ('imu', imu_topic),
                ('cam0_image', cam0_topic),
                ('cam1_image', cam1_topic),
                ('features', '/image_processor/features'),
                ('tracking_info', '/image_processor/tracking_info'),
                ('debug_stereo_image', '/image_processor/debug_stereo_image'),
            ],
        ),
    ])

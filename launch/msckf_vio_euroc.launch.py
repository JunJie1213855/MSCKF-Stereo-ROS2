"""Launch the image processor and the MSCKF VIO filter on EuRoC data.

Topics consumed (EuRoC defaults, override with the launch arguments below):
  /imu0             sensor_msgs/Imu
  /cam0/image_raw   sensor_msgs/Image
  /cam1/image_raw   sensor_msgs/Image

Topics produced:
  /image_processor/features        msckf_vio/CameraMeasurement
  /image_processor/tracking_info   msckf_vio/TrackingInfo
  /odom                            nav_msgs/Odometry
  /feature_point_cloud             sensor_msgs/PointCloud2
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

    declare_imu = DeclareLaunchArgument(
        'imu_topic', default_value='/imu0',
        description='IMU topic for both nodes.')
    declare_cam0 = DeclareLaunchArgument(
        'cam0_topic', default_value='/cam0/image_raw',
        description='Left camera image topic.')
    declare_cam1 = DeclareLaunchArgument(
        'cam1_topic', default_value='/cam1/image_raw',
        description='Right camera image topic.')

    # The two nodes share one parameter file, keyed by node name, so they are
    # started in the root namespace with the names `image_processor` and `vio`.
    image_processor = Node(
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
    )

    vio = Node(
        package='msckf_vio',
        executable='msckf_vio_node',
        name='vio',
        output='screen',
        parameters=[params_file],
        remappings=[
            ('imu', imu_topic),
            ('features', '/image_processor/features'),
        ],
    )

    return LaunchDescription([
        declare_imu,
        declare_cam0,
        declare_cam1,
        image_processor,
        vio,
    ])

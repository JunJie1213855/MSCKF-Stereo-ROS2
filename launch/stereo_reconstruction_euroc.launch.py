"""EuRoC + MSCKF + selectable stereo dense point-cloud fusion.

Set start_vio:=false to attach to an already running MSCKF pipeline.
Camera calibration and body/world frame names come from the VIO parameter
file, so image rectification and odometry use the same calibration.
"""

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_nodes(context):
    pkg_share = FindPackageShare('msckf_vio')
    calibration_path = LaunchConfiguration('vio_params_file').perform(context)
    reconstruction_path = LaunchConfiguration('params_file').perform(context)
    with open(calibration_path, encoding='utf-8') as stream:
        source = yaml.safe_load(stream)
    image = source['image_processor']['ros__parameters']
    vio = source['vio']['ros__parameters']
    # Read only the shared geometric calibration, leaving tracker parameters out.
    calibration = {key: value for key, value in image.items()
                   if key.startswith(('cam0/', 'cam1/'))}
    calibration.update({
        'T_imu_body': vio['T_imu_body'],
        'fixed_frame_id': vio['fixed_frame_id'],
        'pose_child_frame_id': vio['child_frame_id'],
    })
    clock = {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}
    matcher_overrides = {}
    backend = LaunchConfiguration('backend').perform(context)
    engine_path = LaunchConfiguration('engine_path').perform(context)
    if backend:
        matcher_overrides['backend'] = backend
    if engine_path:
        matcher_overrides['igev.engine_path'] = engine_path
    vio_nodes = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            pkg_share, 'launch', 'msckf_vio_euroc.launch.py'])),
        condition=IfCondition(LaunchConfiguration('start_vio')),
        launch_arguments={
            'params_file': calibration_path,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'imu_topic': LaunchConfiguration('imu_topic'),
            'cam0_topic': LaunchConfiguration('cam0_topic'),
            'cam1_topic': LaunchConfiguration('cam1_topic'),
        }.items())
    reconstruction = Node(
        package='msckf_vio', executable='stereo_reconstruction_node',
        name='stereo_reconstruction', output='screen',
        parameters=[reconstruction_path, calibration, clock, matcher_overrides],
        remappings=[('cam0_image', LaunchConfiguration('cam0_topic')),
                    ('cam1_image', LaunchConfiguration('cam1_topic')),
                    ('odom', LaunchConfiguration('odom_topic'))])
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', PathJoinSubstitution([
            pkg_share, 'rviz', 'rviz_stereo_reconstruction.rviz'])],
        parameters=[clock], condition=IfCondition(LaunchConfiguration('rviz')))
    # The included launch also has a params_file argument. Keep its launch
    # configurations scoped so they cannot replace reconstruction settings.
    return [GroupAction(actions=[vio_nodes], scoped=True), reconstruction, rviz]


def generate_launch_description():
    share = FindPackageShare('msckf_vio')
    return LaunchDescription([
        DeclareLaunchArgument('start_vio', default_value='true',
                              description='Start MSCKF and its image processor.'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('backend', default_value='',
                              description='Override the YAML backend: sgbm or igev.'),
        DeclareLaunchArgument('engine_path', default_value='',
                              description='Override igev.engine_path with an absolute TensorRT engine path.'),
        DeclareLaunchArgument('imu_topic', default_value='/imu0'),
        DeclareLaunchArgument('cam0_topic', default_value='/cam0/image_raw'),
        DeclareLaunchArgument('cam1_topic', default_value='/cam1/image_raw'),
        DeclareLaunchArgument('odom_topic', default_value='/odom',
                              description='Odometry input when attaching to existing VIO.'),
        DeclareLaunchArgument('vio_params_file', default_value=PathJoinSubstitution([
            share, 'config', 'msckf_vio_euroc_params.yaml'])),
        DeclareLaunchArgument('params_file', default_value=PathJoinSubstitution([
            share, 'config', 'stereo_reconstruction_euroc_params.yaml'])),
        OpaqueFunction(function=launch_nodes),
    ])

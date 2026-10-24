"""Standalone stereo reconstruction; consumes images and timestamped odometry."""

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def launch_nodes(context):
    parameter_path = LaunchConfiguration('params_file').perform(context)
    vio_path = LaunchConfiguration('vio_params_file').perform(context)
    calibration_path = vio_path or LaunchConfiguration('calibration_file').perform(context)
    with open(calibration_path, encoding='utf-8') as stream:
        source = yaml.safe_load(stream)
    if vio_path:
        image = source['image_processor']['ros__parameters']
        vio = source['vio']['ros__parameters']
        calibration = {key: value for key, value in image.items()
                       if key.startswith(('cam0/', 'cam1/'))}
        calibration.update(T_imu_body=vio['T_imu_body'],
                           fixed_frame_id=vio['fixed_frame_id'],
                           pose_child_frame_id=vio['child_frame_id'])
    else:
        calibration = source['stereo_reconstruction']['ros__parameters']
    backend = LaunchConfiguration('backend').perform(context)
    engine_path = LaunchConfiguration('engine_path').perform(context)
    overrides = {}
    fusion_method = LaunchConfiguration('fusion_method').perform(context)
    if fusion_method:
        if fusion_method not in ('tsdf', 'voxel'):
            raise ValueError('fusion_method must be tsdf or voxel')
        overrides['fusion_method'] = fusion_method
    if backend:
        overrides['backend'] = backend
    if engine_path:
        with open(parameter_path, encoding='utf-8') as stream:
            configured = yaml.safe_load(stream)['stereo_reconstruction']['ros__parameters']
        effective_backend = backend or configured.get('backend', 'sgbm')
        if effective_backend not in ('igev', 's2m2'):
            raise ValueError('engine_path requires backend:=igev or backend:=s2m2')
        overrides[f'{effective_backend}.engine_path'] = engine_path
    clock = {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}
    return [
        Node(package='stereo_recon', executable='stereo_reconstruction_node',
             name='stereo_reconstruction', output='screen',
             parameters=[parameter_path, calibration, clock, overrides],
             remappings=[('cam0_image', LaunchConfiguration('cam0_topic')),
                         ('cam1_image', LaunchConfiguration('cam1_topic')),
                         ('odom', LaunchConfiguration('odom_topic'))]),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='screen',
             arguments=['-d', PathJoinSubstitution([
                 FindPackageShare('stereo_recon'), 'rviz', 'rviz_stereo_reconstruction.rviz'])],
             parameters=[clock], condition=IfCondition(LaunchConfiguration('rviz'))),
    ]


def generate_launch_description():
    share = FindPackageShare('stereo_recon')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=PathJoinSubstitution([
            share, 'config', 'stereo_reconstruction_euroc_params.yaml'])),
        DeclareLaunchArgument('calibration_file', default_value=PathJoinSubstitution([
            share, 'config', 'stereo_calibration_euroc.yaml'])),
        DeclareLaunchArgument('vio_params_file', default_value='',
                              description='Optional shared MSCKF YAML; overrides calibration_file.'),
        DeclareLaunchArgument('backend', default_value='',
                              description='Override YAML backend: sgbm, igev or s2m2.'),
        DeclareLaunchArgument('fusion_method', default_value='',
                              description='Override YAML fusion: tsdf or voxel.'),
        DeclareLaunchArgument('engine_path', default_value=''),
        DeclareLaunchArgument('cam0_topic', default_value='/cam0/image_raw'),
        DeclareLaunchArgument('cam1_topic', default_value='/cam1/image_raw'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=launch_nodes),
    ])

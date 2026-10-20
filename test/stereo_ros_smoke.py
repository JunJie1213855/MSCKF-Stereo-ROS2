#!/usr/bin/env python3
"""Manual end-to-end check against a EuRoC ROS 2 SQLite bag.

Source the built workspace first. Replays only camera/IMU messages in an
isolated localhost ROS domain, starts the real MSCKF and reconstruction nodes,
and checks depth, odometry, map fusion, save and clear. No ROS daemon required.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import threading
import time

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path, help='EuRoC .db3 file or directory containing one')
    parser.add_argument('--duration', type=float, default=15.0, help='Sensor seconds to replay')
    parser.add_argument('--rate', type=float, default=0.5)
    parser.add_argument('--domain-id', type=int, default=91)
    parser.add_argument('--backend', choices=['sgbm', 'igev', 's2m2'], default='sgbm')
    parser.add_argument('--engine', type=Path, help='Completed TensorRT engine for --backend igev/s2m2')
    parser.add_argument('--output', type=Path, default=Path('/tmp/msckf_stereo_euroc_validation'))
    args = parser.parse_args()
    if args.duration <= 2 or args.rate <= 0 or not 0 <= args.domain_id <= 232:
        parser.error('Use duration > 2, positive rate and a domain ID in [0,232]')
    if args.backend in ('igev', 's2m2') and (not args.engine or not args.engine.is_file() or args.engine.stat().st_size == 0):
        parser.error('TensorRT backends require a nonempty --engine file')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['ROS_DOMAIN_ID'] = str(args.domain_id)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    os.environ['ROS_LOG_DIR'] = str(args.output / 'ros_logs')
    # Import after selecting the domain, before creating any ROS context.
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.serialization import deserialize_message
    from ament_index_python.packages import get_package_share_directory
    from sensor_msgs.msg import Image, Imu, PointCloud2
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from std_srvs.srv import Trigger

    share = Path(get_package_share_directory('msckf_vio'))
    config = yaml.safe_load((share / 'config/stereo_reconstruction_euroc_params.yaml').read_text())
    config['stereo_reconstruction']['ros__parameters']['backend'] = args.backend
    if args.engine:
        config['stereo_reconstruction']['ros__parameters'][args.backend]['engine_path'] = str(args.engine.resolve())
    output_map = args.output / 'stereo_map.pcd'
    config['stereo_reconstruction']['ros__parameters']['save_map_path'] = str(output_map)
    parameter_path = args.output / 'parameters.yaml'
    parameter_path.write_text(yaml.safe_dump(config))
    log = (args.output / 'launch.log').open('w')
    process = subprocess.Popen([
        'ros2', 'launch', 'msckf_vio', 'stereo_reconstruction_euroc.launch.py',
        'rviz:=false', 'params_file:=' + str(parameter_path)],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    node = None
    executor = None
    thread = None
    connection = None
    metrics = {'backend': args.backend, 'depth_frames': 0, 'odom_frames': 0, 'map_updates': 0,
               'max_map_points': 0, 'last_map_points': 0, 'world_clouds': 0,
               'last_depth_valid_fraction': 0.0}
    validation_errors = []
    try:
        rclpy.init()
        node = rclpy.create_node('stereo_euroc_smoke_monitor')
        sensor_qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        def depth(message):
            metrics['depth_frames'] += 1
            if message.encoding != '32FC1' or message.header.frame_id != 'cam0_rect_optical_frame':
                validation_errors.append('Invalid depth encoding or frame')
            values = np.frombuffer(message.data, dtype=np.float32)
            valid = values[np.isfinite(values)]
            metrics['last_depth_valid_fraction'] = float(valid.size / values.size)
            if valid.size and (valid.min() < 0.3 - 1e-6 or valid.max() > 10.0 + 1e-6):
                validation_errors.append('Depth outside configured limits')

        def odometry(message):
            metrics['odom_frames'] += 1
            if message.header.frame_id != 'world' or message.child_frame_id != 'odom':
                validation_errors.append('Unexpected VIO odometry frame')

        def map_update(message):
            metrics['map_updates'] += 1
            count = message.width * message.height
            metrics['last_map_points'] = count
            metrics['max_map_points'] = max(metrics['max_map_points'], count)
            if message.header.frame_id != 'world':
                validation_errors.append('Map is not in world frame')

        def world_cloud(message):
            metrics['world_clouds'] += 1
            if message.header.frame_id != 'world' or not message.is_dense:
                validation_errors.append('Invalid current world cloud')

        subscriptions = [
            node.create_subscription(Image, '/stereo/depth', depth, sensor_qos),
            node.create_subscription(Odometry, '/odom', odometry, sensor_qos),
            node.create_subscription(PointCloud2, '/stereo/map', map_update, map_qos),
            node.create_subscription(PointCloud2, '/stereo/points_world', world_cloud, sensor_qos)]
        publishers = {
            '/cam0/image_raw': node.create_publisher(Image, '/cam0/image_raw', sensor_qos),
            '/cam1/image_raw': node.create_publisher(Image, '/cam1/image_raw', sensor_qos),
            '/imu0': node.create_publisher(Imu, '/imu0', sensor_qos)}
        clock_pub = node.create_publisher(Clock, '/clock', 10)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Launch exited; inspect launch.log')
            if node.count_subscribers('/cam0/image_raw') >= 2 and node.count_subscribers('/imu0') >= 2:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError('ROS subscriptions did not become ready')
        time.sleep(0.5)
        database = args.bag
        if database.is_dir():
            databases = sorted(database.glob('*.db3'))
            if len(databases) != 1:
                raise RuntimeError('Choose a single .db3 file for this smoke check')
            database = databases[0]
        connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
        topics = {row[0]: row[1] for row in connection.execute('SELECT id,name FROM topics')
                  if row[1] in publishers}
        if len(topics) != 3:
            raise RuntimeError('Bag must contain /imu0 and /cam{0,1}/image_raw')
        ids = ','.join(str(topic_id) for topic_id in topics)
        first = connection.execute(f'SELECT MIN(timestamp) FROM messages WHERE topic_id IN ({ids})').fetchone()[0]
        end = first + int(args.duration * 1e9)
        records = connection.execute(
            f'SELECT topic_id,timestamp,data FROM messages WHERE topic_id IN ({ids}) AND timestamp<=? ORDER BY timestamp,id',
            (end,))
        started = time.monotonic()
        sent = 0
        for topic_id, stamp, data in records:
            target = started + (stamp - first) * 1e-9 / args.rate
            time.sleep(max(0.0, target - time.monotonic()))
            clock = Clock()
            clock.clock.sec, clock.clock.nanosec = divmod(stamp, 10**9)
            clock_pub.publish(clock)
            topic = topics[topic_id]
            publishers[topic].publish(deserialize_message(data, Imu if topic == '/imu0' else Image))
            sent += 1
        metrics['input_messages'] = sent
        time.sleep(2)
        if metrics['depth_frames'] < 10 or metrics['odom_frames'] < 10:
            raise RuntimeError('Too few depth or odometry frames: ' + str(metrics))
        if metrics['max_map_points'] < 1000 or metrics['world_clouds'] < 5:
            raise RuntimeError('Point-cloud fusion did not populate the map: ' + str(metrics))
        if metrics['last_depth_valid_fraction'] < 0.1:
            raise RuntimeError('Insufficient valid depth')

        def call_service(name):
            client = node.create_client(Trigger, name)
            if not client.wait_for_service(timeout_sec=5):
                raise RuntimeError('Service unavailable: ' + name)
            future = client.call_async(Trigger.Request())
            limit = time.monotonic() + 10
            while not future.done() and time.monotonic() < limit:
                time.sleep(.05)
            if not future.done():
                raise RuntimeError('Service timeout: ' + name)
            result = future.result()
            node.destroy_client(client)
            if not result.success:
                raise RuntimeError(result.message)
            return result.message

        metrics['save_result'] = call_service('/stereo/save_map')
        if not output_map.exists() or output_map.stat().st_size < 10000:
            raise RuntimeError('PCD output is missing or too small')
        metrics['saved_bytes'] = output_map.stat().st_size
        metrics['clear_result'] = call_service('/stereo/clear_map')
        deadline = time.monotonic() + 3
        while metrics['last_map_points'] != 0 and time.monotonic() < deadline:
            time.sleep(.05)
        if metrics['last_map_points'] != 0:
            raise RuntimeError('Clear service did not publish an empty map')
        if validation_errors:
            raise RuntimeError(str(validation_errors[:5]))
        metrics['success'] = True
        (args.output / 'results.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics, indent=2), flush=True)
        print('Artifacts: ' + str(args.output), flush=True)
    except Exception as error:
        metrics['success'] = False
        metrics['error'] = str(error)
        (args.output / 'results.json').write_text(json.dumps(metrics, indent=2) + '\n')
        raise
    finally:
        if connection:
            connection.close()
        if executor:
            executor.shutdown(timeout_sec=3)
        if thread:
            thread.join(timeout=3)
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if process.poll() is None:
            # Let launch forward SIGINT once to its children; signaling the
            # whole group would also deliver a second SIGINT via launch.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        log.close()


if __name__ == '__main__':
    main()

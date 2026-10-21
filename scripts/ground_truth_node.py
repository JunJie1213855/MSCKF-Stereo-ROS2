#!/usr/bin/env python3
"""Replay ground truth at odometry timestamps with fixed first-pose alignment."""

from bisect import bisect_left
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path

import numpy as np


def unit_quaternion(values):
    q = np.asarray(values, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Expected a finite, nonzero quaternion [x, y, z, w]')
    return q / np.linalg.norm(q)


def pose_matrix(position, quaternion):
    p = np.asarray(position, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('Expected a finite XYZ position')
    x, y, z, w = unit_quaternion(quaternion)
    pose = np.eye(4)
    pose[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ]
    pose[:3, 3] = p
    return pose


def matrix_quaternion(rotation):
    """Convert an SO(3) matrix to a ROS-order quaternion."""
    r = np.asarray(rotation)
    if np.trace(r) > 0:
        s = 2 * math.sqrt(1 + np.trace(r))
        q = [(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
             (r[1, 0]-r[0, 1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i+1) % 3, (i+2) % 3
        s = 2 * math.sqrt(max(0.0, 1 + r[i, i]-r[j, j]-r[k, k]))
        q = np.zeros(4)
        q[i], q[j], q[k], q[3] = (
            s/4, (r[j, i]+r[i, j])/s, (r[k, i]+r[i, k])/s,
            (r[k, j]-r[j, k])/s)
    return unit_quaternion(q)


def slerp(first, second, fraction):
    a, b = unit_quaternion(first), unit_quaternion(second)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        return unit_quaternion((1-fraction)*a + fraction*b)
    angle = math.acos(dot)
    return (math.sin((1-fraction)*angle)*a + math.sin(fraction*angle)*b) / math.sin(angle)


class GroundTruthTrajectory:
    """EuRoC CSV (ns, XYZ, wxyz) or TUM text (s, XYZ, xyzw)."""

    def __init__(self, filename, max_gap_sec=0.05):
        if not math.isfinite(max_gap_sec) or max_gap_sec <= 0:
            raise ValueError('max_gap_sec must be finite and positive')
        self.max_gap_ns = round(max_gap_sec * 1e9)
        self.stamps, self.positions, self.quaternions = [], [], []
        with Path(filename).expanduser().open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                is_csv = ',' in line
                fields = line.split(',') if is_csv else line.split()
                try:
                    if len(fields) < 8 or (not is_csv and len(fields) != 8):
                        raise ValueError('Expected EuRoC CSV or 8-column TUM data')
                    timestamp = Decimal(fields[0])
                    if not timestamp.is_finite() or timestamp < 0:
                        raise ValueError('Invalid timestamp')
                    stamp = int(timestamp if is_csv else timestamp * Decimal(10**9))
                    position = np.array(fields[1:4], dtype=float)
                    quaternion = fields[5:8] + fields[4:5] if is_csv else fields[4:8]
                    quaternion = unit_quaternion(quaternion)
                    if not np.isfinite(position).all():
                        raise ValueError('Non-finite position')
                    if self.stamps and stamp <= self.stamps[-1]:
                        raise ValueError('Timestamps must be strictly increasing')
                except (ValueError, InvalidOperation) as error:
                    raise ValueError(f'{filename}:{line_number}: {error}') from error
                self.stamps.append(stamp)
                self.positions.append(position)
                self.quaternions.append(quaternion)
        if len(self.stamps) < 2:
            raise ValueError('Ground truth must contain at least two poses')

    def sample(self, stamp_ns):
        index = bisect_left(self.stamps, stamp_ns)
        if index < len(self.stamps) and self.stamps[index] == stamp_ns:
            return pose_matrix(self.positions[index], self.quaternions[index])
        if index == 0 or index == len(self.stamps):
            return None
        before, after = self.stamps[index-1], self.stamps[index]
        if after-before > self.max_gap_ns:
            return None
        fraction = (stamp_ns-before) / (after-before)
        position = (1-fraction)*self.positions[index-1] + fraction*self.positions[index]
        quaternion = slerp(self.quaternions[index-1], self.quaternions[index], fraction)
        return pose_matrix(position, quaternion)


@dataclass
class Comparison:
    pose: np.ndarray
    position_error: float
    orientation_error_deg: float
    position_rmse: float


class TrajectoryComparison:
    def __init__(self, trajectory, imu_from_body=None):
        self.trajectory = trajectory
        self.imu_from_body = np.eye(4) if imu_from_body is None else np.asarray(imu_from_body)
        transform = self.imu_from_body
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-3)):
            raise ValueError('T_imu_body must be a finite rigid transform')
        self.generation = 0
        self.reset()

    def reset(self):
        self.alignment = None
        self.last_stamp = None
        self.count = 0
        self.squared_error = 0.0
        self.generation += 1

    def update(self, stamp_ns, estimated_pose):
        if self.last_stamp is not None:
            if stamp_ns < self.last_stamp:
                self.reset()
            elif stamp_ns == self.last_stamp:
                return None
        self.last_stamp = stamp_ns
        truth = self.trajectory.sample(stamp_ns)
        if truth is None:
            return None
        # EuRoC state ground truth describes the IMU; compare the same body
        # origin as /odom, including a possible rotational/translational lever arm.
        truth = truth @ self.imu_from_body
        if self.alignment is None:
            self.alignment = estimated_pose @ np.linalg.inv(truth)
        aligned = self.alignment @ truth
        position_error = float(np.linalg.norm(estimated_pose[:3, 3]-aligned[:3, 3]))
        relative_rotation = aligned[:3, :3].T @ estimated_pose[:3, :3]
        angle = math.degrees(math.acos(float(np.clip((np.trace(relative_rotation)-1)/2, -1, 1))))
        self.count += 1
        self.squared_error += position_error**2
        return Comparison(aligned, position_error, angle,
                          math.sqrt(self.squared_error/self.count))


def main(args=None):
    # Keep the geometry and parser importable without a running ROS environment.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry, Path as PathMessage
    from std_msgs.msg import Float64

    class GroundTruthNode(Node):
        def __init__(self):
            super().__init__('ground_truth')
            filename = self.declare_parameter('ground_truth', '').value
            max_gap = self.declare_parameter('max_gap_sec', 0.05).value
            transform = self.declare_parameter('T_imu_body', np.eye(4).ravel().tolist()).value
            if len(transform) != 16:
                raise ValueError('T_imu_body needs 16 row-major values')
            trajectory = GroundTruthTrajectory(filename, max_gap)
            self.comparison = TrajectoryComparison(trajectory, np.array(transform).reshape(4, 4))
            self.path = PathMessage()
            self.frames = None
            self.path_pub = self.create_publisher(
                PathMessage, 'ground_truth/path',
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.pose_pub = self.create_publisher(PoseStamped, 'ground_truth/pose', 10)
            self.error_pubs = [self.create_publisher(Float64, 'ground_truth/' + name, 10)
                               for name in ('position_error', 'orientation_error_deg', 'position_rmse')]
            self.odom_sub = self.create_subscription(Odometry, 'odom', self.on_odometry, 100)
            self.path_sub = self.create_subscription(
                PathMessage, 'path', self.on_path,
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.get_logger().info(
                f'Loaded {len(trajectory.stamps)} ground-truth poses from {filename}; '
                'replay follows odometry timestamps, alignment is fixed at the first match')

        def clear(self, header):
            self.comparison.reset()
            self.path = PathMessage()
            self.path.header = header
            self.path_pub.publish(self.path)

        def on_path(self, message):
            # The VIO publishes an empty path when its world origin is reset.
            if not message.poses:
                self.clear(message.header)

        def on_odometry(self, message):
            frames = (message.header.frame_id, message.child_frame_id)
            if self.frames is not None and frames != self.frames:
                self.clear(message.header)
            self.frames = frames
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            p, q = message.pose.pose.position, message.pose.pose.orientation
            try:
                estimate = pose_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
            except ValueError as error:
                self.get_logger().warning(str(error), throttle_duration_sec=5.0)
                return
            generation = self.comparison.generation
            result = self.comparison.update(stamp, estimate)
            if generation != self.comparison.generation:
                self.path = PathMessage()
                self.path.header = message.header
                self.path_pub.publish(self.path)
            if result is None:
                self.get_logger().warning(
                    f'No new ground-truth match at {stamp/1e9:.9f}s '
                    '(outside time range, excessive gap or repeated timestamp)',
                    throttle_duration_sec=5.0)
                return
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = result.pose[:3, 3].tolist()
            quaternion = matrix_quaternion(result.pose[:3, :3]).tolist()
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quaternion
            self.pose_pub.publish(pose)
            self.path.header = message.header
            self.path.poses.append(pose)
            self.path_pub.publish(self.path)
            values = (result.position_error, result.orientation_error_deg, result.position_rmse)
            for publisher, value in zip(self.error_pubs, values):
                publisher.publish(Float64(data=value))
            self.get_logger().info(
                f't={stamp/1e9:.6f}: position error={result.position_error:.3f} m, '
                f'orientation error={result.orientation_error_deg:.2f} deg, '
                f'position RMSE={result.position_rmse:.3f} m (n={self.comparison.count})',
                throttle_duration_sec=5.0)

    rclpy.init(args=args)
    node = None
    try:
        node = GroundTruthNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

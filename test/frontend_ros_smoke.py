"""Check real frontend nodes with known stereo shifts and moving textures.

Source the built workspace; run with /usr/bin/python3 -s to use ROS's NumPy ABI.
Uses a separate ROS domain and synthetic images, without an IMU/VIO backend.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import yaml


def run_case(executable, detector, distribution, directory):
    width, height, focal = 752, 480, 500.0
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    stereo = identity.copy()
    stereo[3] = -0.1
    params = {
        "cam0/distortion_model": "radtan",
        "cam1/distortion_model": "radtan",
        "cam0/resolution": [width, height],
        "cam1/resolution": [width, height],
        "cam0/intrinsics": [focal, focal, width / 2, height / 2],
        "cam1/intrinsics": [focal, focal, width / 2, height / 2],
        "cam0/distortion_coeffs": [0.0] * 4,
        "cam1/distortion_coeffs": [0.0] * 4,
        "cam0/T_cam_imu": identity,
        "cam1/T_cn_cnm1": stereo,
        "feature_detector": detector, "feature_distribution": distribution,
        "max_features": 160, "min_feature_distance": 20, "feature_border": 8,
        "detection_candidates": 2000, "orb_levels": 4, "orb_scale_factor": 1.2,
        "grid_row": 4, "grid_col": 5,
        "grid_min_feature_num": 8, "grid_max_feature_num": 12,
        "pyramid_levels": 5, "patch_size": 15, "fast_threshold": 10,
        "max_iteration": 30, "track_precision": 0.01,
        "ransac_threshold": 3.0, "stereo_threshold": 5.0,
    }
    case = f"{detector}_{distribution}"
    config = directory / f"{case}.yaml"
    config.write_text(yaml.safe_dump({"image_processor": {"ros__parameters": params}}))
    node = rclpy.create_node(f"frontend_smoke_{case}")
    sensor_qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
    left_pub = node.create_publisher(Image, "cam0_image", sensor_qos)
    right_pub = node.create_publisher(Image, "cam1_image", sensor_qos)
    imu_pub = node.create_publisher(Imu, "imu", sensor_qos)
    received = {}
    def on_features(msg):
        received[msg.header.stamp.nanosec] = msg

    subscription = node.create_subscription(
        CameraMeasurement, "features", on_features, 10)
    log = (directory / f"{case}.log").open("w")
    process = subprocess.Popen(
        [str(executable), "--ros-args", "--params-file", str(config),
         "--log-level", "image_processor:=debug"],
        stdout=log, stderr=subprocess.STDOUT)
    frames = []
    try:
        deadline = time.monotonic() + 15
        while (left_pub.get_subscription_count() == 0 or
               right_pub.get_subscription_count() == 0 or
               node.count_publishers("features") == 0):
            if process.poll() is not None:
                raise RuntimeError(f"{case} exited; see {log.name}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"{case}: no camera subscriptions")
            rclpy.spin_once(node, timeout_sec=0.05)
        # Sensor QoS is volatile: allow both DDS endpoints to finish matching
        # before sending the first one-shot image pair.
        settle_until = time.monotonic() + 1.0
        while time.monotonic() < settle_until:
            rclpy.spin_once(node, timeout_sec=0.05)
        rng = np.random.default_rng(42)
        base = rng.integers(0, 256, (height, width), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), 0.6)
        previous_ids = None
        for frame in range(10):
            transform = np.float32([[1, 0, frame * 0.8], [0, 1, frame * 0.3]])
            left = cv2.warpAffine(base, transform, (width, height))
            if frame == 8:
                left[:] = 0
            right = cv2.warpAffine(left, np.float32([[1, 0, -8], [0, 1, 0]]),
                                   (width, height))
            stamp_ns = (frame + 1) * 50_000_000
            imu = Imu()
            imu.header.stamp.sec = 100
            imu.header.stamp.nanosec = stamp_ns
            imu_pub.publish(imu)
            published = []
            for publisher, pixels in ((left_pub, left), (right_pub, right)):
                message = Image()
                message.header.stamp = imu.header.stamp
                message.height, message.width = height, width
                message.encoding, message.step = "mono8", width
                message.data = pixels.tobytes()
                publisher.publish(message)
                published.append((publisher, message))
                rclpy.spin_once(node, timeout_sec=0.05)
            deadline = time.monotonic() + 10
            retry_at = time.monotonic() + 0.5
            while stamp_ns not in received and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{case} exited; see {log.name}")
                rclpy.spin_once(node, timeout_sec=0.02)
                if frame == 0 and time.monotonic() >= retry_at:
                    for publisher, message in published:
                        publisher.publish(message)
                    retry_at = time.monotonic() + 0.5
            assert stamp_ns in received, f"{case}: missing frame {frame}; received keys={list(received)}"
            features = received.pop(stamp_ns).features
            count = len(features)
            ids = {feature.id for feature in features}
            assert len(ids) == count, "Duplicate feature IDs"
            if frame == 8:
                assert count == 0, f"Blank image retained {count} features"
                previous_ids = None
                continue
            assert 100 <= count <= (160 if distribution == "mask" else 240), (case, frame, count)
            points = np.array([(f.u0 * focal + width / 2,
                                f.v0 * focal + height / 2) for f in features])
            assert np.all(points >= 8)
            assert np.all(points[:, 0] < width - 8)
            assert np.all(points[:, 1] < height - 8)
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            min_distance = float(distances.min())
            if distribution == "mask":
                assert min_distance >= 20 - 1e-3, (case, min_distance)
            disparity_error = np.abs(np.array([(f.u0 - f.u1) * focal for f in features]) - 8)
            assert np.median(disparity_error) < 0.25
            cells = {(min(3, int(y * 4 / height)), min(4, int(x * 5 / width)))
                     for x, y in points}
            assert len(cells) >= 18, (case, "coverage", len(cells))
            retention = len(ids & previous_ids) / len(previous_ids) if previous_ids else None
            if retention is not None:
                assert retention >= 0.7, (case, "retention", retention)
            frames.append({"frame": frame, "count": count,
                           "min_distance": min_distance, "covered_cells": len(cells),
                           "median_disparity_error": float(np.median(disparity_error)),
                           "retention": retention})
            previous_ids = ids
        print(f"PASS {case}: {frames[0]['count']} initial points, "
              f"spacing {frames[0]['min_distance']:.2f}px, blank-frame recovery passed")
        return frames
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        node.destroy_node()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/tmp/msckf_validation/frontend"))
    parser.add_argument("--domain-id", default="173")
    args = parser.parse_args()
    os.environ["ROS_DOMAIN_ID"] = args.domain_id
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
    # Apply the selected domain/transport before loading ROS libraries.
    global rclpy, QoSProfile, ReliabilityPolicy, Image, Imu, CameraMeasurement
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, Imu
    from msckf_vio.msg import CameraMeasurement
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    try:
        results = {}
        for detector, distribution in (("orb", "mask"), ("fast", "mask"),
                                       ("orb", "grid"), ("fast", "grid")):
            results[f"{detector}_{distribution}"] = run_case(
                args.executable, detector, distribution, args.output)
        (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

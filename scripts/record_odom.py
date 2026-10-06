#!/usr/bin/env python3
"""Record /odom into a TUM-format trajectory file for evo.

Runs until Ctrl-C (or for an optional number of seconds). Run it in the same
ROS_DOMAIN_ID as the estimator.

    record_odom.py out.txt          # until interrupted
    record_odom.py out.txt 140      # for 140 s of wall time
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry

out_path = sys.argv[1] if len(sys.argv) > 1 else "odom_tum.txt"
stop_after = float(sys.argv[2]) if len(sys.argv) > 2 else None

rclpy.init()
node = Node('odom_recorder')

# The vio node publishes /odom with the default (reliable) QoS.
qos = QoSProfile(depth=50,
                 reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST)

count = 0
first_stamp = None
fh = open(out_path, 'w')


def cb(msg):
    global count, first_stamp
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if first_stamp is None:
        first_stamp = stamp
    if stop_after is not None and stamp - first_stamp > stop_after:
        return
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    # TUM: timestamp[s] x y z qx qy qz qw
    fh.write('%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n'
             % (stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
    count += 1
    if count % 100 == 0:
        fh.flush()


node.create_subscription(Odometry, '/odom', cb, qos)
print('recording /odom to %s - Ctrl-C to stop' % out_path)

try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    fh.flush()
    fh.close()
    print('\nrecorded %d poses -> %s' % (count, out_path))
    node.destroy_node()
    rclpy.shutdown()

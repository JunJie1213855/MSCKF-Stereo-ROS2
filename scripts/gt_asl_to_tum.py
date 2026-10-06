#!/usr/bin/env python3
"""Convert a EuRoC / ASL ground-truth csv to evo's TUM format.

Input columns : timestamp[ns], px, py, pz, qw, qx, qy, qz, ...
Output columns: timestamp[s],  px, py, pz, qx, qy, qz, qw

    gt_asl_to_tum.py mav0/state_groundtruth_estimate0/data.csv gt_tum.txt
"""
import csv
import sys

src = sys.argv[1]
dst = sys.argv[2]

n = 0
with open(src) as f, open(dst, 'w') as o:
    for row in csv.reader(f):
        if not row or row[0].startswith('#'):
            continue
        try:
            t = float(row[0]) * 1e-9
            px, py, pz = (float(v) for v in row[1:4])
            qw, qx, qy, qz = (float(v) for v in row[4:8])
        except (ValueError, IndexError):
            continue
        o.write('%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n'
                % (t, px, py, pz, qx, qy, qz, qw))
        n += 1

print('wrote %d poses -> %s' % (n, dst))

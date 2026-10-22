# ORB / FAST detection and spatial feature selection

The image processor detects left-image keypoints and keeps stereo feature IDs
using the existing IMU-assisted pyramidal LK tracker. ORB is used through
`detect()` only: descriptors are not computed or matched. Stereo epipolar checks
and temporal two-point RANSAC remain the geometric acceptance stages.

## Configuration

All three VIO YAML files explicitly enable the `mask` distribution. FAST
remains the default detector to preserve current behavior; ORB detection can
be selected with `feature_detector: orb`. The configured maximum counts are
400 for EuRoC, 80 for no-extrinsics, and 60 for FLA, preserving each profile's
previous per-cell upper-bound total.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `feature_detector` | `fast` | `orb` or `fast` |
| `feature_distribution` | `mask` | `mask` or `grid` |
| `max_features` | `160` | Maximum active tracks in mask mode |
| `min_feature_distance` | `20` | Mask radius / minimum accepted left-image spacing, pixels |
| `feature_border` | `8` | Exclude borders in both images, pixels |
| `detection_candidates` | `2000` | Candidate budget before stereo and spatial acceptance |
| `orb_levels` | `4` | ORB detection pyramid levels, separate from LK `pyramid_levels` |
| `orb_scale_factor` | `1.2` | ORB detection pyramid scale |
| `fast_threshold` | profile value | FAST threshold, including the FAST detector inside ORB |

ORB uses Harris scoring, a 31-pixel patch and 31-pixel edge threshold. Its coarse
levels therefore have a larger effective border. Candidate keypoints from all
levels are expressed in original-image pixels before spatial selection.

## Mask selection

The selection order follows the lifetime-first mask idea in
[VINS-Mono's feature tracker](https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/feature_tracker/src/feature_tracker.cpp):

1. Track existing features and reject geometric outliers.
2. Sort valid tracks by lifetime, then response and ID for deterministic ties.
3. Accept older tracks first and mask a disk around each accepted position.
4. Detect replacement candidates in the remaining area, ordered by response.
5. Stereo-match candidates in batches of at most 128. Accept only stereo inliers
   outside the updated mask, immediately masking each accepted new feature.

Both old/new and new/new spacing are enforced across grid boundaries. A
floating-point distance check supplements the pixel mask for subpixel LK
positions. Failed stereo candidates do not consume the spatial budget. If the
image has too little texture, or masking removes too much area, the result may
contain fewer than `max_features` points.

`grid` mode retains cell minimum/maximum quotas for comparison. Grid assignment
now scales by the full image size, avoiding overflow in the last columns of
images such as EuRoC's 752-pixel width. Both modes reject nonfinite and
border-adjacent tracked points. Grid mode does not enforce global spacing.

This implements mask-based selection; it does not implement ORB-SLAM's quadtree.
Changing the detector alone does not establish better trajectory accuracy.

## Verification

After building with `BUILD_TESTING=ON`:

```bash
source /opt/ros/humble/setup.bash
source /home/ros/rosws/msckf_stereo_ws/install/setup.bash
source /home/ros/rosws/msckf_stereo_ws/src/MSCKF-Stereo-ROS2/scripts/setup_wsl_env.sh
cd /home/ros/rosws/msckf_stereo_ws
colcon test --packages-select msckf_vio --ctest-args -R test_feature_mask
/usr/bin/python3 -s src/MSCKF-Stereo-ROS2/test/frontend_ros_smoke.py \
  --executable install/msckf_vio/lib/msckf_vio/image_processor_node
```

The ROS smoke check runs all four detector/distribution combinations on moving
synthetic stereo textures. It checks feature budgets, image borders, grid
coverage, known disparity, retained feature IDs, mask-mode spacing and recovery
after a blank frame. Logs and per-frame metrics are saved under
`/tmp/msckf_validation/frontend`. This is a functional check, not a trajectory
accuracy or real-time performance benchmark. `-s` avoids user-installed NumPy
2.x conflicting with Ubuntu's NumPy-1.x-built OpenCV/cv_bridge.

On the local RTX 4060 Ti WSL environment, all four combinations passed. Initial
mask-mode minimum spacing was 20.10 pixels (ORB) and 20.22 pixels (FAST), versus
0.60 pixels (ORB/grid) and 5.10 pixels (FAST/grid). All initial frames had 160
points; temporal ID retention, known-disparity checks and recovery passed.
These synthetic results establish spatial behavior, not better VIO accuracy.

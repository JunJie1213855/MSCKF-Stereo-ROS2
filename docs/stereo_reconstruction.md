# EuRoC + MSCKF stereo point-cloud fusion

The `stereo_reconstruction_node` runs independently of the MSCKF estimator. Its
first backend is OpenCV SGBM, with a virtual `StereoMatcher` interface for later
TensorRT implementations. The added node needs no CUDA, TensorRT or PCL APIs;
the existing package dependencies still apply.

## Build and run

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select msckf_vio --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
export ROS_DOMAIN_ID=42
ros2 launch msckf_vio stereo_reconstruction_euroc.launch.py
```

In a second terminal with the same environment and domain:

```bash
ros2 bag play /path/to/V2_01_easy_ros2 --clock --rate 0.5 \
  --topics /imu0 /cam0/image_raw /cam1/image_raw
```

The launch starts the existing image processor, MSCKF, dense stereo node and
RViz. It defaults to `use_sim_time:=true`; bag playback must publish `/clock`.
The IMU must start stationary, as in the normal MSCKF workflow. Mapping begins
when MSCKF has initialized and matching odometry becomes available.

To attach to an existing EuRoC MSCKF run:

```bash
ros2 launch msckf_vio stereo_reconstruction_euroc.launch.py \
  start_vio:=false rviz:=false
```

Launch arguments include `cam0_topic`, `cam1_topic`, `imu_topic`, `odom_topic`,
`vio_params_file`, `params_file`, `start_vio`, `rviz` and `use_sim_time`.
`odom_topic` selects the reconstruction input; when starting VIO here, leave it
at `/odom`, which is what the existing VIO launch publishes.

The launch reads calibration from `image_processor.ros__parameters` and body
extrinsics/frame names from `vio.ros__parameters` in the **same** VIO YAML file
passed to MSCKF. The reconstruction YAML contains only matcher/mapping settings.
When attaching to an existing run, choose its actual VIO parameter file.

## Outputs

| Topic | Type | Coordinates / units |
|---|---|---|
| `/stereo/left/image_rect` | `sensor_msgs/Image` | Rectified mono8 or BGR8 |
| `/stereo/right/image_rect` | `sensor_msgs/Image` | Rectified mono8 or BGR8 |
| `/stereo/left/camera_info` | `sensor_msgs/CameraInfo` | Rectified left intrinsics |
| `/stereo/right/camera_info` | `sensor_msgs/CameraInfo` | Rectified right intrinsics and stereo projection |
| `/stereo/disparity` | `sensor_msgs/Image`, `32FC1` | Pixel disparity, NaN invalid |
| `/stereo/depth` | `sensor_msgs/Image`, `32FC1` | Rectified optical Z in metres, NaN invalid |
| `/stereo/points` | `sensor_msgs/PointCloud2` | Current frame, left rectified optical frame |
| `/stereo/points_world` | `sensor_msgs/PointCloud2` | Current frame transformed to the VIO world |
| `/stereo/map` | `sensor_msgs/PointCloud2` | Fused voxel centroids in the VIO world |

Images and per-frame point clouds are generated for subscribers using
best-effort sensor QoS. `/stereo/map` uses reliable, transient-local QoS with
depth 1, so new subscribers receive the latest map. It updates only when dirty,
at most once per `map_publish_period_sec`.

EuRoC's monochrome images produce grayscale RGB point colors. Color input in
`bgr8`, `rgb8`, `bgra8` and `rgba8` is also accepted. Input size must exactly
match the calibration; implicit image resizing is deliberately disabled.

## Save and clear

```bash
ros2 service call /stereo/save_map std_srvs/srv/Trigger '{}'
ros2 service call /stereo/clear_map std_srvs/srv/Trigger '{}'
```

The default save path is `/tmp/stereo_map.pcd`. Set `save_map_path` in the YAML,
or change this parameter while running:

```bash
ros2 param set /stereo_reconstruction save_map_path /tmp/stereo_map.ply
```

PCD and PLY are ASCII, contain XYZ and RGB, and use metres in the VIO world.
The output directory must exist. Saving an empty map returns failure without
creating a file. An existing output file is overwritten.

Clear removes the map, pending stereo pairs, cached poses and keyframe history;
an in-flight job from the old generation cannot be fused into the cleared map.
Clear does not reset MSCKF. After resetting MSCKF's world origin, clear this map
as well. Backwards image timestamps (for example restarting a bag) automatically
clear reconstruction state.

## Geometry and synchronization

1. Raw stereo images are paired with `ExactTime`, matching the existing EuRoC
   image processor. Hardware-synchronized image timestamps are required.
2. Startup computes `R1/R2/P1/P2/Q` and rectification maps from calibrated
   intrinsics, distortion and `cam1/T_cn_cnm1`. Both radtan and equidistant
   models are supported; the matcher requires horizontal, positive disparity.
3. SGBM converts rectified input to grayscale, computes signed 16-bit disparity,
   and converts it to floating pixel disparity with `/16` and NaN invalids.
4. Reprojection uses `Q`. It rejects invalid disparities, rectification borders,
   invalid right-image correspondence pixels, and depths outside configured
   limits. `pixel_stride` decimates point clouds; depth remains full resolution.
5. MSCKF odometry is stamped with the left image timestamp. The default pose
   tolerance is **zero**. A small configurable nearest-pose tolerance is
   available, but this version does not interpolate/extrapolate poses.
   Frame names must match the configured VIO parent/child frames. The latest
   arriving pose is never substituted for a missing timestamp match.
6. Define `T_A_B` as a transform from B coordinates to A coordinates:

   ```text
   T_body_left = inverse(T_imu_body) * inverse(cam0/T_cam_imu)
   T_body_left_rect = T_body_left * [transpose(R1), 0]
   p_world = T_world_body(image_stamp) * T_body_left_rect * p_left_rect
   ```

   Static transforms from the VIO body frame to both rectified optical frames
   are published on `/tf_static`. In the supplied EuRoC configuration the world
   frame is `world` and the body child frame is named `odom`; the latter name
   does not mean that the input pose is already a camera pose.

The worker computes depth independently of pose arrival and waits up to
`pose_wait_sec` afterwards for matching odometry. Missing poses skip fusion but
do not prevent depth publication. The ROS callbacks remain free to receive
odometry during the wait. The bounded input queue discards oldest pending
pairs on overload; slow bag playback for more complete offline reconstruction.
Logs report processed frames, accepted keyframes, queue drops and missing poses.

## Fusion settings

Defaults in `config/stereo_reconstruction_euroc_params.yaml`:

| Setting | Default | Meaning |
|---|---|---|
| `sgbm.num_disparities` | 128 | Positive multiple of 16 |
| `sgbm.block_size` | 5 | Odd matching window |
| `min_depth` / `max_depth` | 0.3 / 10.0 m | Valid optical depth range |
| `pixel_stride` | 2 | Point-cloud sampling stride |
| `keyframe_translation` | 0.1 m | Translation since last fused keyframe |
| `keyframe_rotation_deg` | 5 degrees | Rotation since last fused keyframe |
| `voxel_size` | 0.05 m | World voxel side length |
| `max_voxels` | 500000 | Maximum map entries |
| `max_pending_frames` | 4 | Pending stereo queue capacity |
| `opencv_threads` | 2 | Limit dense-stereo CPU concurrency |

A frame is fused when either motion threshold is exceeded. Voxels average XYZ
and RGB observations; their update weights saturate at 1000 observations.
Negative world coordinates use floor-based voxel indexing. At map capacity,
existing voxels continue updating and unseen voxels are skipped with a warning.
Most parameters are read once at startup; restart to apply geometry/matcher
changes. Only `save_map_path` is read again on each save request.

This is a point-cloud reconstruction baseline with SGBM speckle/uniqueness and
depth filtering. It has no TSDF, surface meshing, loop closure, dynamic-object
removal or pose-graph reintegration. The map inherits VIO drift and any change
in camera/IMU extrinsics estimated online; reconstruction uses startup extrinsics.

## Adding a TensorRT backend

Implement `StereoMatcher::compute()` and `name()`, then register the backend in
`createStereoMatcher()`. The virtual interface is in
`include/msckf_vio/stereo/stereo_matcher.h`. The contract is:

- equally sized, rectified mono8/BGR8 inputs;
- `CV_32FC1` output at the input resolution;
- positive `x_left - x_right` disparity in **input-image pixels**;
- NaN for invalid output.

The backend owns normalization, color conversion, padding, output dtype/layout,
and scaling disparity values when restoring resolution. Rectification, pose
matching and fusion stay shared. No TensorRT backend is implemented in this
SGBM baseline.

## Verification

```bash
colcon build --packages-select msckf_vio --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select msckf_vio --event-handlers console_direct+
colcon test-result --verbose
```

The added core tests cover known-shift SGBM pixel scale, metric depth, invalid
correspondences, inverse rectification rotation, transform layout validation,
timestamp pose selection, keyframe thresholds, bounded voxel fusion and export.

For a manual ROS integration check, source the built workspace and run:

```bash
python3 src/msckf_vio/test/stereo_ros_smoke.py /path/to/V2_01_easy_ros2 \
  --duration 15 --rate 0.5 --domain-id 91
```

This reads camera/IMU records from the bag's SQLite file without changing it,
starts the real VIO and reconstruction nodes in a separate localhost domain,
and checks depth, world point clouds, map save and clear. Use an otherwise
unused domain. Output map, logs and metrics go to
`/tmp/msckf_stereo_euroc_validation` (override with `--output`).

Local verification on `V2_01_easy`, first 15 sensor seconds at 0.5 playback
rate: 279 depth images, 281 odometry messages, 258 current world point clouds,
and 195,987 fused map points. The final depth image had 76.1% valid pixels.
PCD save and map clear passed, and the exported XYZ/RGB values were finite.
This is an integration check on a short sequence, not an accuracy benchmark or
a claim of full-rate real-time performance.

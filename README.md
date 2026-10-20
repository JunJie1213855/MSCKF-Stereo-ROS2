# MSCKF\_VIO

The `MSCKF_VIO` package is a stereo version of MSCKF. The software takes in synchronized stereo images and IMU messages and generates real-time 6DOF pose estimation of the IMU frame.

This is a ROS 2 (Humble) port of the original ROS 1 package; the estimator itself is unchanged.

An independent stereo reconstruction node supports SGBM and an optional IGEV
TensorRT backend to rectify EuRoC images and fuse dense point clouds using the MSCKF poses. See
[stereo reconstruction](docs/stereo_reconstruction.md) for launching, parameters,
map export, IGEV setup and the virtual matcher interface.

Video: [https://www.youtube.com/watch?v=jxfJFgzmNSw&t](https://www.youtube.com/watch?v=jxfJFgzmNSw&t=3s)<br/>
Paper Draft: [https://arxiv.org/abs/1712.00036](https://arxiv.org/abs/1712.00036)

---

## System environment

The versions below are the ones this package was built and run against; newer point releases of the same lines are normally fine.

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS (kernel 6.8) |
| ROS 2 | Humble |
| Compiler | g++ 12.3.0, C++17 |
| CMake | 4.4.4 |
| Python | 3.10.12 |
| Eigen | 3.4.0 |
| OpenCV | 4.5.4 |
| PCL | 1.12.1 |
| Boost | 1.74.0 |
| SuiteSparse | 5.10.1 |

ROS 2 packages used: `rclcpp`, `std_msgs`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `std_srvs`, `tf2`, `tf2_ros`, `tf2_eigen`, `tf2_geometry_msgs`, `image_transport`, `cv_bridge`, `message_filters`, `pcl_conversions`, `rosidl_default_generators`, `eigen3_cmake_module`. These all ship with `ros-humble-desktop`.

---

## Quick start

### 1. Install the dependencies

```bash
sudo apt install -y libeigen3-dev libopencv-dev libpcl-dev libboost-dev libsuitesparse-dev
```

If you already have a colcon workspace containing this package, `rosdep` can resolve the ROS 2 side automatically:

```bash
rosdep install --from-paths src --ignore-src -y
```

### 2. Build

```bash
cd <your_colcon_workspace>
colcon build --packages-select msckf_vio --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

A clean build takes a few minutes; add `--parallel-workers 4` on a machine that is short on cores or that stalls under a full-width build.

### 3. Run

Three terminals, all sourcing the workspace. The `ROS_DOMAIN_ID` line matters if anything else is already running on the host — see [Avoiding topic clashes](#avoiding-topic-clashes).

```bash
# 0) in every terminal
export ROS_DOMAIN_ID=42
source <your_colcon_workspace>/install/setup.bash
```

```bash
# 1) the estimator, with rviz2 alongside it
ros2 launch msckf_vio msckf_vio_euroc_viz.launch.py

# 2) the dataset
ros2 bag play <bag_dir> --topics /imu0 /cam0/image_raw /cam1/image_raw
```

To start rviz2 separately instead (or on another machine), drop `_viz` and run:

```bash
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix msckf_vio)/share/msckf_vio/rviz/rviz_euroc_config.rviz
```

`/odom` starts about a second into the bag, once 200 IMU messages have been consumed to initialize the gravity direction and the IMU biases.

> **The sensor suite has to start stationary.** Initialization fits the initial orientation and biases from the first 200 IMU samples, so a run that begins while the sensor is moving will initialize badly or not at all.

### 4. What a healthy run looks like

The `image_processor` throttles its tracking statistics to 2 Hz:

```
candidates: 49; track: 49; match: 49; ransac: 48/49=0.979592
```

`ransac` is the fraction of stereo matches that survived the 2-point RANSAC outlier rejection; roughly 0.8–1.0 is normal on EuRoC. `/odom` should come out at the camera rate (20 Hz for EuRoC). If the machine cannot keep up, slow the playback with `ros2 bag play --rate 0.5` rather than dropping messages.

In rviz2 the estimate appears as an axis triad plus a fading trail of past poses, the map features as a point cloud, and the tracked features drawn on the stereo pair in the "Stereo Features" panel. The debug image is only generated while something is subscribed to it, which that panel does.

---

## Configuration

Everything tunable lives in one file per dataset:

| File | Dataset |
| --- | --- |
| `config/msckf_vio_euroc_params.yaml` | EuRoC |
| `config/msckf_vio_euroc_noextrinsics_params.yaml` | EuRoC, extrinsics estimated rather than held fixed |
| `config/msckf_vio_fla_params.yaml` | UPenn fast flight (fisheye) |

Both nodes read from the same file, which is split into an `image_processor:` section and a `vio:` section keyed by node name. Because ROS 2 matches a parameter file to nodes by fully qualified name, the two nodes are launched at the top level with the fixed names `image_processor` and `vio`; putting them in a namespace would stop the file from being applied.

The parameters you are most likely to change:

**Camera calibration** (must match your hardware; see [Calibration](#calibration))

- `cam0/intrinsics`, `cam1/intrinsics` — `[fx, fy, cx, cy]`
- `cam0/distortion_coeffs`, `cam1/distortion_coeffs`
- `cam0/distortion_model`, `cam1/distortion_model` — `radtan` or `equidistant`
- `cam0/resolution`, `cam1/resolution`
- `cam0/T_cam_imu`, `cam1/T_cn_cnm1`, `T_imu_body` — 16 row-major doubles each

The transform entries are flat 16-element lists, because ROS 2 parameters cannot express nested arrays. **Every element must be written as a float** (`0.0`, `1.0`): `rcl_yaml_param_parser` rejects a sequence that mixes integers and doubles, and the node will refuse to start.

**Feature tracking** (`image_processor` section)

- `grid_row`, `grid_col`, `grid_min_feature_num`, `grid_max_feature_num` — how features are spread over the image
- `fast_threshold`, `patch_size`, `pyramid_levels`, `max_iteration`, `track_precision`
- `ransac_threshold`, `stereo_threshold` — outlier rejection thresholds

**Filter** (`vio` section)

- `noise/gyro`, `noise/acc`, `noise/gyro_bias`, `noise/acc_bias`, `noise/feature` — noise densities, **given as standard deviations** and squared internally
- `initial_covariance/*` — initial uncertainty; the extrinsic covariances are what separate the `noextrinsics` variant
- `max_cam_state_size` — size of the sliding window of camera clones
- `fixed_frame_id`, `child_frame_id`, `publish_tf`, `frame_rate`

The generated files are derived from the `camchain-imucam-*.yaml` calibration files. If you recalibrate, regenerate rather than hand-editing, and keep the floats: a transform element that lands in the YAML as `0` instead of `0.0` will fail the parse.

---

## Calibration

An accurate calibration is crucial for successfully running the software. To get the best performance of the software, the stereo cameras and IMU should be hardware synchronized. Note that for the stereo calibration, which includes the camera intrinsics, distortion, and extrinsics between the two cameras, you have to use a calibration software. **Manually setting these parameters will not be accurate enough.** [Kalibr](https://github.com/ethz-asl/kalibr) can be used for the stereo calibration and also to get the transformation between the stereo cameras and IMU. The yaml file generated by Kalibr can be directly used in this software. See calibration files in the `config` folder for details. The two calibration files in the `config` folder should work directly with the EuRoC and [fast flight](https://github.com/KumarRobotics/msckf_vio/wiki) datasets. The convention of the calibration file is as follows:

`camx/T_cam_imu`: takes a vector from the IMU frame to the camx frame.

`cam1/T_cn_cnm1`: takes a vector from the cam0 frame to the cam1 frame.

---

## Datasets

### EuRoC

Obtain the [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) dataset. It ships as ROS 1 bags, which ROS 2 cannot play directly. Convert with [`rosbags`](https://pypi.org/project/rosbags/) (no ROS 1 installation needed):

```bash
pip install rosbags
rosbags-convert --src V1_01_easy.bag --dst V1_01_easy_ros2
ros2 bag play V1_01_easy_ros2 --topics /imu0 /cam0/image_raw /cam1/image_raw
```

**The `--topics` filter is required.** The EuRoC bags also carry `/fcu/motor_speed`, typed `asctec_hl_comm/msg/MotorSpeed`, which has no ROS 2 release; playback aborts when it tries to publish that topic.

The EuRoC ground truth (`vicon/firefly_sbx/firefly_sbx`, a `geometry_msgs/TransformStamped`) does not match the `nav_msgs/Odometry` the `vio` node expects on `mocap_odom`, so `gt_odom` is not published from these bags. The estimator itself does not use the ground truth.

### UPenn fast flight

See the [dataset page](https://github.com/KumarRobotics/msckf_vio/wiki/Dataset) and use `msckf_vio_fla.launch.py`. This dataset is fisheye, so the calibration uses the `equidistant` distortion model.

### Avoiding topic clashes

The nodes publish generic names (`/odom`, `/tf`) that collide with most robot stacks. If another ROS 2 system is already running on the host — a Nav2 or turtlebot bringup, for example — give this pipeline its own domain:

```bash
export ROS_DOMAIN_ID=42
```

Every terminal (launch, rviz2 and the bag playback) must use the same value. Without it, `/odom` ends up with two publishers, `ros2 topic hz` reports roughly double the real rate, and rviz2 can draw another robot's pose on top of the estimate.

---

## ROS interface

Neither node uses a namespace, and the relative topic names below are remapped by the launch files, so `features` from the `image_processor` ends up on `/image_processor/features` and the `vio` node subscribes to that same topic.

### `image_processor` node

**Subscribed Topics**

`imu` (`sensor_msgs/Imu`)

IMU messages is used for compensating rotation in feature tracking, and 2-point RANSAC.

`cam0_image`, `cam1_image` (`sensor_msgs/Image`)

Synchronized stereo images.

**Published Topics**

`features` (`msckf_vio/CameraMeasurement`)

Records the feature measurements on the current stereo image pair.

`tracking_info` (`msckf_vio/TrackingInfo`)

Records the feature tracking status for debugging purpose.

`debug_stereo_image` (`sensor_msgs/Image`)

Draw current features on the stereo images for debugging purpose. Note that this debugging image is only generated upon subscription.

### `vio` node

**Subscribed Topics**

`imu` (`sensor_msgs/Imu`)

IMU measurements.

`features` (`msckf_vio/CameraMeasurement`)

Stereo feature measurements from the `image_processor` node.

**Published Topics**

`odom` (`nav_msgs/Odometry`)

Odometry of the IMU frame including a proper covariance.

`feature_point_cloud` (`sensor_msgs/PointCloud2`)

Shows current features in the map which is used for estimation.

**Services**

`reset` (`std_srvs/srv/Trigger`)

Resets the filter:

```bash
ros2 service call /reset std_srvs/srv/Trigger
```

or `ros2 launch msckf_vio reset.launch.py`. Note that this is not an anytime reset: the IMU bias and initial orientation are re-estimated from the next 200 IMU samples, so the sensor suite must be stationary.

### Launch files

| File | Purpose |
| --- | --- |
| `msckf_vio_euroc.launch.py` | EuRoC: `image_processor` + `vio` |
| `msckf_vio_euroc_viz.launch.py` | EuRoC, plus rviz2 with the trajectory display |
| `image_processor_euroc.launch.py` | EuRoC: `image_processor` only |
| `msckf_vio_euroc_noextrinsics.launch.py` | EuRoC with the extrinsics estimated |
| `msckf_vio_fla.launch.py` | Fast flight: `image_processor` + `vio` |
| `image_processor_fla.launch.py` | Fast flight: `image_processor` only |
| `reset.launch.py` | Calls the reset service |

All of them accept `imu_topic`, `cam0_topic` and `cam1_topic` to override the input topics. The viz variant also takes `rviz:=false` to skip rviz2.

---

## Evaluating accuracy with evo

The scripts in `scripts/` record a run and turn it into numbers with [evo](https://github.com/MichaelGrupp/evo). Install evo with `pip install evo`.

**1. Run the estimator and record the trajectory.** In the terminal that plays the bag, leave the estimator running; in a second one:

```bash
python3 $(ros2 pkg prefix msckf_vio)/share/msckf_vio/scripts/record_odom.py /tmp/est.txt
```

It runs until Ctrl-C, writing `/odom` as it arrives.

**2. Convert the ground truth.** EuRoC ships it under `mav0/state_groundtruth_estimate0/data.csv` in the ASL column order, which evo does not read:

```bash
python3 $(ros2 pkg prefix msckf_vio)/share/msckf_vio/scripts/gt_asl_to_tum.py \
    <seq>/mav0/state_groundtruth_estimate0/data.csv /tmp/gt.txt
```

**3. Compare.**

```bash
# absolute trajectory error, SE(3)-aligned (scale NOT free - VIO scale is observable)
evo_ape tum /tmp/gt.txt /tmp/est.txt -a

# relative pose error over 1 m, and over ~1 s (20 frames at 20 Hz)
evo_rpe tum /tmp/gt.txt /tmp/est.txt -a -d 1 -u m
evo_rpe tum /tmp/gt.txt /tmp/est.txt -a -d 20 -u f

# rotational error
evo_rpe tum /tmp/gt.txt /tmp/est.txt -a -d 1 -u m -r angle_deg
```

Use SE(3) alignment (`-a`), not Sim(3) (`-as`): VIO observes metric scale, so letting the alignment absorb a scale error would hide a real failure. Running both is a useful check — if they disagree much, the scale is off.

For reference, on EuRoC `V2_01_easy` this implementation gives ATE RMSE **0.070 m** and RPE RMSE **0.044 m / 1 m** with a scale factor of **1.0069** from the Sim(3) fit, over 2200 poses covering 98% of the sequence.

**If evo's plotting fails** with `cannot import name 'docstring' from 'matplotlib'`, the machine has both an apt and a pip matplotlib and Debian's `matplotlib-3.5.1-nspkg.pth` pins `mpl_toolkits` to the apt copy. `scripts/evo_run.py` works around it (see its docstring for the details); the reported numbers are unaffected either way:

```bash
python3 $(ros2 pkg prefix msckf_vio)/share/msckf_vio/scripts/evo_run.py \
    ape tum /tmp/gt.txt /tmp/est.txt -a --save_plot /tmp/ape.png
```

Use `--save_plot`, not `--plot` — showing the window needs PyQt6. evo appends `_raw` and `_map` to the filename given.

---

## Troubleshooting

**The node exits with `RCLInvalidROSArgsError` / "Sequence should be of same type".** A parameter sequence mixes integers and doubles. See [Configuration](#configuration) — every matrix element needs a decimal point.

**`ros2 bag play` aborts on `/fcu/motor_speed`.** Add the `--topics` filter shown above.

**`/odom` shows more than one publisher, or `ros2 topic hz` reports the wrong rate.** Another ROS 2 system on the host is publishing the same topic. Use a separate `ROS_DOMAIN_ID`, as described under [Avoiding topic clashes](#avoiding-topic-clashes).

**rviz2 reports `PluginlibFactory: The plugin for class 'rviz/...' failed to load`.** You are loading a ROS 1 `.rviz` file. Use the configs in `rviz/`, which are in rviz2 format.

**Nothing is published until well into the bag.** Expected — the filter needs the first 200 IMU messages plus one stereo pair before it initializes.

**Known issue.** An intermittent segfault in `image_processor_node` (exit code -11) has been observed under `ros2 launch` while feeding degenerate stereo input — identical left/right images that yield zero features. It has not been reproduced from well-formed stereo data, and no stack trace was captured. Treat featureless input as a risk area if you are modifying the image processor.

---

## License

Penn Software License. See LICENSE.txt for further details.

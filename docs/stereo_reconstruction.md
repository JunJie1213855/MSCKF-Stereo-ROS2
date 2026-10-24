# EuRoC + MSCKF stereo point-cloud fusion

Default fusion is now sparse CPU TSDF; see [TSDF fusion](tsdf_fusion.md).
Use `fusion_method:=voxel` for the original centroid map. Matcher and fusion
are independent choices.

The `stereo_recon` package owns `stereo_reconstruction_node` and runs independently of the MSCKF estimator.
Use `ros2 launch stereo_recon stereo_recon.launch.py` for reconstruction only;
the combined EuRoC launch remains in `msckf_vio`. Its
backends are OpenCV SGBM, IGEV and S2M2 on TensorRT 10.x/11.x, sharing the virtual
`StereoMatcher` interface. The default SGBM-only build needs no CUDA/TensorRT;
the existing package dependencies still apply.

## Build and run

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select stereo_recon msckf_vio --cmake-args -DCMAKE_BUILD_TYPE=Release
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
| `/stereo/map` | `sensor_msgs/PointCloud2` | TSDF surface points or voxel centroids in the VIO world |

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

This is a point-cloud reconstruction baseline with depth filtering and,
for the SGBM backend, speckle/uniqueness filtering. TSDF surface point fusion is supported; it has no surface meshing, loop closure, dynamic-object
removal or pose-graph reintegration. The map inherits VIO drift and any change
in camera/IMU extrinsics estimated online; reconstruction uses startup extrinsics.

## IGEV TensorRT backend

Enable TensorRT explicitly when building (the default is OFF):

```bash
colcon build --packages-select stereo_recon --cmake-args \
  -DSTEREO_RECON_ENABLE_TENSORRT=ON \
  -DTensorRT_ROOT=/home/ros/lib/TensorRT-11.3.0.99 \
  -DCUDA_ROOT=/usr/local/cuda-12.1
source install/setup.bash
```

Use the SDK roots for the target machine. No `nvcc` compilation is required;
the backend uses the TensorRT/CUDA C++ runtime APIs. To return to a CPU-only
build, set `-DSTEREO_RECON_ENABLE_TENSORRT=OFF` explicitly in an existing build tree.
The node reports startup failure with a nonzero exit status when IGEV was not
built, the engine is absent/empty/incompatible, or tensor names do not match.
It does not fall back silently to SGBM.

After engine conversion has completed:

```bash
ros2 launch msckf_vio stereo_reconstruction_euroc.launch.py \
  backend:=igev \
  engine_path:=/home/ros/rosws/msckf_stereo_ws/src/stereo_recon/weights/model.engine
```

Add `start_vio:=false` to attach to an existing MSCKF run. All topics and map
services stay the same. The original launch without overrides selects the
backend in the YAML, which defaults to SGBM. Engine files are used in place;
model weights are not copied into the package installation or modified.

The supplied `weights/igev_480_752.onnx` was inspected: its actual input names
and shapes are `left` and `right`, both `[1,3,480,768]` FP32, and its output is
`disparity`, `[1,1,480,768]` FP32. The width in the filename is not the engine
input width. For EuRoC, the backend pads 752 columns to 768, then crops the
output back to 752 without scaling the pixel disparity.

The preprocessing contract is RGB FP32 NCHW with values in `[0,255]`; the
exported graph already contains normalization. Monochrome input is replicated
into three channels. TensorRT FP32 and FP16 I/O are supported and the restored
disparity is always `CV_32FC1`. No `/16` conversion is applied to IGEV output.

The `igev` section of the reconstruction YAML provides:

| Parameter | Default | Meaning |
|---|---|---|
| `engine_path` | empty | Absolute path to a completed TensorRT engine |
| `left_input_name` / `right_input_name` | `left` / `right` | Exact engine input names |
| `output_name` | `disparity` | Exact final disparity output name |
| `input_width` / `input_height` | 0 / 0 | Infer static dimensions from the engine |
| `padding_multiple` | 32 | Auto-padding multiple for dynamic dimensions |
| `padding_mode` | `right_bottom` | Match the existing thirdparty IGEV example |
| `device_id` | 0 | CUDA GPU index |

`right_bottom` pads the bottom/right with zeros. `sintel` uses centered
replicated-edge padding, matching the original IGEV `InputPadder` convention.
Choose the same convention as the reference preprocessing used to validate
the exported model. Both cameras use identical offsets. Larger images than a
fixed engine accepts are rejected; no implicit resize/crop changes calibration.
For dynamic engines, profile 0 is used and input sizes must fit its bounds.

The final output must be full-resolution `[H,W]`, `[1,H,W]` or `[1,1,H,W]`
disparity in network-input pixels. Low-resolution predictions are rejected,
because their disparity units cannot safely be inferred from tensor dimensions.
Nonpositive or nonfinite disparities become NaN before depth reconstruction.

The runtime owns one CUDA stream/context and reuses device/host I/O buffers.
It selects the device on the ROS worker thread and checks transfer, enqueue
and synchronization errors. Extra engine outputs receive bindings; the named
final disparity output is the one downloaded. Rectification, timestamp pose
matching, keyframe selection and voxel fusion are shared with SGBM.

## S2M2 TensorRT backend

Build with the same `STEREO_RECON_ENABLE_TENSORRT=ON` option as IGEV, then select
`backend:=s2m2`. `engine_path` overrides `s2m2.engine_path` when that backend
is selected (including when selection comes from the reconstruction YAML).

```bash
ros2 launch msckf_vio stereo_reconstruction_euroc.launch.py \
  backend:=s2m2 \
  engine_path:=/home/ros/rosws/msckf_stereo_ws/src/stereo_recon/weights/s2m2/S2M2_S_768_480_fp16.engine
```

The input/output contract follows the local `/home/ros/lib/s2m2` sources:
`core/utils/image_utils.py:image_pad`, `core/model/s2m2.py:normalize_img`,
and `core/utils/vis_utils.py:visualize_stereo_results_2d`.

| Tensor | Type | EuRoC NCHW shape |
|---|---|---|
| `input_left`, `input_right` | RGB UINT8/FP32/FP16, 0–255 | `[1,3,480,768]` |
| `output_disp` | FP32/FP16 pixel disparity | `[1,1,480,768]` |
| `output_occ` | FP32/FP16 visibility (larger means visible) | `[1,1,480,768]` |
| `output_conf` | FP32/FP16 confidence | `[1,1,480,768]` |

Normalization is inside the exported model. The C++ preprocessing replicates
mono images to three channels, converts BGR to RGB, and produces contiguous
UINT8 NCHW, then casts to the engine input dtype without rescaling.
Both UINT8 and floating-point S2M2 engines receive RGB values in [0,255];
normalization stays inside the exported graph. It reproduces the source's centered padding: zero-pad, adaptive
average pool to `[original_H // 32, original_W // 32]`, bilinearly interpolate
to the engine size, restore the original image in the center, and truncate
float padding values to UINT8. This is neither plain zero padding nor edge
replication. For EuRoC it adds 8 columns on each side. All three outputs are
cropped identically; disparity remains in original-image pixels, with no
`/16` conversion. Input images are never resized or transposed.

Default valid pixels require `output_conf > 0.1`, `output_occ > 0.5` and finite,
positive disparity, matching the upstream display mask. Invalid samples become
NaN before depth projection and fusion. Thresholds are configurable through
`s2m2.min_confidence` and `s2m2.min_visibility`. Quality maps are consumed
internally; the public disparity/depth topics contain the filtered result.

The `s2m2` YAML section also provides `engine_path`, `left_input_name`,
`right_input_name`, `output_name`, `visibility_output_name`,
`confidence_output_name`, `input_width`, `input_height` and `device_id`.
Zero input dimensions read the engine dimensions; dynamic engines pad to a
multiple of 32 using profile 0. Missing quality outputs or incompatible types
are errors, rather than silently skipping the mask.

**Check actual engine shapes, not filenames.** The initially supplied
`weights/s2m2/S2M2_S_480_768_fp16.engine` has actual shape `[1,3,768,480]`
(height 768, width 480). It can run portrait images, but cannot accept native
EuRoC images (height 480, width 752). The re-exported
`weights/s2m2/S2M2_S_768_480_fp16.engine` has the correct `[1,3,480,768]`
inputs and `[1,1,480,768]` outputs; use this landscape engine for EuRoC.
The node checks the calibrated image dimensions at startup and reports the
mismatch. Changing YAML dimensions cannot change a fixed-shape engine.

Local verification with this portrait engine and the upstream `0025_L/R`
sample cropped to height 752, width 480 passed: C++ RGB UINT8 buffers matched
the original Python `image_pad()` buffers byte-for-byte for both cameras.
C++ and Python TensorRT produced identical valid-pixel masks (196,092 of
360,960 pixels) and zero disparity difference on valid pixels after cropping
and quality filtering. This checks preprocessing, bindings and postprocessing;
it is not a depth accuracy benchmark or an end-to-end EuRoC S2M2 test.

## Adding another backend

For the implementation stages and thread/geometry contracts, see
[the Stereo code walkthrough](stereo_code_walkthrough.md).

Implement `StereoMatcher::Compute()` and `name()`, then register the backend in
`CreateStereoMatcher()`. The virtual interface is in
`include/stereo_recon/stereo/stereo_matcher.h`. The contract is:

- equally sized, rectified mono8/BGR8 inputs;
- `CV_32FC1` output at the input resolution;
- positive `x_left - x_right` disparity in **input-image pixels**;
- NaN for invalid output.

The backend owns normalization, color conversion, padding, output dtype/layout,
and scaling disparity values when restoring resolution. Rectification, pose
matching and fusion stay shared.

## Verification

```bash
colcon build --packages-select stereo_recon --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select stereo_recon --event-handlers console_direct+
colcon test-result --verbose
```

The added core tests cover known-shift SGBM pixel scale, metric depth, invalid
correspondences, inverse rectification rotation, transform layout validation,
timestamp pose selection, keyframe thresholds, bounded voxel fusion and export.
IGEV CPU tests additionally cover RGB/NCHW layout, normalization range,
monochrome input, padding offsets, FP16 conversion, pixel-disparity preservation,
invalid output handling and missing-engine/disabled-build errors. They do not
require a GPU or a completed engine; GPU numerical validation is a separate check.

For a manual ROS integration check, source the built workspace and run:

```bash
python3 src/stereo_recon/test/stereo_ros_smoke.py /path/to/V2_01_easy_ros2 \
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

The same integration check supports a completed IGEV engine:

```bash
python3 src/stereo_recon/test/stereo_ros_smoke.py /path/to/V2_01_easy_ros2 \
  --duration 8 --rate 0.25 --domain-id 94 --backend igev \
  --engine /absolute/path/to/model.engine \
  --output /tmp/msckf_igev_euroc_validation
```

Local IGEV verification with `weights/model.engine` on a GTX 1650 Ti, first
8 sensor seconds of `V2_01_easy` at 0.25 playback rate: 122 depth images,
141 odometry messages, 112 world point clouds and 41,851 fused map points.
The final depth image had 97.8% valid pixels. Map save and clear passed,
and all exported XYZ values were finite. Some input frames were dropped
under inference load; start interactive playback at `--rate 0.2` on this GPU.
These results check integration; they do not measure depth or pose accuracy.

For S2M2 ROS integration, use the same smoke script with `--backend s2m2`
and `--engine /absolute/path/to/s2m2_euroc.engine`. A correctly oriented
`[1,3,480,768]` engine is required. CPU tests for S2M2 cover UINT8 RGB layout,
centered pooled padding, strict quality thresholds, FP16 output handling,
invalid measurements and rejection of a transposed fixed engine.

Local end-to-end S2M2 verification with the corrected landscape engine on
`V2_01_easy`, first 8 sensor seconds at 0.2 playback rate, passed: 94 depth
frames, 141 odometry messages, 84 world point clouds and 33,009 fused map
points. The last depth frame had 97.6% valid pixels after quality and geometry
filtering. Save/clear passed and all exported XYZ values were finite.
On this GTX 1650 Ti, processing was about 380 ms per frame including pose wait;
0.2 playback dropped some input frames. Start interactive S2M2 playback at
`--rate 0.1`. This short integration check does not measure depth accuracy.

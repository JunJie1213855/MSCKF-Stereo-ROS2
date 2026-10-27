# stereo_recon

> 当前 AirSLAM 组合工作区使用 TensorRT 11.3.0.99 / CUDA 12.1。
> 请先阅读 [组合构建与运行说明](../airslam_recon_bringup/docs/BUILD_AND_RUN.md)，
> 使用其中的 AirSLAM 相机位姿标定；下方 MSCKF 示例属于另一种接入方式。


独立 ROS 2 Humble 双目稠密重建包，提供 SGBM、IGEV TensorRT、S2M2 TensorRT、图像校正、深度投影与 TSDF/体素点云融合。通过标准 `sensor_msgs/Image` 和 `nav_msgs/Odometry` 接口接入任意满足标定/坐标约定的里程计；C++ 构建不依赖 `msckf_vio`。

默认启用 Ceres 局部点云 BA、推理前关键帧筛选及活动 TSDF 子图重建。
任务书见 [点云 BA 任务书](docs/plan/point_cloud_ba.md)，算法、归档、离线对照及
资源限制见 [点云 BA 使用说明](docs/point_cloud_ba.md)。构建需要 `libceres-dev`
和 `libyaml-cpp-dev`；`point_cloud_ba:=false` 关闭优化。

## 编译

```bash
source /opt/ros/humble/setup.bash
cd /home/ros/rosws/msckf_stereo_ws
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --packages-select stereo_recon --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DSTEREO_RECON_ENABLE_TENSORRT=ON \
  -DTensorRT_ROOT=/home/ros/lib/TensorRT-11.3.0.99 -DCUDA_ROOT=/usr/local/cuda-12.1
source install/setup.bash
```

CPU-only 构建设置 `-DSTEREO_RECON_ENABLE_TENSORRT=OFF`。TensorRT 开关归本包管理，原来的 `MSCKF_ENABLE_TENSORRT` 不再控制 MSCKF。

## 独立启动重建（VIO 已在运行）

```bash
source /opt/ros/humble/setup.bash
source /home/ros/rosws/msckf_stereo_ws/install/setup.bash
source /home/ros/rosws/msckf_stereo_ws/src/stereo_recon/scripts/setup_wsl_env.sh
export ROS_DOMAIN_ID=42
ros2 launch stereo_recon stereo_recon.launch.py \
  backend:=s2m2 \
  engine_path:=/home/ros/rosws/msckf_stereo_ws/src/stereo_recon/weights/S2M2_L_768_480_v2_torch26.model \
  vio_params_file:=/home/ros/rosws/msckf_stereo_ws/src/MSCKF-Stereo-ROS2/config/msckf_vio_euroc_params.yaml
```

默认订阅 `/cam0/image_raw`、`/cam1/image_raw`、`/odom`，默认 `use_sim_time:=true`。真实相机使用 `use_sim_time:=false`。左右图像必须同时间戳，位姿按左图采集时间匹配。重建节点不会启动或重置 VIO。

脱离 MSCKF 时，省略 `vio_params_file`，通过 `calibration_file` 指定本包格式的相机/机体标定 YAML（模板为 `config/stereo_calibration_euroc.yaml`）。标定、`fixed_frame_id` 与 `pose_child_frame_id` 必须匹配里程计；真实设备需替换 EuRoC 标定。

`cam0_topic`、`cam1_topic`、`odom_topic` 可重映射输入；`params_file` 选择重建参数；`rviz:=false` 关闭 RViz。默认后端仍为 SGBM。

## 同时启动 MSCKF + 重建

组合入口仍由 `msckf_vio` 提供，读取同一份 VIO 标定，启动新包的重建节点：

```bash
ros2 launch msckf_vio stereo_reconstruction_euroc.launch.py \
  backend:=s2m2 \
  engine_path:=/home/ros/rosws/msckf_stereo_ws/src/stereo_recon/weights/S2M2_L_768_480_v2_torch26.model
```

播放 ROS 2 bag 时，各终端使用相同 DDS 配置与 ROS_DOMAIN_ID：

```bash
ros2 bag play /path/to/euroc_ros2_bag --clock --rate 0.1 \
  --topics /imu0 /cam0/image_raw /cam1/image_raw
```

## 输出和地图

节点名保留 `/stereo_reconstruction`。输出话题与服务保留：

- `/stereo/disparity`、`/stereo/depth`：视差与米制深度。
- `/stereo/points`、`/stereo/points_world`：相机/世界坐标系当前帧点云。
- `/stereo/map`：TSDF 零交叉表面点云，或 voxel 模式的融合质心点云。
- `/stereo/save_map`、`/stereo/clear_map`：保存与清空服务。

RViz 同时显示当前帧和融合地图。模型文件放在 `weights/` 或由绝对路径指定；不安装进 share，也不提交版本库。旧源码目录的 `weights` 链接指向本包，兼容已有 engine 路径。

BA 模式下地图和当前建图帧点云使用 `stereo_map`，新增 `/stereo/optimized_path`、
`/stereo/optimized_odometry`、`/stereo/ba/status` 与 `/stereo/save_trajectory`。
`archive_directory:=/path/to/archive` 开启原始关键帧归档，
`ros2 run stereo_recon stereo_ba_offline ARCHIVE OUTPUT_PREFIX` 用于离线处理。
只选中的候选帧执行深度推理，地图输出合并活动/冻结 TSDF 表面。

详细参数、S2M2 的 RGB/NCHW 输入（UINT8、FP32 或 FP16，数值范围 0–255）和居中补边约定见 [重建说明](docs/stereo_reconstruction.md)，代码流程见 [代码说明](docs/stereo_code_walkthrough.md)。默认使用 TSDF 融合，`fusion_method:=voxel` 可切换体素质心融合；参数与算法见 [TSDF 融合说明](docs/tsdf_fusion.md)。

S2M2 S/M/L 共用 `backend:=s2m2`，通过 `engine_path` 选择模型；`.model` 扩展名可直接用于 TensorRT engine。当前 L engine 输入为 FP32 `[1,3,480,768]`，图内包含归一化，节点会按实际输入类型转换数据。

# MSCKF VIO + 双目三维重建（ROS 2）

基于 ROS 2 Humble 的双目视觉惯性里程计与点云重建系统。在 MSCKF 位姿估计的基础上，增加独立的双目重建节点，支持 **OpenCV SGBM** 和 **IGEV TensorRT** 两种视差后端，完成图像校正、深度估计和世界坐标系下的点云融合。

![EuRoC 双目点云重建效果](docs/images/euroc_stereo_reconstruction.png)

*EuRoC 数据的 RViz 可视化效果：双目重建点云、稀疏特征点与位姿坐标轴。*

![EuRoC 融合点云地图全景](docs/images/euroc_stereo_map_overview.png)

*融合点云地图的俯视效果，可观察室内墙面与场景结构。*

## 功能与处理流程

- **双目 VIO**：跟踪双目特征，融合 IMU，输出位姿、轨迹和稀疏特征点。
- **图像校正**：复用 MSCKF 的相机标定，支持 radtan 和 equidistant 畸变模型。
- **视差与深度**：通过虚函数接口 `StereoMatcher` 切换 SGBM / IGEV，输出浮点像素视差和米制深度。
- **点云融合**：按图像时间戳匹配位姿，将点云变换到 VIO 世界坐标系，通过关键帧筛选和体素融合构建地图。
- **地图管理**：RViz 显示，PCD / PLY 导出，以及清空地图服务。

```text
双目图像 ──→ 特征跟踪 ──┐
IMU ──────────────────┴──→ MSCKF ──→ 位姿 /odom
                                             │
双目图像 ──→ 校正 ──→ SGBM / IGEV ──→ 深度 ──┴──→ 世界点云 ──→ 体素地图
```

重建节点独立于 MSCKF，两个后端共享校正、位姿匹配和点云融合逻辑。当前实现为点云重建，不包含闭环优化、TSDF、网格生成或历史点云随优化位姿重新融合；地图会受到 VIO 漂移的影响。

## 环境与依赖

| 组件 | 本地验证环境 |
| --- | --- |
| 系统 | Ubuntu 22.04 / ROS 2 Humble |
| 语言 | C++17 / Python 3.10 |
| 基础库 | Eigen 3.4、OpenCV 4.5、PCL 1.12、Boost、SuiteSparse |
| IGEV 可选依赖 | TensorRT 10.x、CUDA Runtime、NVIDIA GPU |
| IGEV 本地验证 | TensorRT 10.14 / CUDA 13.1 / GTX 1650 Ti |

SGBM 和 VIO 无需 TensorRT / CUDA。以下命令假设仓库位于 `~/ros_ws/msckf_vio/src/msckf_vio`，工作空间根目录为 `~/ros_ws/msckf_vio`。

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws/msckf_vio

sudo apt install -y libeigen3-dev libopencv-dev libpcl-dev libboost-dev libsuitesparse-dev
rosdep install --from-paths src --ignore-src -r -y
```

## 编译

### SGBM / 仅运行 VIO

```bash
cd ~/ros_ws/msckf_vio
source /opt/ros/humble/setup.bash

CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --packages-select msckf_vio \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DMSCKF_ENABLE_TENSORRT=OFF
source install/setup.bash
```

### 启用 IGEV TensorRT

将 SDK 路径替换为本机安装位置。该配置同时保留 SGBM 后端。

```bash
cd ~/ros_ws/msckf_vio
source /opt/ros/humble/setup.bash

CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --packages-select msckf_vio \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DMSCKF_ENABLE_TENSORRT=ON \
  -DTensorRT_ROOT=/home/ros/lib/TensorRT-10.14 \
  -DCUDA_ROOT=/usr/local/cuda-13.1
source install/setup.bash
```

模型文件需要自行准备。将转换完成的 engine 放在 `src/msckf_vio/weights/model.engine`，或在启动时指定绝对路径；权重不会随软件包安装，也不纳入 Git。

当前 engine 的接口为：

| 张量 | 类型 | 形状 |
| --- | --- | --- |
| `left` | FP32 | `[1, 3, 480, 768]` |
| `right` | FP32 | `[1, 3, 480, 768]` |
| `disparity` | FP32 | `[1, 1, 480, 768]` |

对 EuRoC 的 `752×480` 图像，后端自动在右侧补齐至 `768×480`，推理后裁回原尺寸，保留像素视差单位。当前导出图包含归一化，节点输入使用 RGB、NCHW、`[0,255]` 数据；更换模型时需核对预处理约定。详细接口见[双目重建说明](docs/stereo_reconstruction.md#igev-tensorrt-backend)。

## 运行 EuRoC

每个终端都需要加载工作空间，并使用相同的 `ROS_DOMAIN_ID`。以下启动方式会同时运行图像处理器、MSCKF、双目重建和 RViz。

### 1. 启动节点

**IGEV TensorRT：**

```bash
cd ~/ros_ws/msckf_vio
source install/setup.bash
export ROS_DOMAIN_ID=42

ros2 launch msckf_vio stereo_reconstruction_euroc.launch.py \
  backend:=igev \
  engine_path:="$(pwd)/src/msckf_vio/weights/model.engine"
```

**SGBM：**

```bash
source ~/ros_ws/msckf_vio/install/setup.bash
export ROS_DOMAIN_ID=42

ros2 launch msckf_vio stereo_reconstruction_euroc.launch.py backend:=sgbm
```

两种方式选择其一。如果已有 MSCKF 在运行，追加 `start_vio:=false`，并确保标定、话题、时钟和 ROS 域一致；已有 RViz 时可追加 `rviz:=false`。

### 2. 播放数据

在另一个终端执行，将数据目录替换为自己的 ROS 2 bag 路径：

```bash
source ~/ros_ws/msckf_vio/install/setup.bash
export ROS_DOMAIN_ID=42

ros2 bag play /path/to/V2_01_easy_ros2 \
  --clock --rate 0.2 \
  --topics /imu0 /cam0/image_raw /cam1/image_raw
```

重建 launch 默认启用仿真时间，因此播放时需要 `--clock`。GTX 1650 Ti 上使用 IGEV 建议从 `--rate 0.2` 开始；SGBM 可先用 `--rate 0.5`，再按实际负载调整。推理跟不上输入时，重建队列会丢弃部分帧。

MSCKF 使用最初 200 条 IMU 消息初始化重力方向和偏置，开始时传感器应保持静止。重建在初始化完成且取得匹配位姿后开始。

EuRoC 原始 ROS 1 bag 需要先转换成 ROS 2 格式。播放时仅选择上述三个输入话题，可避开数据集中的自定义消息依赖。所有发布者和订阅者应使用同一个 ROS 域，避免与其他系统的 `/odom`、`/tf` 冲突。

### 3. 查看与导出

RViz 使用项目自带的重建配置，融合地图话题为 `/stereo/map`。

```bash
# 保存地图，默认路径 /tmp/stereo_map.pcd
ros2 service call /stereo/save_map std_srvs/srv/Trigger '{}'

# 改为导出 PLY，再调用保存服务
ros2 param set /stereo_reconstruction save_map_path /tmp/stereo_map.ply
ros2 service call /stereo/save_map std_srvs/srv/Trigger '{}'

# 清空地图、位姿缓存和待处理图像，不重置 MSCKF
ros2 service call /stereo/clear_map std_srvs/srv/Trigger '{}'
```

导出包含米制 XYZ 和 RGB 的 ASCII PCD / PLY，目录需事先存在，已有同名文件会被覆盖。EuRoC 为灰度图，点云颜色也是灰度。

## 话题与参数

| 话题 | 内容 |
| --- | --- |
| `/imu0` | IMU 输入，`sensor_msgs/Imu` |
| `/cam0/image_raw`、`/cam1/image_raw` | 同步双目输入，`sensor_msgs/Image` |
| `/odom` | MSCKF 位姿，`nav_msgs/Odometry` |
| `/stereo/left/image_rect`、`/stereo/right/image_rect` | 校正后的双目图像 |
| `/stereo/disparity` | `32FC1` 像素视差，无效值为 NaN |
| `/stereo/depth` | `32FC1` 深度，单位米，无效值为 NaN |
| `/stereo/points` | 当前帧点云，左校正相机坐标系 |
| `/stereo/points_world` | 当前帧点云，VIO 世界坐标系 |
| `/stereo/map` | 融合后的体素点云地图，VIO 世界坐标系 |

图像与单帧点云按订阅需求生成，使用 best-effort QoS；地图使用 reliable / transient-local QoS，可供新订阅者获取最近一次地图。

| 文件 | 用途 |
| --- | --- |
| [EuRoC VIO 参数](config/msckf_vio_euroc_params.yaml) | 相机和 IMU 标定、特征跟踪、滤波配置 |
| [重建参数](config/stereo_reconstruction_euroc_params.yaml) | 后端、深度范围、关键帧、体素和地图导出配置 |
| [重建详细说明](docs/stereo_reconstruction.md) | 坐标变换、时间同步、推理约定和扩展接口 |

常用重建参数包括 `backend`、`min_depth` / `max_depth`、`pixel_stride`、`voxel_size`、`max_voxels`、`keyframe_translation` 和 `keyframe_rotation_deg`。除 `save_map_path` 外，修改后应重启节点。

重建 launch 从 VIO 参数文件读取同一份几何标定。双目图像必须时间戳一致，输入尺寸必须匹配标定；换相机时需要更新内参、畸变、双目外参和 IMU 外参。矩阵参数使用 16 个按行排列的浮点数，避免混用 `0` 和 `0.0`。

## 单独运行 VIO 与轨迹评估

不需要稠密重建时，可单独启动 MSCKF：

```bash
ros2 launch msckf_vio msckf_vio_euroc.launch.py use_sim_time:=true
```

同样使用 `--clock` 播放数据。其他入口包括 EuRoC 在线外参估计配置 `msckf_vio_euroc_noextrinsics.launch.py` 和 UPenn fast flight 配置 `msckf_vio_fla.launch.py`。

`scripts/` 提供轨迹记录、EuRoC 真值转换和 evo 辅助脚本：

```bash
python3 src/msckf_vio/scripts/record_odom.py /tmp/est.txt
python3 src/msckf_vio/scripts/gt_asl_to_tum.py \
  /path/to/sequence/mav0/state_groundtruth_estimate0/data.csv /tmp/gt.txt

# 需另行安装 evo；VIO 使用 SE(3) 对齐评估米制轨迹
# 轨迹记录完成后执行
evo_ape tum /tmp/gt.txt /tmp/est.txt -a
```

## 验证

```bash
colcon build --packages-select msckf_vio --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select msckf_vio --return-code-on-test-failure
colcon test-result --verbose
```

测试覆盖 SGBM 视差尺度、深度与坐标变换、位姿匹配、体素融合、地图导出，以及 IGEV 的输入布局、补边、FP16 转换和无效输出处理。

本地 EuRoC `V2_01_easy` 短序列集成验证结果：

| 后端 | 数据时长 / 播放倍率 | 深度帧数 | 融合地图点数 |
| --- | --- | --- | --- |
| SGBM | 前 15 秒 / 0.5 倍速 | 279 | 195,987 |
| IGEV TensorRT | 前 8 秒 / 0.25 倍速 | 122 | 41,851 |

两次验证均通过地图保存与清空检查。数据时长和运行负载不同，这些数字用于确认链路可运行，不用于比较后端精度或宣称实时性能。复现脚本见 [test/stereo_ros_smoke.py](test/stereo_ros_smoke.py)。

## 常见问题

- **IGEV 启动失败**：确认编译时启用了 `MSCKF_ENABLE_TENSORRT`，engine 文件非空，运行环境兼容，张量名称与配置一致。失败会报错退出，不会自动切换到 SGBM。
- **有位姿但没有地图**：检查双目时间戳、标定、`/clock` 和 `/odom`，以及是否存在对应时刻的位姿。挂接已有 VIO 时要使用它实际采用的标定文件。
- **动作快时跟踪点减少或 VIO 退化**：检查曝光、运动模糊、标定和相机 / IMU 同步，结合特征网格、检测阈值及跟踪统计排查。稠密重建不会反向修正 MSCKF 位姿。
- **地图错位或漂移**：先检查 VIO 轨迹及外参。重置 MSCKF 世界原点后，也应清空重建地图。
- **RViz 看不到图像或单帧点云**：检查话题和 QoS，图像 / 单帧点云订阅使用 best-effort；地图订阅使用 reliable / transient-local。

## 来源与许可

本项目基于 [KumarRobotics/msckf_vio](https://github.com/KumarRobotics/msckf_vio) 移植至 ROS 2，并扩展双目深度估计与点云融合。

- MSCKF 双目 VIO 论文：[Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight](https://arxiv.org/abs/1712.00036)。
- 项目代码遵循 [Penn Software License](LICENSE.txt)；第三方代码和模型遵循各自许可。

# 局部点云 BA 与 TSDF 子图重建

任务书：[docs/plan/point_cloud_ba.md](plan/point_cloud_ba.md)。本功能默认启用，
使用 Ceres 2.0；输入频率与 MSCKF 的估计链路保持原有配置。

## 启动

```bash
cd /home/ros/rosws/msckf_stereo_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
source src/stereo_recon/scripts/setup_wsl_env.sh
export ROS_DOMAIN_ID=42
ros2 launch msckf_vio stereo_reconstruction_euroc.launch.py \
  backend:=s2m2 fusion_method:=tsdf point_cloud_ba:=true \
  engine_path:=/home/ros/rosws/msckf_stereo_ws/src/stereo_recon/weights/S2M2_L_768_480_v2_torch26.model \
  archive_directory:=/tmp/stereo_ba_archive
```

已有 VIO 时改用 `ros2 launch stereo_recon stereo_recon.launch.py`，并传入相同参数
及正确的标定文件。真实相机设置 `use_sim_time:=false`。

`archive_directory` 可省略；默认只保留活动子图深度，不写磁盘。
开启归档会创建独立 `session_<时间>` 子目录，启动日志打印实际路径。
应选择容量充足的位置：归档不自动删除历史文件，也没有磁盘容量配额。

关闭 BA 使用 `point_cloud_ba:=false`，恢复原固定 VIO 位姿融合流程。
`fusion_method:=voxel` 且没有显式指定 BA 时，launch 自动关闭 BA。
显式组合 `voxel + point_cloud_ba:=true` 会因不支持而报错。
离线公平对比应使用下面的同一子图流程，避免把融合策略差异混入 BA 效果。

## 处理流程

1. 图像与里程计回调进入缓存；推理线程按图像采集时间查找位姿。
2. BA 模式在推理前按位移/转角筛选关键帧；非候选帧不执行 S2M2-L。
3. 候选帧进行校正、视差推理、质量过滤和深度投影，再进入有界建图队列。
4. 独立建图线程保留深度、颜色、相机内参、基线和 VIO/建图位姿。
5. 在最近窗口内建立共享平面，执行 BA，检查结果；失败保留输入位姿。
6. 位姿发生改变时，从活动子图的完整深度重新融合 TSDF；否则增量融合新帧。
7. 完成后提交地图、优化轨迹和当前点云快照。推理与建图队列满时均丢最旧任务，
   优先追随新数据；不会降低相机、IMU 或 MSCKF 发布频率，但会降低实际建图频率。

BA 模式下 `/stereo/depth` 和当前点云只对应通过初筛的帧。拒绝 BA 不等于拒绝建图：
该关键帧仍按当前建图初值融合，状态话题说明拒绝原因。

## 代价与关联

采用显式平面参数与 SE(3) 位姿联合优化。单位四元数和单位平面法向通过
Ceres 局部参数化保持约束；固定窗口第一帧，使用 DENSE_SCHUR 消去平面增量。
这是点到平面最小二乘 BA，不是直接复刻 BALM2 的特征值解析二阶求解器。
首版只采用平面特征，没有空间直线特征、视觉重投影因子或 IMU 预积分因子。

对属于同一平面、同一帧的采样点，构造相机坐标下的加权齐次二阶矩：

```
C_ij = sum_k w_ijk * [p_ijk; 1] * [p_ijk; 1]^T
C_ij = S_ij^T * S_ij
r_ij = S_ij * [R_i^T * n_j; n_j^T * t_i + d_j]
```

`||r_ij||²` 等于该点簇加权点到平面误差。迭代时每个帧—平面点簇只计算
4 维残差。Huber 损失施加在整个点簇上，不能解释成逐像素独立鲁棒损失。
点的权重按初始平面方向投影测量协方差，求解内固定，并按每帧样本数和该平面
的支撑帧数归一化。这是一种平衡约束数量的工程权重，不是独立同分布似然。

采样过滤无效深度及其边界，按网格均匀采样并限制每帧最多 5000 点。
协方差由 `(u,v,disparity)` 一阶传播，像素标准差取 0.5 像素，视差标准差
由参数指定，并叠加米制噪声下限。S2M2 的 confidence/visibility 仍用于匹配器
内的有效性过滤，没有把网络 confidence 直接当作概率或方差。

使用当前建图位姿将样本分配到粗体素，再递归八叉划分。平面需满足点数、
横向展开范围、最小特征值比例、厚度与支撑帧数要求；同一帧每个平面最多取
64 个样本。每次新窗口重新建立关联，单次求解内关联保持固定。

接受检查包括：足够的共享平面、平面法向二阶矩的最小特征值、每帧至少三个
平面支撑、关联图与首帧连接、求解可用、几何/总目标不增、修正幅度及相对原始
VIO 的累计偏差。法向检查是保守退化筛选，不是完整的后验 Hessian 可观测性分析。

邻接 VIO 相对运动因子为保守正则。没有使用两个边际协方差简单相减来构造相对
协方差；既未恢复 MSCKF 的跨帧相关性，也不把 BA 位姿回灌滤波器。

## 子图、坐标与清空

默认活动子图最多 20 帧，BA 窗口 10 帧。达到上限后，前 16 帧按已接受位姿
构建冻结 TSDF，提取表面并压缩到冻结表面地图；最后 4 帧保留深度，连接下一子图。
重叠帧不会同时进入冻结 TSDF，因此后续优化不会修改已经冻结的同一帧观测。

活动子图可重新融合；冻结部分仅保留经过 TSDF 提取的表面样本，不保留完整距离场。
输出合并活动/冻结表面并按 `voxel_size` 压缩，因此不等同于单个全局 TSDF 的零交叉。
首版不跨冻结子图进行 BA，不支持回环、PGO 或冻结表面变形。
保留原始归档后，可以离线重新处理全部关键帧，但离线入口目前仍是局部 BA。

`max_voxels` 分别约束活动距离场、冻结表面存储和发布表面数，**不是整个进程的
统一内存预算**。冻结时还会临时构建一个 TSDF，重新融合时会同时存在新旧活动场。
满容量后停止新增元素但继续更新已有元素；发布合并优先保留活动表面。

地图参考系默认为 `stereo_map`，原始 VIO 参考系仍由标定配置决定（通常 `world`）。
节点随里程计时间戳发布 `stereo_map -> world` 的最新修正，适用于当前实时 TF；
历史帧分别使用各自优化位姿，不能使用当前修正去替代历史 BA 结果。
TF 修正会随下一条里程计消息更新，存在异步处理延迟。
回看历史原始 VIO Path 时，RViz 的刚体 TF 变换也不等于逐帧优化。

清空服务和图像时间倒退会清空队列与可见快照、复位修正并增加 generation。
在途任务在提交前失效；建图线程处理新 generation 时清空内部子图。
停止会等待工作线程退出；Ceres 时间预算不包含关联、归档、表面提取和 TSDF 重建，
也不是严格实时保证。取消在关键步骤之间检查，不中断正在进行的一次 TSDF 融合。

## 话题、服务与诊断

| 接口 | 含义 |
|---|---|
| `/stereo/map` | 最新完整融合表面快照，BA 模式参考系为 stereo_map |
| `/stereo/points_world` | 最近完成建图关键帧的优化后点云，参考系为 stereo_map |
| `/stereo/optimized_path` | 校正左相机关键帧路径，最近最多 20000 帧 |
| `/stereo/optimized_odometry` | 最近完成建图关键帧的校正左相机位姿，非连续高频里程计 |
| `/stereo/ba/status` | 结果接受原因、平面数、代价、BA/建图耗时、容量与队列丢帧 |
| `/stereo/save_map` | 按 save_map_path 保存当前表面 PCD/PLY |
| `/stereo/save_trajectory` | 按 save_trajectory_path 保存关键帧 TUM 轨迹 |
| `/stereo/clear_map` | 清空当前重建状态 |

优化里程计不提供校准后的后验协方差或速度，其协方差默认零值不能用于下游
独立测量融合。RViz 自动选择对应参考系，新增 Optimized Stereo Path 显示。

```bash
ros2 topic echo /stereo/ba/status
ros2 service call /stereo/save_trajectory std_srvs/srv/Trigger '{}'
```

常见状态：`waiting_for_window`、`insufficient_shared_planes`、
`degenerate_plane_directions`、`disconnected_or_unsupported_frame`、
`solver_or_cost_rejected`、`pose_update_rejected` 和 `accepted`。
持续退化时应先检查有效深度、跨帧重叠和平面方向，不能只放松阈值强行接受。

## 离线处理

每个归档帧包含 float32 TIFF 深度、PNG 颜色和 YAML 元数据（内参、基线、原始
校正左相机 VIO 位姿、精确纳秒时间戳、序列号）。图像写完后才提交元数据文件。
残留临时元数据不会被回放。归档的是到达建图线程的原始关键帧，未包含队列中被丢帧。
归档参数自动写入 `mapping_parameters.yaml`；清空不会覆盖旧文件，sequence 区分重置段。

```bash
ros2 run stereo_recon stereo_ba_offline \
  /tmp/stereo_ba_archive/session_<实际目录名> /tmp/stereo_ba_result

# 使用同一归档和子图策略、固定 VIO 位姿，作为对照
ros2 run stereo_recon stereo_ba_offline \
  /tmp/stereo_ba_archive/session_<实际目录名> /tmp/stereo_fixed_result --no-ba
```

离线自动读取归档参数。可用 `--params` 指定本包 ROS YAML 替换参数，或通过
`--window-size`、`--max-plane-rms`、`--disparity-sigma`、`--voxel-size`、
`--solver-time` 覆盖对应配置。`--max-frames` 限制处理帧数。一个归档包含多个
重置段时，需用 `--sequence N` 选择实际序列，避免混合不连续的轨迹。

输出为 `.ply`、优化 `.tum`、原始 `.vio.tum` 和逐帧 `.metrics.csv`。
两条 TUM 轨迹均对应校正左相机；与机体/IMU 真值比较时需统一外参和参考系。
离线保留整个所选段的轨迹，在线轨迹受 `max_trajectory_poses` 限制。
在线优化位姿可通过保存轨迹服务导出；原始归档可重新计算，未保存可恢复的在线
优化器状态。在线/离线可能因机器负载和求解时间预算而得到略有差异的结果。

## 参数与资源预算

完整参数在 `config/stereo_reconstruction_euroc_params.yaml` 的 `ba` 下，启动时读取。

| 参数 | 默认值 | 含义 |
|---|---|---|
| window_size / submap_keyframes / overlap_keyframes | 10 / 20 / 4 | 窗口、子图、重叠帧数 |
| max_pending_keyframes | 2 | 建图队列容量 |
| optimize_every_n_keyframes | 1 | 每 N 个关键帧触发一次 |
| root_voxel_size / max_voxel_depth | 0.8 m / 2 | 自适应平面关联体素 |
| max_plane_rms / plane_eigenvalue_ratio | 0.05 m / 0.1 | 平面厚度与特征值比例 |
| min_frames_per_plane / min_planes | 3 / 12 | 跨帧支撑与特征数量 |
| min_normal_eigenvalue | 0.01 | 归一化法向二阶矩最小特征值门限 |
| disparity_sigma / noise_floor | 0.5 px / 0.005 m | 深度噪声假设与下限 |
| vio_translation_sigma / vio_rotation_sigma_deg | 0.15 m / 5° | 相邻帧运动正则尺度 |
| max_translation_update / max_rotation_update_deg | 0.2 m / 5° | 单次修正上限 |
| max_vio_translation / max_vio_rotation_deg | 1.0 m / 15° | 相对原始 VIO 的偏差上限 |
| max_solver_time_sec / max_iterations / num_threads | 0.15 / 15 / 1 | Ceres 求解预算 |

这些是工程起始值，未经过当前设备的精度/全速性能标定。深度有系统偏差、
动态物体或严重失配时，BA 也可能收敛到错误解，筛选与正则不能提供绝对保证。

## 编译与代码规范

新增依赖：`libceres-dev`、`libyaml-cpp-dev`。本机已有 Ceres 2.0 和 yaml-cpp 0.7。
根目录 `.clang-format` 基于 Google 风格，列宽 80；新增/修改的 C++ 已格式化。

```bash
source /opt/ros/humble/setup.bash
cd /home/ros/rosws/msckf_stereo_ws
CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --packages-select stereo_recon \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DSTEREO_RECON_ENABLE_TENSORRT=ON \
  -DTensorRT_ROOT=/home/ros/lib/TensorRT-11.3.0.99 -DCUDA_ROOT=/usr/local/cuda-12.1
```

本轮完成代码审阅和 Release 编译，未新增或运行测试，未执行数据集精度评估。
编译成功只说明源码及依赖可以构建。后续应在同一归档上比较有/无 BA 的
ATE/RPE、保留观测的一致性、平面厚度、耗时、丢帧与内存，不能仅依据目标下降
或训练内平面变薄判断精度提升。

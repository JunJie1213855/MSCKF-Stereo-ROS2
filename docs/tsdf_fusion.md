# TSDF 融合

当前默认启用局部点云 BA：推理前筛选关键帧，优化后重新融合活动 TSDF 子图。
以下固定 VIO 位姿的单 TSDF 描述对应 `point_cloud_ba:=false`；
启用 BA 时的子图、坐标、表面合并和资源限制见 [点云 BA](point_cloud_ba.md)。

`fusion_method` 与匹配 `backend` 独立。SGBM、IGEV、S2M2 的深度均可进入 TSDF；默认启用 TSDF，`fusion_method:=voxel` 选择原体素质心融合。

```bash
source /opt/ros/humble/setup.bash
cd /home/ros/rosws/msckf_stereo_ws
source install/setup.bash
source src/stereo_recon/scripts/setup_wsl_env.sh
export ROS_DOMAIN_ID=42
ros2 launch msckf_vio stereo_reconstruction_euroc.launch.py \
  backend:=s2m2 fusion_method:=tsdf \
  engine_path:=/home/ros/rosws/msckf_stereo_ws/src/stereo_recon/weights/S2M2_L_768_480_v2_torch26.model
```

已有 VIO 时使用 `ros2 launch stereo_recon stereo_recon.launch.py` 并传入同样参数，或在组合 launch 加 `start_vio:=false`。真实相机使用 `use_sim_time:=false`。

## 算法与坐标

输入为校正左相机的光轴深度 Z（米），无效像素为 NaN。内参取 stereoRectify 的 P1 左上角 3×3；位姿为图像采集时间匹配的 `T_world_body * T_body_left_rect`。双目边界、深度范围和 S2M2 质量过滤在生成深度时处理。

体素中心变换回校正相机并投影到深度图，以最近邻采样避免混合前后景。残差 `sdf = depth(u,v) - voxel_camera.z`；正值表示表面之前，负值表示之后。仅融合 `sdf >= -mu`，归一化为 `min(1, sdf/mu)`；mu 是光轴 Z 截断距离。每关键帧对每体素最多更新一次；权重封顶后使用有界渐进平均。

世界坐标稀疏哈希体素通过 DDA 射线遍历分配表面前后 mu 的窄带。随后对已有体素做可见性投影更新，包括旧表面前方已分配的自由空间。未知区域不视为已知自由空间；颜色只在靠近表面的区域融合。

相邻、已观测且达到最小权重的体素之间发生符号变化时，线性插值提取零交叉表面点。`/stereo/map` 发布表面点，而非体素中心。当前帧 `/stereo/points` 和 `/stereo/points_world` 来自有效深度，可与融合表面对照。

## 参数

参数位于 `config/stereo_reconstruction_euroc_params.yaml`，修改后重启：

| 参数 | 默认值 | 含义 |
|---|---|---|
| fusion_method | tsdf | tsdf 或 voxel |
| voxel_size | 0.05 | 世界体素边长，米 |
| max_voxels | 500000 | 距离场体素上限，不是输出点数 |
| tsdf.truncation_distance | 0.15 | 光轴 Z 截断距离，米，至少为 voxel_size |
| tsdf.integration_stride | 2 | 分配射线的像素步长；更新体素使用完整深度 |
| tsdf.max_weight | 100.0 | 体素权重上限，至少 1 |
| tsdf.min_weight | 1.0 | 提取表面所需权重，不大于 max_weight |

`pixel_stride` 控制当前帧点云和原 voxel 融合的采样，与 TSDF 深度融合独立。达到容量后停止新增体素，已有体素继续更新。CPU TSDF 不占用 S2M2 CUDA stream；计算耗时仍可能使重建队列丢帧，日志单独显示 fusion 耗时。

## 显示与保存

RViz 的 `Fused Stereo Map` 显示 `/stereo/map`，`Current Stereo Cloud` 显示当前帧世界点云。地图沿用 reliable/transient-local QoS。
XYZRGB 消息采用标准字段的 16 字节紧凑布局，减少大点云的 DDS 分片，不减少点数。

```bash
ros2 param set /stereo_reconstruction save_map_path /tmp/stereo_tsdf.ply
ros2 service call /stereo/save_map std_srvs/srv/Trigger '{}'
ros2 service call /stereo/clear_map std_srvs/srv/Trigger '{}'
```

保存 PCD/PLY 表面点云。清空操作删除距离场、权重、颜色、待处理帧、位姿缓存及关键帧状态；时间倒退也触发重置。保存结果不包含可恢复融合的完整 TSDF，也不生成三角网格。

本实现使用固定 VIO 位姿融合，不执行回环、历史位姿重新融合或动态物体分割。采用独立 C++/Eigen/OpenCV 实现，不新增 Open3D/nvblox 运行依赖。TSDF 概念可参考 [Open3D RGB-D integration](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html)。

## 验证

`test/tsdf_map_test.cpp` 覆盖平面、位姿、负坐标、每帧权重、多帧平均、权重上限、无效深度、自由空间更新、容量、清空和输入校验。

```bash
colcon test --packages-select stereo_recon --ctest-args -R test_tsdf_map --output-on-failure
python3 -s src/stereo_recon/test/stereo_ros_smoke.py /path/to/euroc_ros2_bag \
  --fusion-method tsdf --backend s2m2 --engine /absolute/path/to/model.engine \
  --duration 5 --rate 0.2 --domain-id 102 --output /tmp/stereo_tsdf_validation
```

本机短时验证：SGBM + TSDF 在 8 秒传感器数据、0.5 倍播放速率下输出 147 帧深度、140 帧世界点云及 8,593 个地图表面点；S2M2-L + TSDF 在 5 秒传感器数据、0.2 倍播放速率下输出 67 帧深度、58 帧世界点云及 5,724 个表面点。两次保存/清空均通过。这些结果用于确认链路运行，不是精度或全速性能基准。

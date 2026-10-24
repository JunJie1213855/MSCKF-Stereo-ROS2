#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "Eigen/Geometry"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/exact_time.h"
#include "message_filters/synchronizer.h"
#include "nav_msgs/msg/odometry.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/header.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

#include "stereo_recon/stereo/reconstruction.h"
#include "stereo_recon/stereo/tsdf_map.h"
#include "stereo_recon/stereo/stereo_matcher.h"

namespace stereo_recon {
using Image = sensor_msgs::msg::Image;
using Odometry = nav_msgs::msg::Odometry;
using Trigger = std_srvs::srv::Trigger;
using SyncPolicy = message_filters::sync_policies::ExactTime<Image, Image>;

class ReconstructionNode final : public rclcpp::Node {
 public:
  ReconstructionNode() : Node("stereo_reconstruction") {
    // 初始化顺序：先确定标定和几何约定，再创建后端与 ROS 接口，最后启动线程。
    LoadParameters();
    CreateRectifier();
    CreateMatcher();
    CreateRosInterfaces();
    worker_ = std::thread(&ReconstructionNode::RunWorker, this);
    RCLCPP_INFO(
        get_logger(),
        "Stereo reconstruction: %s, %dx%d, pose %s <- %s, exact stereo sync, fusion=%s",
        matcher_->name().c_str(), rectifier_->size().width,
        rectifier_->size().height, fixed_frame_.c_str(), body_frame_.c_str(),
        fusion_method_.c_str());
  }

  ~ReconstructionNode() override { Stop(); }
  void Stop() {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
      RCLCPP_INFO(get_logger(),
                  "Reconstruction stopped: processed=%" PRIu64
                  " keyframes=%" PRIu64 " queue_drops=%" PRIu64
                  " missing_pose=%" PRIu64,
                  processed_.load(), fused_.load(), dropped_.load(),
                  missing_pose_.load());
    }
  }

 private:
  struct Job {
    Image::ConstSharedPtr left, right;
    uint64_t generation;
  };

  int DeclarePositiveInt(const std::string& name, int default_value) {
    const int value = declare_parameter<int>(name, default_value);
    if (value <= 0) {
      throw std::invalid_argument(name + " must be positive");
    }
    return value;
  }

  // 配置只在启动时读取；地图保存路径除外，它可由服务调用时动态读取。
  void LoadParameters() {
    fixed_frame_ = declare_parameter<std::string>("fixed_frame_id", "world");
    body_frame_ = declare_parameter<std::string>("pose_child_frame_id", "odom");
    left_frame_ = declare_parameter<std::string>("left_rect_frame_id",
                                                 "cam0_rect_optical_frame");
    right_frame_ = declare_parameter<std::string>("right_rect_frame_id",
                                                  "cam1_rect_optical_frame");
    if (fixed_frame_.empty() || body_frame_.empty() || left_frame_.empty() ||
        right_frame_.empty() || fixed_frame_ == body_frame_ ||
        left_frame_ == right_frame_ || left_frame_ == body_frame_ ||
        right_frame_ == body_frame_ || left_frame_ == fixed_frame_ ||
        right_frame_ == fixed_frame_) {
      throw std::invalid_argument(
          "World, body and rectified camera frame names must be distinct and "
          "nonempty");
    }
    min_depth_ = declare_parameter<double>("min_depth", 0.3);
    max_depth_ = declare_parameter<double>("max_depth", 10.0);
    pixel_stride_ = DeclarePositiveInt("pixel_stride", 2);
    queue_capacity_ = DeclarePositiveInt("max_pending_frames", 4);
    pose_buffer_ = std::make_unique<PoseBuffer>(
        DeclarePositiveInt("pose_cache_size", 400));
    const double tolerance =
        declare_parameter<double>("pose_tolerance_sec", 0.0);
    pose_wait_sec_ = declare_parameter<double>("pose_wait_sec", 0.3);
    map_publish_period_sec_ =
        declare_parameter<double>("map_publish_period_sec", 0.5);
    if (!std::isfinite(min_depth_) || !std::isfinite(max_depth_) ||
        min_depth_ <= 0 || max_depth_ <= min_depth_ ||
        !std::isfinite(tolerance) || tolerance < 0 || tolerance > 1 ||
        !std::isfinite(pose_wait_sec_) || pose_wait_sec_ < 0 ||
        !std::isfinite(map_publish_period_sec_) ||
        map_publish_period_sec_ <= 0) {
      throw std::invalid_argument(
          "Invalid depth limits, pose timing or map publish period");
    }
    pose_tolerance_ns_ = static_cast<int64_t>(tolerance * 1e9);
    const double voxel_size = declare_parameter<double>("voxel_size", 0.05);
    const size_t max_voxels = DeclarePositiveInt("max_voxels", 500000);
    fusion_method_ = declare_parameter<std::string>("fusion_method", "tsdf");
    TsdfParameters tsdf;
    tsdf.voxel_size = voxel_size;
    tsdf.max_voxels = max_voxels;
    tsdf.truncation_distance = declare_parameter<double>("tsdf.truncation_distance", 0.15);
    tsdf.max_weight = static_cast<float>(declare_parameter<double>("tsdf.max_weight", 100.0));
    tsdf.min_weight = static_cast<float>(declare_parameter<double>("tsdf.min_weight", 1.0));
    tsdf.integration_stride = DeclarePositiveInt("tsdf.integration_stride", 2);
    if (fusion_method_ == "tsdf") {
      tsdf_map_ = std::make_unique<TsdfMap>(tsdf);
    } else if (fusion_method_ == "voxel") {
      map_ = std::make_unique<VoxelMap>(voxel_size, max_voxels);
    } else {
      throw std::invalid_argument("fusion_method must be tsdf or voxel");
    }
    keyframes_ = std::make_unique<KeyframeSelector>(
        declare_parameter<double>("keyframe_translation", 0.1),
        declare_parameter<double>("keyframe_rotation_deg", 5.0) *
            std::acos(-1.0) / 180.0);
    declare_parameter<std::string>("save_map_path", "/tmp/stereo_map.pcd");
  }

  // t_a_b 表示把 B 坐标变换到
  // A。这里从相机/IMU/机体外参得到原左目到机体的变换。
  void CreateRectifier() {
    const auto left = ReadCamera("cam0");
    const auto right = ReadCamera("cam1");
    const auto t_left_imu = ReadTransform("cam0/T_cam_imu");
    const auto t_imu_body = ReadTransform("T_imu_body");
    rectifier_ = std::make_unique<StereoRectifier>(
        left, right, ReadTransform("cam1/T_cn_cnm1"),
        t_imu_body.inverse() * t_left_imu.inverse(),
        declare_parameter<double>("rectification_alpha", 0.0));
  }

  // 三种后端都遵守同一接口：原图分辨率、浮点像素视差、无效值 NaN。
  void CreateMatcher() {
    SgbmParameters p;
    p.min_disparity =
        declare_parameter<int>("sgbm.min_disparity", p.min_disparity);
    p.num_disparities =
        declare_parameter<int>("sgbm.num_disparities", p.num_disparities);
    p.block_size = declare_parameter<int>("sgbm.block_size", p.block_size);
    p.uniqueness_ratio =
        declare_parameter<int>("sgbm.uniqueness_ratio", p.uniqueness_ratio);
    p.speckle_window_size = declare_parameter<int>("sgbm.speckle_window_size",
                                                   p.speckle_window_size);
    p.speckle_range =
        declare_parameter<int>("sgbm.speckle_range", p.speckle_range);
    p.disp12_max_diff =
        declare_parameter<int>("sgbm.disp12_max_diff", p.disp12_max_diff);
    p.pre_filter_cap =
        declare_parameter<int>("sgbm.pre_filter_cap", p.pre_filter_cap);
    IgevParameters igev;
    igev.engine_path = declare_parameter<std::string>("igev.engine_path", "");
    igev.left_input_name = declare_parameter<std::string>(
        "igev.left_input_name", igev.left_input_name);
    igev.right_input_name = declare_parameter<std::string>(
        "igev.right_input_name", igev.right_input_name);
    igev.output_name =
        declare_parameter<std::string>("igev.output_name", igev.output_name);
    igev.input_width =
        declare_parameter<int>("igev.input_width", igev.input_width);
    igev.input_height =
        declare_parameter<int>("igev.input_height", igev.input_height);
    igev.padding_multiple =
        declare_parameter<int>("igev.padding_multiple", igev.padding_multiple);
    igev.padding_mode =
        declare_parameter<std::string>("igev.padding_mode", igev.padding_mode);
    igev.device_id = declare_parameter<int>("igev.device_id", igev.device_id);
    S2m2Parameters s2m2;
    s2m2.engine_path = declare_parameter<std::string>("s2m2.engine_path", "");
    s2m2.left_input_name = declare_parameter<std::string>(
        "s2m2.left_input_name", s2m2.left_input_name);
    s2m2.right_input_name = declare_parameter<std::string>(
        "s2m2.right_input_name", s2m2.right_input_name);
    s2m2.output_name =
        declare_parameter<std::string>("s2m2.output_name", s2m2.output_name);
    s2m2.visibility_output_name = declare_parameter<std::string>(
        "s2m2.visibility_output_name", s2m2.visibility_output_name);
    s2m2.confidence_output_name = declare_parameter<std::string>(
        "s2m2.confidence_output_name", s2m2.confidence_output_name);
    s2m2.input_width =
        declare_parameter<int>("s2m2.input_width", s2m2.input_width);
    s2m2.input_height =
        declare_parameter<int>("s2m2.input_height", s2m2.input_height);
    s2m2.device_id = declare_parameter<int>("s2m2.device_id", s2m2.device_id);
    s2m2.min_confidence =
        declare_parameter<double>("s2m2.min_confidence", s2m2.min_confidence);
    s2m2.min_visibility =
        declare_parameter<double>("s2m2.min_visibility", s2m2.min_visibility);
    matcher_ = CreateStereoMatcher(
        declare_parameter<std::string>("backend", "sgbm"), p, igev, s2m2);
    matcher_->ValidateInputSize(rectifier_->size());
    // Bound OpenCV's CPU pool so dense stereo leaves CPU time for VIO.
    cv::setNumThreads(DeclarePositiveInt("opencv_threads", 2));
  }

  // 图像回调只排队；推理在独立线程完成，避免阻塞 IMU 和里程计回调。
  void CreateRosInterfaces() {
    const auto image_qos = rclcpp::SensorDataQoS().keep_last(10);
    left_pub_ = create_publisher<Image>("stereo/left/image_rect", image_qos);
    right_pub_ = create_publisher<Image>("stereo/right/image_rect", image_qos);
    disparity_pub_ = create_publisher<Image>("stereo/disparity", image_qos);
    depth_pub_ = create_publisher<Image>("stereo/depth", image_qos);
    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "stereo/left/camera_info", image_qos);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        "stereo/right/camera_info", image_qos);
    points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "stereo/points", image_qos);
    world_points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "stereo/points_world", image_qos);
    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "stereo/map", rclcpp::QoS(1).reliable().transient_local());
    static_tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
    static_tf_->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{
        MakeCameraTransform(left_frame_, rectifier_->body_from_left_rect()),
        MakeCameraTransform(right_frame_, rectifier_->body_from_right_rect())});
    left_sub_.subscribe(this, "cam0_image", image_qos.get_rmw_qos_profile());
    right_sub_.subscribe(this, "cam1_image", image_qos.get_rmw_qos_profile());
    synchronizer_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(DeclarePositiveInt("stereo_sync_queue_size", 20)), left_sub_,
        right_sub_);
    synchronizer_->registerCallback(std::bind(&ReconstructionNode::OnStereo,
                                              this, std::placeholders::_1,
                                              std::placeholders::_2));
    odom_sub_ =
        create_subscription<Odometry>("odom", rclcpp::QoS(200).best_effort(),
                                      std::bind(&ReconstructionNode::OnOdometry,
                                                this, std::placeholders::_1));
    service_group_ =
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    clear_service_ = create_service<Trigger>(
        "stereo/clear_map",
        [this](const std::shared_ptr<Trigger::Request>,
               std::shared_ptr<Trigger::Response> response) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          ResetLocked();
          response->success = true;
          response->message =
              "Cleared map, keyframes, pose cache and pending frames";
        },
        rmw_qos_profile_services_default, service_group_);
    save_service_ = create_service<Trigger>(
        "stereo/save_map",
        [this](const std::shared_ptr<Trigger::Request>,
               std::shared_ptr<Trigger::Response> response) {
          std::vector<ColorPoint> points;
          {
            std::lock_guard<std::mutex> lock(map_mutex_);
            points = MapPointsLocked();
          }
          if (points.empty()) {
            response->success = false;
            response->message = "Map is empty; no file was written";
            return;
          }
          try {
            const auto path = get_parameter("save_map_path").as_string();
            SavePointCloud(path, points);
            response->success = true;
            response->message =
                "Saved " + std::to_string(points.size()) + " points to " + path;
          } catch (const std::exception& error) {
            response->success = false;
            response->message = error.what();
          }
        },
        rmw_qos_profile_services_default, service_group_);
    map_stamp_ = now();
    map_timer_ = create_wall_timer(
        std::chrono::duration<double>(map_publish_period_sec_),
        std::bind(&ReconstructionNode::PublishMap, this));
  }

  CameraCalibration ReadCamera(const std::string& prefix) {
    const auto resolution = declare_parameter<std::vector<int64_t>>(
        prefix + "/resolution", std::vector<int64_t>{});
    const auto k = declare_parameter<std::vector<double>>(
        prefix + "/intrinsics", std::vector<double>{});
    const auto d = declare_parameter<std::vector<double>>(
        prefix + "/distortion_coeffs", std::vector<double>{});
    if (resolution.size() != 2 || k.size() != 4 || d.size() != 4 ||
        resolution[0] <= 0 || resolution[1] <= 0 || resolution[0] > 20000 ||
        resolution[1] > 20000) {
      throw std::invalid_argument("Missing or invalid calibration for " +
                                  prefix);
    }
    return {
        cv::Size(static_cast<int>(resolution[0]),
                 static_cast<int>(resolution[1])),
        cv::Matx33d(k[0], 0, k[2], 0, k[1], k[3], 0, 0, 1), cv::Mat(d).clone(),
        declare_parameter<std::string>(prefix + "/distortion_model", "radtan")};
  }

  Eigen::Isometry3d ReadTransform(const std::string& name) {
    return TransformFromRowMajor(
        declare_parameter<std::vector<double>>(name, std::vector<double>{}));
  }

  geometry_msgs::msg::TransformStamped MakeCameraTransform(
      const std::string& frame, const Eigen::Isometry3d& transform) {
    geometry_msgs::msg::TransformStamped message;
    message.header.stamp = now();
    message.header.frame_id = body_frame_;
    message.child_frame_id = frame;
    message.transform.translation.x = transform.translation().x();
    message.transform.translation.y = transform.translation().y();
    message.transform.translation.z = transform.translation().z();
    const Eigen::Quaterniond q(transform.linear());
    message.transform.rotation.x = q.x();
    message.transform.rotation.y = q.y();
    message.transform.rotation.z = q.z();
    message.transform.rotation.w = q.w();
    return message;
  }

  // 调用者必须持有 state_mutex_；统一锁顺序为 state→map，反向获取会死锁。
  // generation 让正在 GPU 上计算的旧任务失效，清空后不会把旧结果写回地图。
  void ResetLocked() {
    ++generation_;
    jobs_.clear();
    pose_buffer_->Clear();
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      if (tsdf_map_) tsdf_map_->Clear();
      else map_->Clear();
      keyframes_->Clear();
      map_stamp_ = now();
      map_dirty_ = true;
    }
    condition_.notify_all();
  }

  void OnStereo(const Image::ConstSharedPtr& left,
                const Image::ConstSharedPtr& right) {
    const int64_t stamp = rclcpp::Time(left->header.stamp).nanoseconds();
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (last_image_stamp_ && stamp < *last_image_stamp_) {
      ResetLocked();
      RCLCPP_WARN(get_logger(),
                  "Image time moved backwards; reconstruction state cleared");
    }
    last_image_stamp_ = stamp;
    // 推理落后时丢弃最旧任务，保留新图像；队列有界，避免延迟持续增长。
    while (jobs_.size() >= queue_capacity_) {
      jobs_.pop_front();
      ++dropped_;
    }
    jobs_.push_back({left, right, generation_.load()});
    condition_.notify_all();
  }

  void OnOdometry(const Odometry::ConstSharedPtr message) {
    if (message->header.frame_id != fixed_frame_ ||
        message->child_frame_id != body_frame_) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Rejecting odometry frames %s <- %s; expected %s <- %s",
          message->header.frame_id.c_str(), message->child_frame_id.c_str(),
          fixed_frame_.c_str(), body_frame_.c_str());
      return;
    }
    const auto& p = message->pose.pose;
    Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y,
                         p.orientation.z);
    const Eigen::Vector3d position(p.position.x, p.position.y, p.position.z);
    if (!position.allFinite() || !q.coeffs().allFinite() || q.norm() < 1e-6) {
      return;
    }
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = q.normalized().toRotationMatrix();
    pose.translation() = position;
    std::lock_guard<std::mutex> lock(state_mutex_);
    pose_buffer_->Add(rclcpp::Time(message->header.stamp).nanoseconds(), pose);
    condition_.notify_all();
  }

  cv::Mat DecodeImage(const Image::ConstSharedPtr& message) {
    if (message->encoding == "mono8") {
      return cv_bridge::toCvShare(message, "mono8")->image;
    }
    if (message->encoding == "bgr8" || message->encoding == "rgb8" ||
        message->encoding == "bgra8" || message->encoding == "rgba8") {
      return cv_bridge::toCvCopy(message, "bgr8")->image;
    }
    throw std::invalid_argument(
        "Supported input encodings: mono8, bgr8, rgb8, bgra8, rgba8");
  }

  void PublishImage(const rclcpp::Publisher<Image>::SharedPtr& publisher,
                    const std_msgs::msg::Header& header, const cv::Mat& image,
                    const std::string& encoding) {
    if (publisher->get_subscription_count() > 0) {
      publisher->publish(
          *cv_bridge::CvImage(header, encoding, image).toImageMsg());
    }
  }

  sensor_msgs::msg::CameraInfo MakeCameraInfo(
      const std_msgs::msg::Header& header, const cv::Matx34d& projection) {
    sensor_msgs::msg::CameraInfo info;
    info.header = header;
    info.width = rectifier_->size().width;
    info.height = rectifier_->size().height;
    info.distortion_model = "plumb_bob";
    info.d.assign(5, 0.0);
    info.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        info.k[row * 3 + col] = projection(row, col);
      }
      for (int col = 0; col < 4; ++col) {
        info.p[row * 4 + col] = projection(row, col);
      }
    }
    return info;
  }

  sensor_msgs::msg::PointCloud2 MakePointCloud(
      const std::vector<ColorPoint>& points,
      const std_msgs::msg::Header& header) {
    sensor_msgs::msg::PointCloud2 message;
    message.header = header;
    message.is_dense = true;
    sensor_msgs::PointCloud2Modifier modifier(message);
    // Explicit fields avoid the 32-byte alignment padding introduced by
    // setPointCloud2FieldsByString. Standard XYZ/RGB stays intact at 16 bytes
    // per point, reducing fragmentation for dense best-effort DDS clouds.
    using Field = sensor_msgs::msg::PointField;
    modifier.setPointCloud2Fields(4,
        "x", 1, Field::FLOAT32, "y", 1, Field::FLOAT32,
        "z", 1, Field::FLOAT32, "rgb", 1, Field::FLOAT32);
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x(message, "x"), y(message, "y"),
        z(message, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> r(message, "r"), g(message, "g"),
        b(message, "b");
    for (const auto& point : points) {
      *x = point.position.x();
      *y = point.position.y();
      *z = point.position.z();
      *r = point.r;
      *g = point.g;
      *b = point.b;
      ++x;
      ++y;
      ++z;
      ++r;
      ++g;
      ++b;
    }
    return message;
  }

  void RunWorker() {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(state_mutex_);
        condition_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
        if (stopping_) {
          return;
        }
        job = jobs_.front();
        jobs_.pop_front();
      }
      try {
        ProcessFrame(job);
      } catch (const std::exception& error) {
        if (rclcpp::ok()) {
          RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                                "Skipping stereo pair: %s", error.what());
        }
      }
    }
  }

  // 单帧主流程。所有耗时计算都在工作线程，只有队列/位姿和地图读写时加锁。
  void ProcessFrame(const Job& job) {
    const auto start = std::chrono::steady_clock::now();
    double fusion_ms = 0.0;
    // 阶段 1：解码与校正。校正保持标定分辨率，并改变相机坐标轴方向。
    cv::Mat left, right;
    rectifier_->Rectify(DecodeImage(job.left), DecodeImage(job.right), left,
                        right);
    // 阶段 2：输出统一的浮点像素视差；SGBM 的 /16 和模型补边都在后端处理。
    const auto disparity = matcher_->Compute(left, right);
    // 阶段 3：Q 反投影得到米制深度，再按 pixel_stride 生成当前相机点云。
    const auto geometry =
        ProjectDisparity(disparity, left, rectifier_->reprojection_matrix(),
                         rectifier_->left_mask(), rectifier_->right_mask(),
                         min_depth_, max_depth_, pixel_stride_);
    if (!rclcpp::ok() || job.generation != generation_) {
      return;
    }
    auto left_header = job.left->header;
    left_header.frame_id = left_frame_;
    auto right_header = job.right->header;
    right_header.frame_id = right_frame_;
    PublishImage(left_pub_, left_header, left,
                 left.channels() == 1 ? "mono8" : "bgr8");
    PublishImage(right_pub_, right_header, right,
                 right.channels() == 1 ? "mono8" : "bgr8");
    PublishImage(disparity_pub_, left_header, disparity, "32FC1");
    PublishImage(depth_pub_, left_header, geometry.depth, "32FC1");
    left_info_pub_->publish(
        MakeCameraInfo(left_header, rectifier_->left_projection()));
    right_info_pub_->publish(
        MakeCameraInfo(right_header, rectifier_->right_projection()));
    if (points_pub_->get_subscription_count() > 0) {
      points_pub_->publish(MakePointCloud(geometry.points, left_header));
    }
    ++processed_;
    // 阶段 4：始终按图像采集时间匹配位姿，不能使用推理完成时的最新位姿。
    std::optional<Eigen::Isometry3d> pose;
    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      const int64_t stamp = rclcpp::Time(job.left->header.stamp).nanoseconds();
      condition_.wait_for(
          lock, std::chrono::duration<double>(pose_wait_sec_), [&] {
            pose = pose_buffer_->Find(stamp, pose_tolerance_ns_);
            return stopping_ || job.generation != generation_ ||
                   pose.has_value();
          });
      if (stopping_) {
        return;
      }
      if (job.generation != generation_) {
        return;
      }
    }
    if (!pose) {
      ++missing_pose_;
    } else if (!geometry.points.empty() ||
               (tsdf_map_ && cv::countNonZero(geometry.depth > 0) > 0)) {
      // 阶段 5：世界←机体←校正相机。逆校正旋转已经包含在后一个变换中。
      const Eigen::Isometry3d t_world_camera =
          *pose * rectifier_->body_from_left_rect();
      auto world_header = left_header;
      world_header.frame_id = fixed_frame_;
      if (world_points_pub_->get_subscription_count() > 0) {
        auto points = geometry.points;
        for (auto& point : points) {
          point.position = t_world_camera * point.position;
        }
        world_points_pub_->publish(MakePointCloud(points, world_header));
      }
      // 阶段 6：只有关键帧进入地图；地图更新与发布/保存由同一个锁保护。
      std::lock_guard<std::mutex> lock(map_mutex_);
      if (job.generation == generation_ && keyframes_->ShouldInsert(*pose)) {
        const auto fusion_start = std::chrono::steady_clock::now();
        bool capacity_reached = false;
        if (tsdf_map_) {
          const auto& projection = rectifier_->left_projection();
          const cv::Matx33d intrinsics(
              projection(0, 0), 0, projection(0, 2),
              0, projection(1, 1), projection(1, 2), 0, 0, 1);
          capacity_reached = tsdf_map_->Integrate(
              geometry.depth, left, intrinsics, t_world_camera);
        } else {
          capacity_reached = map_->Integrate(geometry.points, t_world_camera) > 0;
        }
        fusion_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fusion_start).count();
        keyframes_->Accept(*pose);
        map_stamp_ = rclcpp::Time(job.left->header.stamp);
        map_dirty_ = true;
        ++fused_;
        if (capacity_reached) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                               "Map capacity reached; existing voxels "
                               "still update, new voxels are skipped");
        }
      }
    }
    const double elapsed = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    RCLCPP_INFO_THROTTLE(
        get_logger(), diagnostic_clock_, 5000,
        "Stereo %.1f ms (includes pose wait), fusion=%s %.1f ms, processed=%" PRIu64
        " "
        "keyframes=%" PRIu64 " queue_drops=%" PRIu64 " missing_pose=%" PRIu64,
        elapsed, fusion_method_.c_str(), fusion_ms, processed_.load(), fused_.load(), dropped_.load(),
        missing_pose_.load());
  }

  // Caller holds map_mutex_. TSDF publishes observed zero crossings, not voxel centres.
  std::vector<ColorPoint> MapPointsLocked() const {
    return tsdf_map_ ? tsdf_map_->points() : map_->points();
  }

  void PublishMap() {
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (!map_dirty_) {
      return;
    }
    std_msgs::msg::Header header;
    header.frame_id = fixed_frame_;
    header.stamp = map_stamp_;
    map_pub_->publish(MakePointCloud(MapPointsLocked(), header));
    map_dirty_ = false;
  }

  std::string fixed_frame_, body_frame_, left_frame_, right_frame_;
  std::string fusion_method_;
  rclcpp::Clock diagnostic_clock_{RCL_STEADY_TIME};
  double min_depth_, max_depth_, pose_wait_sec_;
  double map_publish_period_sec_;
  int pixel_stride_;
  size_t queue_capacity_;
  int64_t pose_tolerance_ns_;
  std::unique_ptr<StereoMatcher> matcher_;
  std::unique_ptr<StereoRectifier> rectifier_;
  std::unique_ptr<VoxelMap> map_;
  std::unique_ptr<TsdfMap> tsdf_map_;
  std::unique_ptr<KeyframeSelector> keyframes_;
  std::unique_ptr<PoseBuffer> pose_buffer_;
  std::mutex state_mutex_, map_mutex_;
  std::condition_variable condition_;
  std::deque<Job> jobs_;
  std::optional<int64_t> last_image_stamp_;
  bool stopping_ = false, map_dirty_ = true;
  rclcpp::Time map_stamp_{0, 0, RCL_ROS_TIME};
  std::atomic<uint64_t> generation_{0}, dropped_{0}, processed_{0}, fused_{0},
      missing_pose_{0};
  std::thread worker_;
  message_filters::Subscriber<Image> left_sub_, right_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer_;
  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<Image>::SharedPtr left_pub_, right_pub_, disparity_pub_,
      depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_,
      right_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_,
      world_points_pub_, map_pub_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::Service<Trigger>::SharedPtr clear_service_, save_service_;
  rclcpp::TimerBase::SharedPtr map_timer_;
};
}  // namespace stereo_recon

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    auto node = std::make_shared<stereo_recon::ReconstructionNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),
                                                      2);
    executor.add_node(node);
    executor.spin();
    node->Stop();
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("stereo_reconstruction"), "%s",
                 error.what());
    status = 1;
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return status;
}

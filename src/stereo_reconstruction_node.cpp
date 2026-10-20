#include <msckf_vio/stereo/reconstruction.h>
#include <msckf_vio/stereo/stereo_matcher.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

namespace msckf_vio::stereo {
using Image = sensor_msgs::msg::Image;
using Odometry = nav_msgs::msg::Odometry;
using Trigger = std_srvs::srv::Trigger;
using SyncPolicy = message_filters::sync_policies::ExactTime<Image, Image>;

class ReconstructionNode final : public rclcpp::Node {
 public:
  ReconstructionNode() : Node("stereo_reconstruction") {
    const auto positive_int = [this](const std::string& name, int default_value) {
      const int value = declare_parameter<int>(name, default_value);
      if (value <= 0) throw std::invalid_argument(name + " must be positive");
      return value;
    };
    fixed_frame_ = declare_parameter<std::string>("fixed_frame_id", "world");
    body_frame_ = declare_parameter<std::string>("pose_child_frame_id", "odom");
    left_frame_ = declare_parameter<std::string>("left_rect_frame_id", "cam0_rect_optical_frame");
    right_frame_ = declare_parameter<std::string>("right_rect_frame_id", "cam1_rect_optical_frame");
    if (fixed_frame_.empty() || body_frame_.empty() || left_frame_.empty() || right_frame_.empty() ||
        fixed_frame_ == body_frame_ || left_frame_ == right_frame_ ||
        left_frame_ == body_frame_ || right_frame_ == body_frame_ ||
        left_frame_ == fixed_frame_ || right_frame_ == fixed_frame_)
      throw std::invalid_argument("World, body and rectified camera frame names must be distinct and nonempty");
    min_depth_ = declare_parameter<double>("min_depth", 0.3);
    max_depth_ = declare_parameter<double>("max_depth", 10.0);
    pixel_stride_ = positive_int("pixel_stride", 2);
    queue_capacity_ = positive_int("max_pending_frames", 4);
    pose_buffer_ = std::make_unique<PoseBuffer>(positive_int("pose_cache_size", 400));
    const double tolerance = declare_parameter<double>("pose_tolerance_sec", 0.0);
    pose_wait_sec_ = declare_parameter<double>("pose_wait_sec", 0.3);
    const double map_period = declare_parameter<double>("map_publish_period_sec", 0.5);
    if (!std::isfinite(min_depth_) || !std::isfinite(max_depth_) || min_depth_ <= 0 ||
        max_depth_ <= min_depth_ || !std::isfinite(tolerance) || tolerance < 0 || tolerance > 1 ||
        !std::isfinite(pose_wait_sec_) || pose_wait_sec_ < 0 ||
        !std::isfinite(map_period) || map_period <= 0)
      throw std::invalid_argument("Invalid depth limits, pose timing or map publish period");
    pose_tolerance_ns_ = static_cast<int64_t>(tolerance * 1e9);
    map_ = std::make_unique<VoxelMap>(declare_parameter<double>("voxel_size", 0.05),
                                    positive_int("max_voxels", 500000));
    keyframes_ = std::make_unique<KeyframeSelector>(
        declare_parameter<double>("keyframe_translation", 0.1),
        declare_parameter<double>("keyframe_rotation_deg", 5.0) * std::acos(-1.0) / 180.0);
    declare_parameter<std::string>("save_map_path", "/tmp/stereo_map.pcd");

    const auto left = readCamera("cam0");
    const auto right = readCamera("cam1");
    const auto T_left_imu = readTransform("cam0/T_cam_imu");
    const auto T_imu_body = readTransform("T_imu_body");
    rectifier_ = std::make_unique<StereoRectifier>(left, right,
        readTransform("cam1/T_cn_cnm1"), T_imu_body.inverse() * T_left_imu.inverse(),
        declare_parameter<double>("rectification_alpha", 0.0));
    SgbmParameters p;
    p.min_disparity = declare_parameter<int>("sgbm.min_disparity", p.min_disparity);
    p.num_disparities = declare_parameter<int>("sgbm.num_disparities", p.num_disparities);
    p.block_size = declare_parameter<int>("sgbm.block_size", p.block_size);
    p.uniqueness_ratio = declare_parameter<int>("sgbm.uniqueness_ratio", p.uniqueness_ratio);
    p.speckle_window_size = declare_parameter<int>("sgbm.speckle_window_size", p.speckle_window_size);
    p.speckle_range = declare_parameter<int>("sgbm.speckle_range", p.speckle_range);
    p.disp12_max_diff = declare_parameter<int>("sgbm.disp12_max_diff", p.disp12_max_diff);
    p.pre_filter_cap = declare_parameter<int>("sgbm.pre_filter_cap", p.pre_filter_cap);
    IgevParameters igev;
    igev.engine_path = declare_parameter<std::string>("igev.engine_path", "");
    igev.left_input_name = declare_parameter<std::string>("igev.left_input_name", igev.left_input_name);
    igev.right_input_name = declare_parameter<std::string>("igev.right_input_name", igev.right_input_name);
    igev.output_name = declare_parameter<std::string>("igev.output_name", igev.output_name);
    igev.input_width = declare_parameter<int>("igev.input_width", igev.input_width);
    igev.input_height = declare_parameter<int>("igev.input_height", igev.input_height);
    igev.padding_multiple = declare_parameter<int>("igev.padding_multiple", igev.padding_multiple);
    igev.padding_mode = declare_parameter<std::string>("igev.padding_mode", igev.padding_mode);
    igev.device_id = declare_parameter<int>("igev.device_id", igev.device_id);
    matcher_ = createStereoMatcher(declare_parameter<std::string>("backend", "sgbm"), p, igev);
    // Bound OpenCV's CPU pool so dense stereo leaves CPU time for VIO.
    cv::setNumThreads(positive_int("opencv_threads", 2));

    const auto image_qos = rclcpp::SensorDataQoS().keep_last(10);
    left_pub_ = create_publisher<Image>("stereo/left/image_rect", image_qos);
    right_pub_ = create_publisher<Image>("stereo/right/image_rect", image_qos);
    disparity_pub_ = create_publisher<Image>("stereo/disparity", image_qos);
    depth_pub_ = create_publisher<Image>("stereo/depth", image_qos);
    left_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("stereo/left/camera_info", image_qos);
    right_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("stereo/right/camera_info", image_qos);
    points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("stereo/points", image_qos);
    world_points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("stereo/points_world", image_qos);
    map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "stereo/map", rclcpp::QoS(1).reliable().transient_local());
    static_tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
    static_tf_->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{
        cameraTransform(left_frame_, rectifier_->bodyFromLeftRect()),
        cameraTransform(right_frame_, rectifier_->bodyFromRightRect())});
    left_sub_.subscribe(this, "cam0_image", image_qos.get_rmw_qos_profile());
    right_sub_.subscribe(this, "cam1_image", image_qos.get_rmw_qos_profile());
    synchronizer_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(positive_int("stereo_sync_queue_size", 20)), left_sub_, right_sub_);
    synchronizer_->registerCallback(std::bind(&ReconstructionNode::onStereo, this,
        std::placeholders::_1, std::placeholders::_2));
    odom_sub_ = create_subscription<Odometry>("odom", rclcpp::QoS(200).best_effort(),
        std::bind(&ReconstructionNode::onOdometry, this, std::placeholders::_1));
    service_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    clear_service_ = create_service<Trigger>("stereo/clear_map",
        [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> response) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          resetLocked();
          response->success = true;
          response->message = "Cleared map, keyframes, pose cache and pending frames";
        }, rmw_qos_profile_services_default, service_group_);
    save_service_ = create_service<Trigger>("stereo/save_map",
        [this](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> response) {
          std::vector<ColorPoint> points;
          {
            std::lock_guard<std::mutex> lock(map_mutex_);
            points = map_->points();
          }
          if (points.empty()) {
            response->success = false;
            response->message = "Map is empty; no file was written";
            return;
          }
          try {
            const auto path = get_parameter("save_map_path").as_string();
            savePointCloud(path, points);
            response->success = true;
            response->message = "Saved " + std::to_string(points.size()) + " points to " + path;
          } catch (const std::exception& error) {
            response->success = false;
            response->message = error.what();
          }
        }, rmw_qos_profile_services_default, service_group_);
    map_stamp_ = now();
    map_timer_ = create_wall_timer(std::chrono::duration<double>(map_period),
                                  std::bind(&ReconstructionNode::publishMap, this));
    worker_ = std::thread(&ReconstructionNode::work, this);
    RCLCPP_INFO(get_logger(), "Stereo reconstruction: %s, %dx%d, pose %s <- %s, exact stereo sync",
        matcher_->name().c_str(), left.size.width, left.size.height,
        fixed_frame_.c_str(), body_frame_.c_str());
  }

  ~ReconstructionNode() override { stop(); }
  void stop() {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

 private:
  struct Job { Image::ConstSharedPtr left, right; uint64_t generation; };

  CameraCalibration readCamera(const std::string& prefix) {
    const auto resolution = declare_parameter<std::vector<int64_t>>(
        prefix + "/resolution", std::vector<int64_t>{});
    const auto k = declare_parameter<std::vector<double>>(prefix + "/intrinsics", std::vector<double>{});
    const auto d = declare_parameter<std::vector<double>>(prefix + "/distortion_coeffs", std::vector<double>{});
    if (resolution.size() != 2 || k.size() != 4 || d.size() != 4 ||
        resolution[0] <= 0 || resolution[1] <= 0 || resolution[0] > 20000 || resolution[1] > 20000)
      throw std::invalid_argument("Missing or invalid calibration for " + prefix);
    return {cv::Size(static_cast<int>(resolution[0]), static_cast<int>(resolution[1])),
        cv::Matx33d(k[0], 0, k[2], 0, k[1], k[3], 0, 0, 1), cv::Mat(d).clone(),
        declare_parameter<std::string>(prefix + "/distortion_model", "radtan")};
  }

  Eigen::Isometry3d readTransform(const std::string& name) {
    return transformFromRowMajor(declare_parameter<std::vector<double>>(name, std::vector<double>{}));
  }

  geometry_msgs::msg::TransformStamped cameraTransform(
      const std::string& frame, const Eigen::Isometry3d& transform) {
    geometry_msgs::msg::TransformStamped message;
    message.header.stamp = now();
    message.header.frame_id = body_frame_;
    message.child_frame_id = frame;
    message.transform.translation.x = transform.translation().x();
    message.transform.translation.y = transform.translation().y();
    message.transform.translation.z = transform.translation().z();
    const Eigen::Quaterniond q(transform.linear());
    message.transform.rotation.x = q.x(); message.transform.rotation.y = q.y();
    message.transform.rotation.z = q.z(); message.transform.rotation.w = q.w();
    return message;
  }

  // Caller holds state_mutex_; no code takes state_mutex_ while holding map_mutex_.
  void resetLocked() {
    ++generation_;
    jobs_.clear();
    pose_buffer_->clear();
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      map_->clear();
      keyframes_->clear();
      map_stamp_ = now();
      map_dirty_ = true;
    }
    condition_.notify_all();
  }

  void onStereo(const Image::ConstSharedPtr& left, const Image::ConstSharedPtr& right) {
    const int64_t stamp = rclcpp::Time(left->header.stamp).nanoseconds();
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (last_image_stamp_ && stamp < *last_image_stamp_) {
      resetLocked();
      RCLCPP_WARN(get_logger(), "Image time moved backwards; reconstruction state cleared");
    }
    last_image_stamp_ = stamp;
    while (jobs_.size() >= queue_capacity_) { jobs_.pop_front(); ++dropped_; }
    jobs_.push_back({left, right, generation_.load()});
    condition_.notify_all();
  }

  void onOdometry(const Odometry::ConstSharedPtr message) {
    if (message->header.frame_id != fixed_frame_ || message->child_frame_id != body_frame_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "Rejecting odometry frames %s <- %s; expected %s <- %s",
          message->header.frame_id.c_str(), message->child_frame_id.c_str(),
          fixed_frame_.c_str(), body_frame_.c_str());
      return;
    }
    const auto& p = message->pose.pose;
    Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
    const Eigen::Vector3d position(p.position.x, p.position.y, p.position.z);
    if (!position.allFinite() || !q.coeffs().allFinite() || q.norm() < 1e-6) return;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = q.normalized().toRotationMatrix();
    pose.translation() = position;
    std::lock_guard<std::mutex> lock(state_mutex_);
    pose_buffer_->add(rclcpp::Time(message->header.stamp).nanoseconds(), pose);
    condition_.notify_all();
  }

  cv::Mat decode(const Image::ConstSharedPtr& message) {
    if (message->encoding == "mono8") return cv_bridge::toCvShare(message, "mono8")->image;
    if (message->encoding == "bgr8" || message->encoding == "rgb8" ||
        message->encoding == "bgra8" || message->encoding == "rgba8")
      return cv_bridge::toCvCopy(message, "bgr8")->image;
    throw std::invalid_argument("Supported input encodings: mono8, bgr8, rgb8, bgra8, rgba8");
  }

  void publishImage(const rclcpp::Publisher<Image>::SharedPtr& publisher,
      const std_msgs::msg::Header& header, const cv::Mat& image, const std::string& encoding) {
    if (publisher->get_subscription_count() > 0)
      publisher->publish(*cv_bridge::CvImage(header, encoding, image).toImageMsg());
  }

  sensor_msgs::msg::CameraInfo cameraInfo(
      const std_msgs::msg::Header& header, const cv::Matx34d& projection) {
    sensor_msgs::msg::CameraInfo info;
    info.header = header;
    info.width = rectifier_->size().width;
    info.height = rectifier_->size().height;
    info.distortion_model = "plumb_bob";
    info.d.assign(5, 0.0);
    info.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) info.k[row * 3 + col] = projection(row, col);
      for (int col = 0; col < 4; ++col) info.p[row * 4 + col] = projection(row, col);
    }
    return info;
  }

  sensor_msgs::msg::PointCloud2 cloud(const std::vector<ColorPoint>& points,
                                     const std_msgs::msg::Header& header) {
    sensor_msgs::msg::PointCloud2 message;
    message.header = header;
    message.is_dense = true;
    sensor_msgs::PointCloud2Modifier modifier(message);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x(message, "x"), y(message, "y"), z(message, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> r(message, "r"), g(message, "g"), b(message, "b");
    for (const auto& point : points) {
      *x = point.position.x(); *y = point.position.y(); *z = point.position.z();
      *r = point.r; *g = point.g; *b = point.b;
      ++x; ++y; ++z; ++r; ++g; ++b;
    }
    return message;
  }

  void work() {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(state_mutex_);
        condition_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
        if (stopping_) return;
        job = jobs_.front(); jobs_.pop_front();
      }
      try {
        const auto start = std::chrono::steady_clock::now();
        cv::Mat left, right;
        rectifier_->rectify(decode(job.left), decode(job.right), left, right);
        const auto disparity = matcher_->compute(left, right);
        const auto geometry = projectDisparity(disparity, left, rectifier_->Q(),
            rectifier_->leftMask(), rectifier_->rightMask(), min_depth_, max_depth_, pixel_stride_);
        if (!rclcpp::ok() || job.generation != generation_) continue;
        auto left_header = job.left->header;
        left_header.frame_id = left_frame_;
        auto right_header = job.right->header;
        right_header.frame_id = right_frame_;
        publishImage(left_pub_, left_header, left, left.channels() == 1 ? "mono8" : "bgr8");
        publishImage(right_pub_, right_header, right, right.channels() == 1 ? "mono8" : "bgr8");
        publishImage(disparity_pub_, left_header, disparity, "32FC1");
        publishImage(depth_pub_, left_header, geometry.depth, "32FC1");
        left_info_pub_->publish(cameraInfo(left_header, rectifier_->leftProjection()));
        right_info_pub_->publish(cameraInfo(right_header, rectifier_->rightProjection()));
        if (points_pub_->get_subscription_count() > 0) points_pub_->publish(cloud(geometry.points, left_header));
        ++processed_;
        std::optional<Eigen::Isometry3d> pose;
        {
          std::unique_lock<std::mutex> lock(state_mutex_);
          const int64_t stamp = rclcpp::Time(job.left->header.stamp).nanoseconds();
          condition_.wait_for(lock, std::chrono::duration<double>(pose_wait_sec_), [&] {
            pose = pose_buffer_->find(stamp, pose_tolerance_ns_);
            return stopping_ || job.generation != generation_ || pose.has_value();
          });
          if (stopping_) return;
          if (job.generation != generation_) continue;
        }
        if (!pose) {
          ++missing_pose_;
        } else if (!geometry.points.empty()) {
          const Eigen::Isometry3d T_world_camera = *pose * rectifier_->bodyFromLeftRect();
          auto world_header = left_header;
          world_header.frame_id = fixed_frame_;
          if (world_points_pub_->get_subscription_count() > 0) {
            auto points = geometry.points;
            for (auto& point : points) point.position = T_world_camera * point.position;
            world_points_pub_->publish(cloud(points, world_header));
          }
          std::lock_guard<std::mutex> lock(map_mutex_);
          if (job.generation == generation_ && keyframes_->shouldInsert(*pose)) {
            const auto skipped = map_->integrate(geometry.points, T_world_camera);
            keyframes_->accept(*pose);
            map_stamp_ = rclcpp::Time(job.left->header.stamp);
            map_dirty_ = true;
            ++fused_;
            if (skipped) RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                "Voxel map capacity reached; existing voxels still update, new voxels are skipped");
          }
        }
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
            "Stereo %.1f ms (includes pose wait), processed=%llu keyframes=%llu queue_drops=%llu missing_pose=%llu",
            elapsed, static_cast<unsigned long long>(processed_.load()),
            static_cast<unsigned long long>(fused_.load()), static_cast<unsigned long long>(dropped_.load()),
            static_cast<unsigned long long>(missing_pose_.load()));
      } catch (const std::exception& error) {
        if (rclcpp::ok()) RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
            "Skipping stereo pair: %s", error.what());
      }
    }
  }

  void publishMap() {
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (!map_dirty_) return;
    std_msgs::msg::Header header;
    header.frame_id = fixed_frame_;
    header.stamp = map_stamp_;
    map_pub_->publish(cloud(map_->points(), header));
    map_dirty_ = false;
  }

  std::string fixed_frame_, body_frame_, left_frame_, right_frame_;
  double min_depth_, max_depth_, pose_wait_sec_;
  int pixel_stride_;
  size_t queue_capacity_;
  int64_t pose_tolerance_ns_;
  std::unique_ptr<StereoMatcher> matcher_;
  std::unique_ptr<StereoRectifier> rectifier_;
  std::unique_ptr<VoxelMap> map_;
  std::unique_ptr<KeyframeSelector> keyframes_;
  std::unique_ptr<PoseBuffer> pose_buffer_;
  std::mutex state_mutex_, map_mutex_;
  std::condition_variable condition_;
  std::deque<Job> jobs_;
  std::optional<int64_t> last_image_stamp_;
  bool stopping_ = false, map_dirty_ = true;
  rclcpp::Time map_stamp_{0, 0, RCL_ROS_TIME};
  std::atomic<uint64_t> generation_{0}, dropped_{0}, processed_{0}, fused_{0}, missing_pose_{0};
  std::thread worker_;
  message_filters::Subscriber<Image> left_sub_, right_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer_;
  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<Image>::SharedPtr left_pub_, right_pub_, disparity_pub_, depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_, right_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_, world_points_pub_, map_pub_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::Service<Trigger>::SharedPtr clear_service_, save_service_;
  rclcpp::TimerBase::SharedPtr map_timer_;
};
}  // namespace msckf_vio::stereo

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    auto node = std::make_shared<msckf_vio::stereo::ReconstructionNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
    node->stop();
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("stereo_reconstruction"), "%s", error.what());
    status = 1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return status;
}

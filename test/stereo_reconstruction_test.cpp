#include <msckf_vio/stereo/reconstruction.h>
#include <msckf_vio/stereo/stereo_matcher.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <gtest/gtest.h>
#include <opencv2/core/eigen.hpp>

namespace msckf_vio::stereo {
namespace {
CameraCalibration camera() {
  return {cv::Size(320, 240), cv::Matx33d(200, 0, 160, 0, 200, 120, 0, 0, 1),
          cv::Mat::zeros(4, 1, CV_64F), "radtan"};
}
Eigen::Isometry3d baseline() {
  auto pose = Eigen::Isometry3d::Identity();
  pose.translation().x() = -0.1;
  return pose;
}
}  // namespace

TEST(StereoMatcher, RecoversKnownShiftInPixelsThroughVirtualInterface) {
  SgbmParameters p;
  p.num_disparities = 64;
  p.speckle_window_size = 0;
  auto matcher = createStereoMatcher("sgbm", p);
  cv::Mat left(160, 320, CV_8UC1), right;
  cv::RNG random(42);
  random.fill(left, cv::RNG::UNIFORM, 0, 256);
  right = cv::Mat::zeros(left.size(), left.type());
  left(cv::Rect(8, 0, 312, 160)).copyTo(right(cv::Rect(0, 0, 312, 160)));
  const auto disparity = matcher->compute(left, right);
  ASSERT_EQ(disparity.type(), CV_32FC1);
  ASSERT_EQ(disparity.size(), left.size());
  int correct = 0, count = 0;
  for (int y = 10; y < 150; ++y) {
    for (int x = 80; x < 300; ++x) {
      ++count;
      if (std::abs(disparity.at<float>(y, x) - 8.0f) < 0.25f) ++correct;
    }
  }
  EXPECT_GT(double(correct) / count, 0.95);
  EXPECT_TRUE(std::isnan(disparity.at<float>(80, 0)));
  EXPECT_THROW(matcher->compute(left, cv::Mat()), std::invalid_argument);
  EXPECT_THROW(createStereoMatcher("tensorrt", p), std::invalid_argument);
  p.num_disparities = 63;
  EXPECT_THROW(SgbmMatcher{p}, std::invalid_argument);
}

TEST(StereoGeometry, ProjectsMetricDepthAndRejectsInvalidCorrespondences) {
  StereoRectifier rectifier(camera(), camera(), baseline(), Eigen::Isometry3d::Identity(), 0);
  cv::Mat left(camera().size, CV_8UC1, cv::Scalar(128));
  cv::Mat disparity(camera().size, CV_32FC1, cv::Scalar(10));
  auto mask = cv::Mat(camera().size, CV_8UC1, cv::Scalar(255));
  auto right_mask = mask.clone();
  disparity.at<float>(100, 100) = std::numeric_limits<float>::quiet_NaN();
  disparity.at<float>(100, 101) = 0;
  disparity.at<float>(100, 102) = -1;
  disparity.at<float>(100, 103) = std::numeric_limits<float>::infinity();
  right_mask.at<uint8_t>(100, 100) = 0;  // Invalid right correspondence of left x=110.
  const auto result = projectDisparity(disparity, left, rectifier.Q(), mask, right_mask, .3, 10, 2);
  EXPECT_NEAR(result.depth.at<float>(120, 160), 2.0, 1e-6);
  for (int x : {100, 101, 102, 103, 110}) EXPECT_TRUE(std::isnan(result.depth.at<float>(100, x)));
  EXPECT_TRUE(std::isnan(result.depth.at<float>(120, 5)));
  ASSERT_FALSE(result.points.empty());
  EXPECT_EQ(result.points.front().r, 128);
  const auto clipped = projectDisparity(disparity, left, rectifier.Q(), mask, right_mask, .3, 1, 2);
  EXPECT_TRUE(clipped.points.empty());
}

TEST(StereoGeometry, RestoresOriginalCameraAxesAfterRectification) {
  auto T_right_left = baseline();
  T_right_left.linear() = Eigen::AngleAxisd(.06, Eigen::Vector3d::UnitY()).toRotationMatrix();
  auto T_body_left = Eigen::Isometry3d::Identity();
  T_body_left.linear() = Eigen::AngleAxisd(.4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  T_body_left.translation() = Eigen::Vector3d(.1, -.2, .3);
  StereoRectifier rectifier(camera(), camera(), T_right_left, T_body_left, 0);
  cv::Mat R, t, R1, R2, P1, P2, Q;
  cv::eigen2cv(Eigen::Matrix3d(T_right_left.linear()), R);
  cv::eigen2cv(Eigen::Vector3d(T_right_left.translation()), t);
  cv::stereoRectify(camera().K, camera().distortion, camera().K, camera().distortion,
      camera().size, R, t, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0);
  Eigen::Matrix3d rectification;
  cv::cv2eigen(R1, rectification);
  const Eigen::Vector3d original_point(.2, .1, 2);
  EXPECT_LT((rectifier.bodyFromLeftRect() * (rectification * original_point) -
             T_body_left * original_point).norm(), 1e-9);
  cv::Mat left(240, 320, CV_8UC1), left_rect, right_rect;
  EXPECT_THROW(rectifier.rectify(left, cv::Mat(200, 320, CV_8UC1), left_rect, right_rect),
               std::invalid_argument);
  EXPECT_THROW(StereoRectifier(camera(), camera(), Eigen::Isometry3d::Identity(),
                              T_body_left, 0), std::invalid_argument);
}

TEST(StereoGeometry, ValidatesRigidTransformDirectionAndLayout) {
  const std::vector<double> values{1, 0, 0, .1, 0, 1, 0, -.2, 0, 0, 1, .3, 0, 0, 0, 1};
  const auto transform = transformFromRowMajor(values);
  EXPECT_LT((transform.translation() - Eigen::Vector3d(.1, -.2, .3)).norm(), 1e-12);
  EXPECT_THROW(transformFromRowMajor({}), std::invalid_argument);
  auto bad = values;
  bad[0] = 2;
  EXPECT_THROW(transformFromRowMajor(bad), std::invalid_argument);
}

TEST(PoseBuffer, RequiresTimestampMatchAndBoundsCache) {
  PoseBuffer buffer(2);
  auto pose = Eigen::Isometry3d::Identity();
  buffer.add(10, pose);
  pose.translation().x() = 1;
  buffer.add(20, pose);
  ASSERT_TRUE(buffer.find(20, 0));
  EXPECT_DOUBLE_EQ(buffer.find(20, 0)->translation().x(), 1);
  EXPECT_FALSE(buffer.find(19, 0));
  EXPECT_TRUE(buffer.find(19, 1));
  EXPECT_FALSE(buffer.find(30, 1));
  buffer.add(30, pose);
  EXPECT_FALSE(buffer.find(10, 0));
  buffer.clear();
  EXPECT_FALSE(buffer.find(30, 0));
}

TEST(VoxelMap, FusesWorldPointsAndKeepsNegativeVoxelsSeparate) {
  VoxelMap map(.1, 2);
  const auto identity = Eigen::Isometry3d::Identity();
  map.integrate({{{-.01, 0, 1}, 0, 0, 0}, {{.01, 0, 1}, 100, 100, 100},
                 {{.03, 0, 1}, 200, 200, 200}}, identity);
  ASSERT_EQ(map.size(), 2u);
  auto points = map.points();
  std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
    return a.position.x() < b.position.x();
  });
  EXPECT_NEAR(points[0].position.x(), -.01, 1e-12);
  EXPECT_NEAR(points[1].position.x(), .02, 1e-12);
  EXPECT_EQ(points[1].r, 150);
  auto translated = Eigen::Isometry3d::Identity();
  translated.translation().x() = 2;
  EXPECT_EQ(map.integrate({{{0, 0, 1}, 255, 0, 0}}, translated), 1u);
  EXPECT_EQ(map.size(), 2u);
  map.clear();
  map.integrate({{{0, 0, 1}, 255, 0, 0}}, translated);
  EXPECT_NEAR(map.points()[0].position.x(), 2, 1e-12);
}

TEST(Keyframes, UsesTranslationOrRotationAndResets) {
  KeyframeSelector selector(.1, .1);
  auto pose = Eigen::Isometry3d::Identity();
  EXPECT_TRUE(selector.shouldInsert(pose));
  selector.accept(pose);
  EXPECT_FALSE(selector.shouldInsert(pose));
  pose.translation().x() = .11;
  EXPECT_TRUE(selector.shouldInsert(pose));
  pose.translation().setZero();
  pose.linear() = Eigen::AngleAxisd(.11, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  EXPECT_TRUE(selector.shouldInsert(pose));
  selector.clear();
  EXPECT_TRUE(selector.shouldInsert(Eigen::Isometry3d::Identity()));
}

TEST(PointCloudExport, WritesPcdAndPlyAndReportsErrors) {
  const auto directory = std::filesystem::temp_directory_path() / "msckf_stereo_export_test";
  std::filesystem::create_directories(directory);
  for (const auto& extension : {".pcd", ".ply"}) {
    const auto file = directory / (std::string("map") + extension);
    savePointCloud(file.string(), {{{1, 2, 3}, 10, 20, 30}});
    std::ifstream input(file);
    const std::string content((std::istreambuf_iterator<char>(input)), {});
    EXPECT_NE(content.find("1 2 3"), std::string::npos);
    EXPECT_GT(content.size(), 100u);
  }
  EXPECT_THROW(savePointCloud((directory / "map.bin").string(), {}), std::invalid_argument);
  EXPECT_THROW(savePointCloud((directory / "absent" / "map.pcd").string(), {}), std::ios_base::failure);
  std::filesystem::remove_all(directory);
}
}  // namespace msckf_vio::stereo

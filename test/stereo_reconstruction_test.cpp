#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "gtest/gtest.h"
#include "opencv2/core/eigen.hpp"

#include "stereo_recon/stereo/reconstruction.h"
#include "stereo_recon/stereo/stereo_matcher.h"

namespace stereo_recon {
namespace {
CameraCalibration MakeTestCamera() {
  return {cv::Size(320, 240), cv::Matx33d(200, 0, 160, 0, 200, 120, 0, 0, 1),
          cv::Mat::zeros(4, 1, CV_64F), "radtan"};
}
Eigen::Isometry3d MakeTestBaseline() {
  auto pose = Eigen::Isometry3d::Identity();
  pose.translation().x() = -0.1;
  return pose;
}
}  // namespace

TEST(StereoMatcher, RecoversKnownShiftInPixelsThroughVirtualInterface) {
  SgbmParameters p;
  p.num_disparities = 64;
  p.speckle_window_size = 0;
  auto matcher = CreateStereoMatcher("sgbm", p);
  cv::Mat left(160, 320, CV_8UC1), right;
  cv::RNG random(42);
  random.fill(left, cv::RNG::UNIFORM, 0, 256);
  right = cv::Mat::zeros(left.size(), left.type());
  left(cv::Rect(8, 0, 312, 160)).copyTo(right(cv::Rect(0, 0, 312, 160)));
  const auto disparity = matcher->Compute(left, right);
  ASSERT_EQ(disparity.type(), CV_32FC1);
  ASSERT_EQ(disparity.size(), left.size());
  int correct = 0, count = 0;
  for (int y = 10; y < 150; ++y) {
    for (int x = 80; x < 300; ++x) {
      ++count;
      if (std::abs(disparity.at<float>(y, x) - 8.0f) < 0.25f) {
        ++correct;
      }
    }
  }
  EXPECT_GT(static_cast<double>(correct) / count, 0.95);
  EXPECT_TRUE(std::isnan(disparity.at<float>(80, 0)));
  EXPECT_THROW(matcher->Compute(left, cv::Mat()), std::invalid_argument);
  EXPECT_THROW(CreateStereoMatcher("tensorrt", p), std::invalid_argument);
  p.num_disparities = 63;
  EXPECT_THROW(SgbmMatcher{p}, std::invalid_argument);
}

TEST(StereoGeometry, ProjectsMetricDepthAndRejectsInvalidCorrespondences) {
  StereoRectifier rectifier(MakeTestCamera(), MakeTestCamera(),
                            MakeTestBaseline(), Eigen::Isometry3d::Identity(),
                            0);
  cv::Mat left(MakeTestCamera().size, CV_8UC1, cv::Scalar(128));
  cv::Mat disparity(MakeTestCamera().size, CV_32FC1, cv::Scalar(10));
  auto mask = cv::Mat(MakeTestCamera().size, CV_8UC1, cv::Scalar(255));
  auto right_mask = mask.clone();
  disparity.at<float>(100, 100) = std::numeric_limits<float>::quiet_NaN();
  disparity.at<float>(100, 101) = 0;
  disparity.at<float>(100, 102) = -1;
  disparity.at<float>(100, 103) = std::numeric_limits<float>::infinity();
  right_mask.at<uint8_t>(100, 100) =
      0;  // Invalid right correspondence of left x=110.
  const auto result =
      ProjectDisparity(disparity, left, rectifier.reprojection_matrix(), mask,
                       right_mask, .3, 10, 2);
  EXPECT_NEAR(result.depth.at<float>(120, 160), 2.0, 1e-6);
  for (int x : {100, 101, 102, 103, 110}) {
    EXPECT_TRUE(std::isnan(result.depth.at<float>(100, x)));
  }
  EXPECT_TRUE(std::isnan(result.depth.at<float>(120, 5)));
  ASSERT_FALSE(result.points.empty());
  EXPECT_EQ(result.points.front().r, 128);
  const auto clipped =
      ProjectDisparity(disparity, left, rectifier.reprojection_matrix(), mask,
                       right_mask, .3, 1, 2);
  EXPECT_TRUE(clipped.points.empty());
}

TEST(StereoGeometry, RestoresOriginalCameraAxesAfterRectification) {
  auto t_right_left = MakeTestBaseline();
  t_right_left.linear() =
      Eigen::AngleAxisd(.06, Eigen::Vector3d::UnitY()).toRotationMatrix();
  auto t_body_left = Eigen::Isometry3d::Identity();
  t_body_left.linear() =
      Eigen::AngleAxisd(.4, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  t_body_left.translation() = Eigen::Vector3d(.1, -.2, .3);
  StereoRectifier rectifier(MakeTestCamera(), MakeTestCamera(), t_right_left,
                            t_body_left, 0);
  cv::Mat rotation_matrix, t, left_rect_rotation, right_rect_rotation,
      left_projection, right_projection, reprojection_matrix;
  cv::eigen2cv(Eigen::Matrix3d(t_right_left.linear()), rotation_matrix);
  cv::eigen2cv(Eigen::Vector3d(t_right_left.translation()), t);
  cv::stereoRectify(MakeTestCamera().intrinsics, MakeTestCamera().distortion,
                    MakeTestCamera().intrinsics, MakeTestCamera().distortion,
                    MakeTestCamera().size, rotation_matrix, t,
                    left_rect_rotation, right_rect_rotation, left_projection,
                    right_projection, reprojection_matrix,
                    cv::CALIB_ZERO_DISPARITY, 0);
  Eigen::Matrix3d rectification;
  cv::cv2eigen(left_rect_rotation, rectification);
  const Eigen::Vector3d original_point(.2, .1, 2);
  EXPECT_LT(
      (rectifier.body_from_left_rect() * (rectification * original_point) -
       t_body_left * original_point)
          .norm(),
      1e-9);
  cv::Mat left(240, 320, CV_8UC1), left_rect, right_rect;
  EXPECT_THROW(rectifier.Rectify(left, cv::Mat(200, 320, CV_8UC1), left_rect,
                                 right_rect),
               std::invalid_argument);
  EXPECT_THROW(StereoRectifier(MakeTestCamera(), MakeTestCamera(),
                               Eigen::Isometry3d::Identity(), t_body_left, 0),
               std::invalid_argument);
}

TEST(StereoGeometry, ValidatesRigidTransformDirectionAndLayout) {
  const std::vector<double> values{1, 0, 0, .1, 0, 1, 0, -.2,
                                   0, 0, 1, .3, 0, 0, 0, 1};
  const auto transform = TransformFromRowMajor(values);
  EXPECT_LT((transform.translation() - Eigen::Vector3d(.1, -.2, .3)).norm(),
            1e-12);
  EXPECT_THROW(TransformFromRowMajor({}), std::invalid_argument);
  auto bad = values;
  bad[0] = 2;
  EXPECT_THROW(TransformFromRowMajor(bad), std::invalid_argument);
}

TEST(PoseBuffer, RequiresTimestampMatchAndBoundsCache) {
  PoseBuffer buffer(2);
  auto pose = Eigen::Isometry3d::Identity();
  buffer.Add(10, pose);
  pose.translation().x() = 1;
  buffer.Add(20, pose);
  ASSERT_TRUE(buffer.Find(20, 0));
  EXPECT_DOUBLE_EQ(buffer.Find(20, 0)->translation().x(), 1);
  EXPECT_FALSE(buffer.Find(19, 0));
  EXPECT_TRUE(buffer.Find(19, 1));
  EXPECT_FALSE(buffer.Find(30, 1));
  buffer.Add(30, pose);
  EXPECT_FALSE(buffer.Find(10, 0));
  buffer.Clear();
  EXPECT_FALSE(buffer.Find(30, 0));
}

TEST(VoxelMap, FusesWorldPointsAndKeepsNegativeVoxelsSeparate) {
  VoxelMap map(.1, 2);
  const auto identity = Eigen::Isometry3d::Identity();
  map.Integrate({{{-.01, 0, 1}, 0, 0, 0},
                 {{.01, 0, 1}, 100, 100, 100},
                 {{.03, 0, 1}, 200, 200, 200}},
                identity);
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
  EXPECT_EQ(map.Integrate({{{0, 0, 1}, 255, 0, 0}}, translated), 1u);
  EXPECT_EQ(map.size(), 2u);
  map.Clear();
  map.Integrate({{{0, 0, 1}, 255, 0, 0}}, translated);
  EXPECT_NEAR(map.points()[0].position.x(), 2, 1e-12);
}

TEST(Keyframes, UsesTranslationOrRotationAndResets) {
  KeyframeSelector selector(.1, .1);
  auto pose = Eigen::Isometry3d::Identity();
  EXPECT_TRUE(selector.ShouldInsert(pose));
  selector.Accept(pose);
  EXPECT_FALSE(selector.ShouldInsert(pose));
  pose.translation().x() = .11;
  EXPECT_TRUE(selector.ShouldInsert(pose));
  pose.translation().setZero();
  pose.linear() =
      Eigen::AngleAxisd(.11, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  EXPECT_TRUE(selector.ShouldInsert(pose));
  selector.Clear();
  EXPECT_TRUE(selector.ShouldInsert(Eigen::Isometry3d::Identity()));
}

TEST(PointCloudExport, WritesPcdAndPlyAndReportsErrors) {
  const auto directory =
      std::filesystem::temp_directory_path() / "msckf_stereo_export_test";
  std::filesystem::create_directories(directory);
  for (const auto& extension : {".pcd", ".ply"}) {
    const auto file = directory / (std::string("map") + extension);
    SavePointCloud(file.string(), {{{1, 2, 3}, 10, 20, 30}});
    std::ifstream input(file);
    const std::string content((std::istreambuf_iterator<char>(input)), {});
    EXPECT_NE(content.find("1 2 3"), std::string::npos);
    EXPECT_GT(content.size(), 100u);
  }
  EXPECT_THROW(SavePointCloud((directory / "map.bin").string(), {}),
               std::invalid_argument);
  EXPECT_THROW(SavePointCloud((directory / "absent" / "map.pcd").string(), {}),
               std::ios_base::failure);
  std::filesystem::remove_all(directory);
}
}  // namespace stereo_recon

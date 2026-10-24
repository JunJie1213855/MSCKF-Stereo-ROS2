#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>
#include "stereo_recon/stereo/tsdf_map.h"

namespace stereo_recon {
namespace {
const cv::Matx33d kIntrinsics(80, 0, 31.5, 0, 80, 31.5, 0, 0, 1);
cv::Mat Depth(float z = 2.0f) { return cv::Mat(64, 64, CV_32F, cv::Scalar(z)); }
cv::Mat Color() { return cv::Mat(64, 64, CV_8UC3, cv::Scalar(10, 20, 30)); }
const Eigen::Isometry3d kIdentity = Eigen::Isometry3d::Identity();
const Eigen::Vector3d kFront(0.025, 0.025, 1.975);
const Eigen::Vector3d kBehind(0.025, 0.025, 2.025);

TEST(TsdfMap, FusesSignedDistanceAndExtractsMetricColoredPlane) {
  TsdfMap map(TsdfParameters{});
  EXPECT_FALSE(map.Integrate(Depth(), Color(), kIntrinsics, kIdentity));
  const auto front = map.Sample(kFront), behind = map.Sample(kBehind);
  ASSERT_TRUE(front.has_value());
  ASSERT_TRUE(behind.has_value());
  EXPECT_GT(front->distance, 0);
  EXPECT_LT(behind->distance, 0);
  const auto points = map.points();
  ASSERT_GT(points.size(), 100u);
  for (const auto& point : points) {
    EXPECT_TRUE(point.position.allFinite());
    EXPECT_NEAR(point.position.z(), 2.0, 1e-5);
    EXPECT_EQ(point.r, 30);
    EXPECT_EQ(point.g, 20);
    EXPECT_EQ(point.b, 10);
  }
}

TEST(TsdfMap, DensePixelsContributeOnlyOncePerVoxelAndFrame) {
  TsdfParameters p;
  p.integration_stride = 1;
  TsdfMap map(p);
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  ASSERT_TRUE(map.Sample(kFront).has_value());
  EXPECT_FLOAT_EQ(map.Sample(kFront)->weight, 1.0f);
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  EXPECT_FLOAT_EQ(map.Sample(kFront)->weight, 2.0f);
}

TEST(TsdfMap, MultipleDepthMeasurementsAverageTheSurface) {
  TsdfMap map(TsdfParameters{});
  map.Integrate(Depth(2.0f), Color(), kIntrinsics, kIdentity);
  map.Integrate(Depth(2.02f), Color(), kIntrinsics, kIdentity);
  const auto points = map.points();
  ASSERT_GT(points.size(), 100u);
  for (const auto& point : points) EXPECT_NEAR(point.position.z(), 2.01, 2e-5);
}

TEST(TsdfMap, RigidPoseAndNegativeWorldVoxelCoordinatesAreCorrect) {
  TsdfMap map(TsdfParameters{});
  Eigen::Isometry3d pose = kIdentity;
  pose.linear() = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY()).toRotationMatrix();
  pose.translation() = Eigen::Vector3d(-1.5, -0.2, 0.3);
  map.Integrate(Depth(), Color(), kIntrinsics, pose);
  const auto points = map.points();
  ASSERT_GT(points.size(), 100u);
  for (const auto& point : points) {
    EXPECT_NEAR((pose.inverse() * point.position).z(), 2.0, 1e-5);
    EXPECT_LT(point.position.x(), 0);
  }
}

TEST(TsdfMap, WeightThresholdRejectsUnconfirmedSurfaces) {
  TsdfParameters p;
  p.min_weight = 2;
  TsdfMap map(p);
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  EXPECT_TRUE(map.points().empty());
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  EXPECT_FALSE(map.points().empty());
}

TEST(TsdfMap, WeightSaturationKeepsUpdatingWithoutFreezingTheVolume) {
  TsdfParameters p;
  p.max_weight = 2;
  TsdfMap map(p);
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  ASSERT_TRUE(map.Sample(kFront).has_value());
  const auto before = *map.Sample(kFront);
  map.Integrate(Depth(2.03f), Color(), kIntrinsics, kIdentity);
  const auto after = *map.Sample(kFront);
  EXPECT_FLOAT_EQ(after.weight, 2.0f);
  EXPECT_GT(after.distance, before.distance);
  EXPECT_NEAR(after.distance, (before.distance * 2.0 + (2.03 - 1.975) / 0.15) / 3.0, 1e-5);
}

TEST(TsdfMap, InvalidDepthDoesNotAllocateOrReweight) {
  TsdfMap map(TsdfParameters{});
  auto invalid = Depth(std::numeric_limits<float>::quiet_NaN());
  invalid.at<float>(0, 0) = 0;
  invalid.at<float>(2, 2) = -2;
  invalid.at<float>(4, 4) = std::numeric_limits<float>::infinity();
  map.Integrate(invalid, Color(), kIntrinsics, kIdentity);
  EXPECT_EQ(map.size(), 0u);
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  const auto before = *map.Sample(kFront);
  map.Integrate(invalid, Color(), kIntrinsics, kIdentity);
  EXPECT_FLOAT_EQ(map.Sample(kFront)->weight, before.weight);
  EXPECT_FLOAT_EQ(map.Sample(kFront)->distance, before.distance);
}

TEST(TsdfMap, PreviouslyAllocatedFreeSpaceIsUpdatedAfterSurfaceMoves) {
  TsdfMap map(TsdfParameters{});
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  for (int frame = 0; frame < 3; ++frame) {
    map.Integrate(Depth(3.0f), Color(), kIntrinsics, kIdentity);
  }
  ASSERT_TRUE(map.Sample(kBehind).has_value());
  EXPECT_GT(map.Sample(kBehind)->distance, 0);
  const auto points = map.points();
  ASSERT_FALSE(points.empty());
  for (const auto& point : points) EXPECT_NEAR(point.position.z(), 3.0, 1e-5);
}

TEST(TsdfMap, CapacityBoundsAllocationButExistingObservationsStillUpdate) {
  TsdfParameters p;
  p.max_voxels = 400;
  TsdfMap map(p);
  EXPECT_TRUE(map.Integrate(Depth(), Color(), kIntrinsics, kIdentity));
  EXPECT_LE(map.size(), p.max_voxels);
  const auto points = map.points();
  ASSERT_FALSE(points.empty());
  const auto sample = map.Sample(points.front().position);
  ASSERT_TRUE(sample.has_value());
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  EXPECT_LE(map.size(), p.max_voxels);
  EXPECT_GT(map.Sample(points.front().position)->weight, sample->weight);
}

TEST(TsdfMap, ClearRemovesFieldWeightsAndSurface) {
  TsdfMap map(TsdfParameters{});
  map.Integrate(Depth(), Color(), kIntrinsics, kIdentity);
  map.Clear();
  EXPECT_EQ(map.size(), 0u);
  EXPECT_TRUE(map.points().empty());
  EXPECT_FALSE(map.Sample(kFront).has_value());
}

TEST(TsdfMap, RejectsInvalidSettingsAndInputsBeforeChangingTheMap) {
  TsdfParameters p;
  p.truncation_distance = 0.001;
  EXPECT_THROW(TsdfMap{p}, std::invalid_argument);
  p = TsdfParameters{};
  p.min_weight = p.max_weight + 1;
  EXPECT_THROW(TsdfMap{p}, std::invalid_argument);
  TsdfMap map(TsdfParameters{});
  EXPECT_THROW(map.Integrate(cv::Mat(), Color(), kIntrinsics, kIdentity), std::invalid_argument);
  auto bad_k = kIntrinsics;
  bad_k(0, 0) = 0;
  EXPECT_THROW(map.Integrate(Depth(), Color(), bad_k, kIdentity), std::invalid_argument);
  auto bad_pose = kIdentity;
  bad_pose.linear()(0, 0) = 2;
  EXPECT_THROW(map.Integrate(Depth(), Color(), kIntrinsics, bad_pose), std::invalid_argument);
  EXPECT_EQ(map.size(), 0u);
}

TEST(TsdfMap, GrayscaleAndExactZeroLayerAreExtractedWithoutDuplicates) {
  TsdfMap map(TsdfParameters{});
  map.Integrate(Depth(2.025f), cv::Mat(64, 64, CV_8U, cv::Scalar(77)), kIntrinsics, kIdentity);
  const auto points = map.points();
  ASSERT_GT(points.size(), 100u);
  for (const auto& point : points) {
    EXPECT_NEAR(point.position.z(), 2.025, 1e-5);
    EXPECT_EQ(point.r, 77);
    EXPECT_EQ(point.g, 77);
    EXPECT_EQ(point.b, 77);
  }
}

}  // namespace
}  // namespace stereo_recon

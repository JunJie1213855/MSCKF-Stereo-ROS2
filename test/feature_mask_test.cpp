#include <limits>

#include <gtest/gtest.h>

#include "msckf_vio/feature_mask.h"

namespace msckf_vio {
namespace {

TEST(FeatureMask, RejectsBordersAndNonfiniteTracks) {
  FeatureMask mask(cv::Size(752, 480), 8, 20);
  EXPECT_FALSE(mask.Allows(cv::Point2f(7.9f, 50)));
  EXPECT_FALSE(mask.Allows(cv::Point2f(744, 50)));
  EXPECT_FALSE(mask.Allows(cv::Point2f(50, 472)));
  EXPECT_FALSE(mask.Allows(cv::Point2f(
      std::numeric_limits<float>::quiet_NaN(), 50)));
  EXPECT_FALSE(mask.Allows(cv::Point2f(50,
      std::numeric_limits<float>::infinity())));
  EXPECT_TRUE(mask.Allows(cv::Point2f(8, 8)));
}

TEST(FeatureMask, AppliesSpacingAcrossGridBoundaries) {
  FeatureMask mask(cv::Size(752, 480), 8, 20);
  mask.Add(cv::Point2f(149, 119));
  EXPECT_FALSE(mask.Allows(cv::Point2f(153, 123)));
  EXPECT_FALSE(mask.Allows(cv::Point2f(149, 137)));
  EXPECT_TRUE(mask.Allows(cv::Point2f(180, 150)));
  EXPECT_EQ(0, mask.image().at<uchar>(123, 153));
}

TEST(FeatureMask, RejectsSubpixelPairsInsideMinimumDistance) {
  FeatureMask mask(cv::Size(752, 480), 8, 20);
  const cv::Point2f accepted(100.49f, 100.49f);
  mask.Add(accepted);
  // Exercise positions around the integer-mask boundary used by LK tracks.
  for (int dx = -210; dx <= 210; ++dx) {
    for (int dy = -210; dy <= 210; ++dy) {
      const cv::Point2f candidate(accepted.x + dx * 0.1f,
                                accepted.y + dy * 0.1f);
      if (cv::norm(candidate - accepted) < 20) {
        EXPECT_FALSE(mask.Allows(candidate));
      }
    }
  }
}

TEST(FeatureMask, GridCoversRemainderColumnsWithoutOverflow) {
  const cv::Size size(752, 480);
  EXPECT_EQ(19, FeatureGridCode(cv::Point2f(751, 479), size, 4, 5));
  EXPECT_EQ(4, FeatureGridCode(cv::Point2f(750, 20), size, 4, 5));
  EXPECT_EQ(0, FeatureGridCode(cv::Point2f(0, 0), size, 4, 5));
  EXPECT_EQ(19, FeatureGridCode(cv::Point2f(743.9f, 471.9f), size, 4, 5));
}

}  // namespace
}  // namespace msckf_vio

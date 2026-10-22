#include <cmath>
#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

#include "msckf_vio/stereo/stereo_matcher.h"

namespace msckf_vio::stereo {
namespace {

TEST(TensorRtRuntime, IgevGpuInferencePreservesDisparityAndCropsPadding) {
  const char* directory = std::getenv("MSCKF_TRT_SMOKE_DIR");
  if (!directory) GTEST_SKIP() << "Set MSCKF_TRT_SMOKE_DIR to generated engines";
  IgevParameters parameters;
  parameters.engine_path = std::string(directory) + "/igev_smoke.engine";
  IgevMatcher matcher(parameters);
  matcher.ValidateInputSize(cv::Size(60, 30));
  const cv::Mat left(30, 60, CV_8UC1, cv::Scalar(70));
  const cv::Mat right(30, 60, CV_8UC1, cv::Scalar(62));
  const auto disparity = matcher.Compute(left, right);
  ASSERT_EQ(CV_32FC1, disparity.type());
  ASSERT_EQ(left.size(), disparity.size());
  EXPECT_LT(cv::norm(disparity, cv::Mat(left.size(), CV_32F, cv::Scalar(16)),
                     cv::NORM_INF), 1e-5);
  // A second frame exercises reuse of the same stream/context/buffers.
  const auto next = matcher.Compute(right, right);
  EXPECT_LT(cv::norm(next, cv::Mat(left.size(), CV_32F, cv::Scalar(8)),
                     cv::NORM_INF), 1e-5);
}

TEST(TensorRtRuntime, S2m2GpuInferenceAppliesQualityMaskAtCroppedPixel) {
  const char* directory = std::getenv("MSCKF_TRT_SMOKE_DIR");
  if (!directory) GTEST_SKIP() << "Set MSCKF_TRT_SMOKE_DIR to generated engines";
  S2m2Parameters parameters;
  parameters.engine_path = std::string(directory) + "/s2m2_smoke.engine";
  S2m2Matcher matcher(parameters);
  matcher.ValidateInputSize(cv::Size(60, 34));
  const cv::Mat left(34, 60, CV_8UC3, cv::Scalar(70, 70, 70));
  const cv::Mat right(34, 60, CV_8UC3, cv::Scalar(62, 62, 62));
  const auto disparity = matcher.Compute(left, right);
  ASSERT_EQ(CV_32FC1, disparity.type());
  ASSERT_EQ(left.size(), disparity.size());
  int invalid = 0;
  for (int y = 0; y < disparity.rows; ++y) {
    for (int x = 0; x < disparity.cols; ++x) {
      if (y == 3 && x == 5) {
        EXPECT_TRUE(std::isnan(disparity.at<float>(y, x)));
        ++invalid;
      } else {
        EXPECT_NEAR(16.0f, disparity.at<float>(y, x), 1e-5f);
      }
    }
  }
  EXPECT_EQ(1, invalid);
}

}  // namespace
}  // namespace msckf_vio::stereo

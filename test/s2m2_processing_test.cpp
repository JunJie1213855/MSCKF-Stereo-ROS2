#include "stereo_recon/stereo/s2m2_processing.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "gtest/gtest.h"

#include "stereo_recon/stereo/stereo_matcher.h"

namespace stereo_recon {

TEST(S2m2Input, RgbUint8NchwAndCenteredEurocPadding) {
  cv::Mat image(480, 752, CV_8UC3, cv::Scalar(10, 20, 255));
  const auto input = PrepareS2m2Input(image, image, cv::Size(768, 480));
  ASSERT_EQ(input.left_blob.type(), CV_8UC1);
  EXPECT_EQ(input.left_blob.size[0], 1);
  EXPECT_EQ(input.left_blob.size[1], 3);
  EXPECT_EQ(input.left_blob.size[2], 480);
  EXPECT_EQ(input.left_blob.size[3], 768);
  EXPECT_EQ(input.image_roi, cv::Rect(8, 0, 752, 480));
  EXPECT_EQ(input.left_blob.ptr<uint8_t>(0, 0)[8], 255);
  EXPECT_EQ(input.left_blob.ptr<uint8_t>(0, 1)[8], 20);
  EXPECT_EQ(input.left_blob.ptr<uint8_t>(0, 2)[8], 10);
  EXPECT_EQ(cv::norm(input.left_blob, input.right_blob), 0);
}

TEST(S2m2Input,
     PaddingUsesUpstreamPoolingAndTruncationRatherThanZeroOrReplication) {
  // For H=32,W=33 -> 32x64, adaptive pooling has one bin. The padded
  // border is 255*33/64=131.484375 and torch uint8 conversion truncates.
  cv::Mat image(32, 33, CV_8UC1, cv::Scalar(255));
  const auto input = PrepareS2m2Input(image, image, cv::Size(64, 32));
  EXPECT_EQ(input.image_roi, cv::Rect(15, 0, 33, 32));
  for (int c = 0; c < 3; ++c) {
    const auto* data = input.left_blob.ptr<uint8_t>(0, c);
    EXPECT_EQ(data[0], 131);
    EXPECT_EQ(data[14], 131);
    EXPECT_EQ(data[15], 255);
    EXPECT_EQ(data[47], 255);
    EXPECT_EQ(data[48], 131);
  }
}

TEST(S2m2Input, HandlesRoiAndIndependentStereoImages) {
  cv::Mat parent(64, 64, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::Mat image = parent(cv::Rect(2, 2, 33, 32));
  image.setTo(cv::Scalar(0, 0, 128));
  cv::Mat right(32, 33, CV_8UC1, cv::Scalar(64));
  const auto input = PrepareS2m2Input(image, right, cv::Size(64, 32));
  EXPECT_EQ(input.left_blob.ptr<uint8_t>(0, 0)[0], 66);
  EXPECT_EQ(input.left_blob.ptr<uint8_t>(0, 1)[0], 0);
  EXPECT_EQ(input.right_blob.ptr<uint8_t>(0, 2)[0], 33);
}

TEST(S2m2Input, RejectsTransposedEngineAndImplicitResizing) {
  cv::Mat euroc(480, 752, CV_8UC1, cv::Scalar(100));
  EXPECT_THROW(PrepareS2m2Input(euroc, euroc, cv::Size(480, 768)),
               std::invalid_argument);
  EXPECT_THROW(PrepareS2m2Input(euroc, cv::Mat(), cv::Size(768, 480)),
               std::invalid_argument);
  EXPECT_THROW(PrepareS2m2Input(euroc, euroc, cv::Size(770, 480)),
               std::invalid_argument);
  cv::Mat floats(32, 32, CV_32F);
  EXPECT_THROW(PrepareS2m2Input(floats, floats, cv::Size(32, 32)),
               std::invalid_argument);
}

TEST(S2m2Output, CropsAllOutputsAndPreservesDisparityPixels) {
  cv::Mat image(32, 33, CV_8UC1, cv::Scalar(100));
  const auto input = PrepareS2m2Input(image, image, cv::Size(64, 32));
  const int shape[] = {1, 1, 32, 64};
  cv::Mat d(4, shape, CV_32F, cv::Scalar(16.5));
  cv::Mat o(4, shape, CV_32F, cv::Scalar(0.9)), c = o.clone();
  // Just below / at the strict upstream boundaries, and an occluded pixel.
  c.ptr<float>()[15] = 0.09f;
  o.ptr<float>()[16] = 0.5f;
  o.ptr<float>()[17] = 0.0f;
  c.ptr<float>()[18] = 0.1f;
  const auto result = RestoreS2m2Disparity(d, o, c, input, 0.1, 0.5);
  EXPECT_EQ(result.size(), image.size());
  EXPECT_EQ(result.type(), CV_32FC1);
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isnan(result.at<float>(0, i)));
  }
  EXPECT_FLOAT_EQ(result.at<float>(0, 4), 16.5);
  d.setTo(cv::Scalar(1));
  EXPECT_FLOAT_EQ(result.at<float>(0, 4), 16.5);
}

TEST(S2m2Output, RejectsNonfiniteMeasurementsAndSupportsFp16) {
  cv::Mat image(32, 32, CV_8UC1, cv::Scalar(100));
  const auto input = PrepareS2m2Input(image, image, image.size());
  cv::Mat d(32, 32, CV_32F, cv::Scalar(8));
  cv::Mat o(32, 32, CV_32F, cv::Scalar(1)), c = o.clone();
  d.at<float>(0, 0) = -1;
  d.at<float>(0, 1) = 0;
  d.at<float>(0, 2) = std::numeric_limits<float>::infinity();
  c.at<float>(0, 3) = std::numeric_limits<float>::quiet_NaN();
  o.at<float>(0, 4) = std::numeric_limits<float>::infinity();
  cv::Mat half;
  d.convertTo(half, CV_16F);
  const auto result = RestoreS2m2Disparity(half, o, c, input, 0.1, 0.5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(std::isnan(result.at<float>(0, i)));
  }
  EXPECT_FLOAT_EQ(result.at<float>(0, 5), 8);
  EXPECT_THROW(
      RestoreS2m2Disparity(d, cv::Mat(16, 16, CV_32F), c, input, 0.1, 0.5),
      std::invalid_argument);
  EXPECT_THROW(RestoreS2m2Disparity(d, o, c, input, -1, 0.5),
               std::invalid_argument);
}

TEST(S2m2Backend, MissingEngineOrDisabledBuildGivesAnException) {
  S2m2Parameters parameters;
  parameters.engine_path = "/nonexistent/msckf_s2m2.engine";
  EXPECT_THROW(CreateStereoMatcher("s2m2", SgbmParameters{}, IgevParameters{},
                                   parameters),
               std::runtime_error);
}
}  // namespace stereo_recon

#include "msckf_vio/stereo/igev_processing.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "gtest/gtest.h"

#include "msckf_vio/stereo/stereo_matcher.h"

namespace msckf_vio::stereo {

TEST(IgevInput, ConvertsBgrToRgbNchwWithoutRenormalizing) {
  const cv::Mat image(2, 3, CV_8UC3, cv::Scalar(10, 20, 255));
  const auto input =
      PrepareIgevInput(image, image, cv::Size(4, 4), "right_bottom");
  ASSERT_EQ(input.left_blob.dims, 4);
  EXPECT_EQ(input.left_blob.type(), CV_32FC1);
  EXPECT_EQ(input.left_blob.size[0], 1);
  EXPECT_EQ(input.left_blob.size[1], 3);
  EXPECT_EQ(input.left_blob.size[2], 4);
  EXPECT_EQ(input.left_blob.size[3], 4);
  EXPECT_EQ(input.image_roi, cv::Rect(0, 0, 3, 2));
  EXPECT_FLOAT_EQ(input.left_blob.ptr<float>(0, 0)[0], 255);
  EXPECT_FLOAT_EQ(input.left_blob.ptr<float>(0, 1)[0], 20);
  EXPECT_FLOAT_EQ(input.left_blob.ptr<float>(0, 2)[0], 10);
  EXPECT_FLOAT_EQ(input.left_blob.ptr<float>(0, 0)[3], 0);
  EXPECT_FLOAT_EQ(input.left_blob.ptr<float>(0, 0)[8], 0);
  EXPECT_EQ(cv::norm(input.left_blob, input.right_blob), 0);
}

TEST(IgevInput, ReplicatesMonoChannelsAndSupportsSintelPadding) {
  const cv::Mat image(3, 4, CV_8UC1, cv::Scalar(137));
  const auto input = PrepareIgevInput(image, image, cv::Size(8, 8), "sintel");
  EXPECT_EQ(input.image_roi, cv::Rect(2, 2, 4, 3));
  for (int channel = 0; channel < 3; ++channel) {
    for (int i = 0; i < 64; ++i) {
      EXPECT_FLOAT_EQ(input.left_blob.ptr<float>(0, channel)[i], 137);
    }
  }
}

TEST(IgevInput, PaddingDoesNotReadPixelsOutsideImageRoi) {
  cv::Mat parent(6, 6, CV_8UC3, cv::Scalar(255, 255, 255));
  auto image = parent(cv::Rect(1, 1, 3, 3));
  image.setTo(cv::Scalar(10, 20, 30));
  const auto input =
      PrepareIgevInput(image, image, cv::Size(4, 4), "right_bottom");
  EXPECT_FLOAT_EQ(input.left_blob.ptr<float>(0, 0)[3], 0);
  EXPECT_FLOAT_EQ(input.left_blob.ptr<float>(0, 0)[0], 30);
}

TEST(IgevInput, RejectsImplicitResizingAndUnsupportedFormats) {
  cv::Mat image(10, 10, CV_8UC1);
  EXPECT_THROW(PrepareIgevInput(image, image, cv::Size(8, 8), "right_bottom"),
               std::invalid_argument);
  EXPECT_THROW(
      PrepareIgevInput(image, cv::Mat(), cv::Size(32, 32), "right_bottom"),
      std::invalid_argument);
  EXPECT_THROW(PrepareIgevInput(image, image, cv::Size(32, 32), "unknown"),
               std::invalid_argument);
  cv::Mat floats(10, 10, CV_32FC1);
  EXPECT_THROW(
      PrepareIgevInput(floats, floats, cv::Size(32, 32), "right_bottom"),
      std::invalid_argument);
}

TEST(IgevOutput, RestoresEurocSizeAndPreservesPixelDisparity) {
  cv::Mat image(480, 752, CV_8UC1, cv::Scalar(100));
  const auto input =
      PrepareIgevInput(image, image, cv::Size(768, 480), "right_bottom");
  const int shape[] = {1, 1, 480, 768};
  cv::Mat output(4, shape, CV_32FC1, cv::Scalar(16.5));
  const auto disparity = RestoreIgevDisparity(output, input);
  EXPECT_EQ(disparity.size(), image.size());
  EXPECT_EQ(disparity.type(), CV_32FC1);
  EXPECT_FLOAT_EQ(disparity.at<float>(240, 751), 16.5);
  output.setTo(cv::Scalar(1));
  EXPECT_FLOAT_EQ(disparity.at<float>(240, 751),
                  16.5);  // Owns its output memory.
}

TEST(IgevOutput, UnpadsAtTheCorrectOffsetAndAcceptsFp16) {
  cv::Mat image(3, 4, CV_8UC1, cv::Scalar(100));
  const auto input = PrepareIgevInput(image, image, cv::Size(8, 8), "sintel");
  cv::Mat grid(8, 8, CV_32FC1);
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      grid.at<float>(y, x) = static_cast<float>(y * 8 + x + 1);
    }
  }
  cv::Mat half;
  grid.convertTo(half, CV_16F);
  const auto disparity = RestoreIgevDisparity(half, input);
  EXPECT_FLOAT_EQ(disparity.at<float>(0, 0), 19);
  EXPECT_FLOAT_EQ(disparity.at<float>(2, 3), 38);
}

TEST(IgevOutput, ConvertsInvalidValuesToNanAndRejectsAmbiguousShapes) {
  cv::Mat image(4, 4, CV_8UC1, cv::Scalar(100));
  const auto input =
      PrepareIgevInput(image, image, image.size(), "right_bottom");
  const int shape[] = {1, 4, 4};
  cv::Mat output(3, shape, CV_32FC1, cv::Scalar(8));
  auto* values = output.ptr<float>();
  values[0] = -1;
  values[1] = 0;
  values[2] = std::numeric_limits<float>::quiet_NaN();
  values[3] = std::numeric_limits<float>::infinity();
  const auto disparity = RestoreIgevDisparity(output, input);
  for (int x = 0; x < 4; ++x) {
    EXPECT_TRUE(std::isnan(disparity.at<float>(0, x)));
  }
  EXPECT_FLOAT_EQ(disparity.at<float>(1, 0), 8);
  EXPECT_THROW(RestoreIgevDisparity(cv::Mat(2, 2, CV_32FC1), input),
               std::invalid_argument);
  EXPECT_THROW(RestoreIgevDisparity(cv::Mat(4, 4, CV_8UC1), input),
               std::invalid_argument);
  const int multiple[] = {1, 2, 4, 4};
  EXPECT_THROW(RestoreIgevDisparity(cv::Mat(4, multiple, CV_32FC1), input),
               std::invalid_argument);
}

TEST(IgevBackend, MissingEngineOrDisabledBuildGivesAnException) {
  IgevParameters parameters;
  parameters.engine_path = "/nonexistent/msckf_igev.engine";
  EXPECT_THROW(CreateStereoMatcher("igev", SgbmParameters{}, parameters),
               std::runtime_error);
}
}  // namespace msckf_vio::stereo

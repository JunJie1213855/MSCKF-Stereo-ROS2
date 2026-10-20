#pragma once

#include <string>
#include <opencv2/core.hpp>

namespace msckf_vio::stereo {

// CPU-side IGEV contract, also used in tests without a CUDA device.
struct IgevInput {
  cv::Mat left_blob, right_blob;  // RGB FP32 NCHW, values in [0,255].
  cv::Size network_size;
  cv::Rect image_roi;
};

// right_bottom: bottom/right constant-zero padding, matching thirdparty's
// IGEV example. sintel: symmetric edge replication, matching InputPadder.
IgevInput prepareIgevInput(const cv::Mat& left, const cv::Mat& right,
                          cv::Size network_size, const std::string& padding_mode);

// Accepts [H,W], [1,H,W] or [1,1,H,W] FP32/FP16 final disparity in network
// input pixels. Other resolutions are rejected rather than guessing units.
cv::Mat restoreIgevDisparity(const cv::Mat& output, const IgevInput& input);

}  // namespace msckf_vio::stereo

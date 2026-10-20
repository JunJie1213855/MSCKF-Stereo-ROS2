#ifndef MSCKF_VIO_STEREO_IGEV_PROCESSING_H_
#define MSCKF_VIO_STEREO_IGEV_PROCESSING_H_

#include <string>

#include "opencv2/core.hpp"

#include "msckf_vio/stereo/stereo_tensor_input.h"

namespace msckf_vio::stereo {

using IgevInput = StereoTensorInput;

// right_bottom: bottom/right constant-zero padding, matching thirdparty's
// IGEV example. sintel: symmetric edge replication, matching InputPadder.
IgevInput PrepareIgevInput(const cv::Mat& left, const cv::Mat& right,
                           cv::Size network_size,
                           const std::string& padding_mode);

// Accepts [H,W], [1,H,W] or [1,1,H,W] FP32/FP16 final disparity in network
// input pixels. Other resolutions are rejected rather than guessing units.
cv::Mat RestoreIgevDisparity(const cv::Mat& output, const IgevInput& input);

}  // namespace msckf_vio::stereo

#endif  // MSCKF_VIO_STEREO_IGEV_PROCESSING_H_

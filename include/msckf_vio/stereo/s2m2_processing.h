#ifndef MSCKF_VIO_STEREO_S2M2_PROCESSING_H_
#define MSCKF_VIO_STEREO_S2M2_PROCESSING_H_

#include "opencv2/core.hpp"

#include "msckf_vio/stereo/stereo_tensor_input.h"

namespace msckf_vio::stereo {

// Matches s2m2/core/utils/image_utils.py:image_pad(factor=32): centered
// padding filled by adaptive average pooling + bilinear interpolation, then
// the original image is restored exactly. RGB UINT8 NCHW; no normalization.
StereoTensorInput PrepareS2m2Input(const cv::Mat& left, const cv::Mat& right,
                                   cv::Size network_size);

// output_occ is VISIBILITY: larger values are more visible. Thresholds use
// strict comparisons conf > min_confidence and occ > min_visibility.
cv::Mat RestoreS2m2Disparity(const cv::Mat& disparity,
                             const cv::Mat& visibility,
                             const cv::Mat& confidence,
                             const StereoTensorInput& input,
                             double min_confidence, double min_visibility);

}  // namespace msckf_vio::stereo

#endif  // MSCKF_VIO_STEREO_S2M2_PROCESSING_H_

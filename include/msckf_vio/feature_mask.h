#ifndef MSCKF_VIO_FEATURE_MASK_H
#define MSCKF_VIO_FEATURE_MASK_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace msckf_vio {

// Greedy spatial selection shared by tracked features and new detections.
// The image mask speeds rejection; the floating-point check also covers LK
// positions near a rasterized circle's boundary.
class FeatureMask {
 public:
  FeatureMask(cv::Size size, int border, int min_distance)
      : border_(border), min_distance_(min_distance),
        mask_(size, CV_8UC1, cv::Scalar(0)) {
    if (border < 0 || min_distance <= 0 ||
        size.width <= 2 * border || size.height <= 2 * border) {
      throw std::invalid_argument("Invalid feature mask geometry");
    }
    mask_(cv::Rect(border, border, size.width - 2 * border,
                   size.height - 2 * border)).setTo(255);
  }

  static bool InBorder(const cv::Point2f& point, cv::Size size, int border) {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           point.x >= border && point.y >= border &&
           point.x < size.width - border &&
           point.y < size.height - border;
  }

  bool Allows(const cv::Point2f& point) const {
    if (!InBorder(point, mask_.size(), border_)) return false;
    const int x = cvRound(point.x);
    const int y = cvRound(point.y);
    if (x >= mask_.cols || y >= mask_.rows || !mask_.at<uchar>(y, x)) {
      return false;
    }
    const double squared_distance =
        static_cast<double>(min_distance_) * min_distance_;
    for (const auto& accepted : accepted_) {
      const cv::Point2f difference = point - accepted;
      if (difference.dot(difference) < squared_distance) return false;
    }
    return true;
  }

  void Add(const cv::Point2f& point) {
    accepted_.push_back(point);
    cv::circle(mask_, point, min_distance_, cv::Scalar(0), -1);
  }

  const cv::Mat& image() const { return mask_; }

 private:
  int border_;
  int min_distance_;
  cv::Mat mask_;
  std::vector<cv::Point2f> accepted_;
};

// Scale by the full image size: integer cell widths lose the final columns
// when image dimensions are not divisible by the grid dimensions.
inline int FeatureGridCode(const cv::Point2f& point, cv::Size size,
                           int rows, int cols) {
  const int row = std::min(rows - 1,
      std::max(0, static_cast<int>(point.y * rows / size.height)));
  const int col = std::min(cols - 1,
      std::max(0, static_cast<int>(point.x * cols / size.width)));
  return row * cols + col;
}

}  // namespace msckf_vio

#endif  // MSCKF_VIO_FEATURE_MASK_H

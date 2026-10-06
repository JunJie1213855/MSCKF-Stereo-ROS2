/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <msckf_vio/utils.h>
#include <msckf_vio/param_utils.hpp>
#include <vector>

namespace msckf_vio {
namespace utils {

Eigen::Isometry3d getTransformEigen(const rclcpp::Node::SharedPtr& nh,
                                    const std::string& field) {
  Eigen::Isometry3d T;
  cv::Mat c = getTransformCV(nh, field);

  T.linear()(0, 0)   = c.at<double>(0, 0);
  T.linear()(0, 1)   = c.at<double>(0, 1);
  T.linear()(0, 2)   = c.at<double>(0, 2);
  T.linear()(1, 0)   = c.at<double>(1, 0);
  T.linear()(1, 1)   = c.at<double>(1, 1);
  T.linear()(1, 2)   = c.at<double>(1, 2);
  T.linear()(2, 0)   = c.at<double>(2, 0);
  T.linear()(2, 1)   = c.at<double>(2, 1);
  T.linear()(2, 2)   = c.at<double>(2, 2);
  T.translation()(0) = c.at<double>(0, 3);
  T.translation()(1) = c.at<double>(1, 3);
  T.translation()(2) = c.at<double>(2, 3);
  return T;
}

cv::Mat getTransformCV(const rclcpp::Node::SharedPtr& nh,
                       const std::string& field) {
  cv::Mat T;
  try {
    T = getVec16Transform(nh, field);
  } catch (std::runtime_error& e) {
    std::string msg = "cannot read transform " + field + " error: " + e.what();
    RCLCPP_ERROR_STREAM(nh->get_logger(), msg);
    throw std::runtime_error(msg);
  }
  return T;
}

/*
 * The ROS 2 parameter interface only carries flat arrays, so a Kalibr
 * style transform is stored as 16 row-major doubles (the bracketed form
 * used in config/camchain-*.yaml) rather than as a 4x4 nested array.
 */
cv::Mat getVec16Transform(const rclcpp::Node::SharedPtr& nh,
                          const std::string& field) {
  std::vector<double> v =
    params::get<std::vector<double>>(nh, field, std::vector<double>());
  if (v.size() != 16) {
    throw std::runtime_error("invalid vec16!");
  }
  cv::Mat T = cv::Mat(v).clone().reshape(1, 4); // one channel 4 rows
  return T;
}

} // namespace utils
} // namespace msckf_vio

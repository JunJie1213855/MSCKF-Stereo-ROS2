/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <msckf_vio/msckf_vio.h>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  // Parameters are supplied through a parameter file, so declare whatever
  // was provided before MsckfVio reads them.
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("vio", options);

  msckf_vio::MsckfVioPtr msckf_vio_ptr(new msckf_vio::MsckfVio(node));
  if (!msckf_vio_ptr->initialize()) {
    RCLCPP_ERROR(node->get_logger(), "Cannot initialize MSCKF VIO...");
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

/*
 * Helpers for reading node parameters.
 *
 * rcl_yaml_param_parser infers the type of a parameter from the YAML text,
 * which does not always agree with the C++ type the code wants: a
 * `frame_rate: 20` entry becomes an integer while frame_rate is a double.
 * Parameter::get_value<T>() throws on any such mismatch, so numeric values
 * are converted explicitly here instead.
 *
 * Nodes using these helpers must be created with
 * rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
 * so that values supplied through a parameter file are already declared by
 * the time the parameters are read.
 */

#ifndef MSCKF_VIO_PARAM_UTILS_HPP
#define MSCKF_VIO_PARAM_UTILS_HPP

#include <string>
#include <type_traits>
#include <vector>

#include <rclcpp/rclcpp.hpp>

namespace msckf_vio {
namespace params {

/*
 * @brief convert Convert a stored parameter to the requested C++ type.
 */
template <typename T>
T convert(const rclcpp::Parameter& parameter, const std::string& name) {
  const rclcpp::ParameterType type = parameter.get_type();

  if constexpr (std::is_same_v<T, bool>) {
    if (type == rclcpp::ParameterType::PARAMETER_BOOL) {
      return parameter.as_bool();
    }
  } else if constexpr (std::is_same_v<T, std::string>) {
    if (type == rclcpp::ParameterType::PARAMETER_STRING) {
      return parameter.as_string();
    }
  } else if constexpr (std::is_floating_point_v<T>) {
    if (type == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      return static_cast<T>(parameter.as_double());
    }
    if (type == rclcpp::ParameterType::PARAMETER_INTEGER) {
      return static_cast<T>(parameter.as_int());
    }
  } else if constexpr (std::is_integral_v<T>) {
    if (type == rclcpp::ParameterType::PARAMETER_INTEGER) {
      return static_cast<T>(parameter.as_int());
    }
    if (type == rclcpp::ParameterType::PARAMETER_DOUBLE) {
      return static_cast<T>(parameter.as_double());
    }
  } else if constexpr (std::is_same_v<T, std::vector<double>>) {
    if (type == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
      return parameter.as_double_array();
    }
    if (type == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
      const std::vector<int64_t> integers = parameter.as_integer_array();
      return std::vector<double>(integers.begin(), integers.end());
    }
  } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
    if (type == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
      return parameter.as_integer_array();
    }
  } else {
    return parameter.get_value<T>();
  }

  throw rclcpp::exceptions::InvalidParameterTypeException(
      name, "parameter has an unexpected type");
}

/*
 * @brief get Read a parameter, declaring it with `default_value` the first
 *    time it is seen. Parameters already declared from a parameter file
 *    take precedence over the default.
 */
template <typename T>
T get(const rclcpp::Node::SharedPtr& nh, const std::string& name,
      const T& default_value) {
  if (!nh->has_parameter(name)) {
    nh->declare_parameter<T>(name, default_value);
    return default_value;
  }
  return convert<T>(nh->get_parameter(name), name);
}

} // namespace params
} // namespace msckf_vio

#endif

cmake_minimum_required(VERSION 3.10)
project(stereo_recon)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_library(stereo_recon_lib SHARED
  src/stereo/stereo_matcher.cpp
  src/stereo/igev_processing.cpp
  src/stereo/s2m2_processing.cpp
  src/stereo/reconstruction.cpp
  src/stereo/tsdf_map.cpp
)
target_compile_features(stereo_recon_lib PUBLIC cxx_std_17)
target_include_directories(stereo_recon_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(stereo_recon_lib PUBLIC Eigen3::Eigen ${OpenCV_LIBS})

add_executable(stereo_reconstruction_node src/stereo_reconstruction_node.cpp)
ament_target_dependencies(stereo_reconstruction_node
  rclcpp std_msgs sensor_msgs nav_msgs geometry_msgs std_srvs
  tf2_ros cv_bridge message_filters)
target_link_libraries(stereo_reconstruction_node stereo_recon_lib Threads::Threads)
include(cmake/TensorRTBackend.cmake)

install(TARGETS stereo_recon_lib stereo_reconstruction_node
  EXPORT stereo_reconTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config rviz scripts docs DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)
install(FILES README.md LICENSE.txt DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  foreach(test_name IN ITEMS stereo_reconstruction igev_processing s2m2_processing tsdf_map)
    ament_add_gtest(test_${test_name} test/${test_name}_test.cpp)
    target_link_libraries(test_${test_name} stereo_recon_lib)
  endforeach()
  if(STEREO_RECON_ENABLE_TENSORRT)
    ament_add_gtest(test_tensorrt_runtime test/tensorrt_runtime_test.cpp)
    target_link_libraries(test_tensorrt_runtime stereo_recon_lib)
  endif()
endif()

ament_export_targets(stereo_reconTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 OpenCV)
ament_package()

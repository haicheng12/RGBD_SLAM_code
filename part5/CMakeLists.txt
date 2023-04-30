cmake_minimum_required(VERSION 3.0)

project(part5)

set(CMAKE_CXX_COMPILER "g++")
set( CMAKE_BUILD_TYPE Debug  )
set(CMAKE_PREFIX_PATH /home/bluesky/OpenCV/opencv-4.5.5/build)

find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(visual_odometry
  src/slam_base.cpp
  src/visual_odometry.cpp)
target_link_libraries(visual_odometry ${PCL_LIBRARIES} ${OpenCV_LIBS})

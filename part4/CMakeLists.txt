cmake_minimum_required(VERSION 3.0)

project(part4)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_PREFIX_PATH /home/bluesky/OpenCV/opencv-4.5.5/build)

find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(join_point_cloud src/join_point_cloud.cpp
  src/slam_base.cpp)
target_link_libraries(join_point_cloud ${PCL_LIBRARIES} ${OpenCV_LIBS})

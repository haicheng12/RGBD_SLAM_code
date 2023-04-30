cmake_minimum_required(VERSION 3.0)

project(orb_odom_part1)

set (CMAKE_C_COMPILER /usr/bin/gcc)
set (CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_PREFIX_PATH /home/bluesky/OpenCV/opencv-4.5.5/build)

find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(run_vo 
  src/camera.cpp
  src/frame.cpp
  src/map.cpp
  src/mappoints.cpp
  src/run_vo.cpp
  src/system.cpp
  src/viewer.cpp
  src/visual_odometry.cpp)
target_link_libraries(run_vo 
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS} 
  ${Pangolin_LIBRARIES})

cmake_minimum_required(VERSION 3.0)

project(part6)

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH /home/bluesky/OpenCV/opencv-4.5.5/build)

find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(slam_end
  src/slam_end.cpp)
target_link_libraries(slam_end 
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS} 
  ${EIGEN3_LIBS} 
  ${Pangolin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so)

cmake_minimum_required(VERSION 3.0)
project(slam)

set (CMAKE_C_COMPILER /usr/bin/gcc)
set (CMAKE_CXX_COMPILER /usr/bin/g++)
set(CMAKE_PREFIX_PATH /home/bluesky/OpenCV/opencv-4.5.5/build)

find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(generate_pointCloud src/generate_pointCloud.cpp)
target_link_libraries(generate_pointCloud ${PCL_LIBRARIES} ${OpenCV_LIBS})

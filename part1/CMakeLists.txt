cmake_minimum_required(VERSION 3.0)
project(slam)
set (CMAKE_C_COMPILER /usr/bin/gcc)
set (CMAKE_CXX_COMPILER /usr/bin/g++)

include_directories(${PROJECT_SOURCE_DIR}/include)
	
add_executable( main src/main.cpp )

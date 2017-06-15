cmake_minimum_required(VERSION 2.8)
project(stitcher)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED)
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP found.")
endif()
add_executable(stitcher descriptor.cpp harris.cpp misc.cpp main.cpp)
set(CMAKE_BUILD_TYPE Debug)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(stitcher ${OpenCV_LIBS})
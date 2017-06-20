cmake_minimum_required(VERSION 2.8)

project(image-stitcher CXX)

set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(EIGEN_DIR ${EXT_DIR}/eigen)
set(APP "image-stitcher")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP found.")
endif()

file(GLOB MAIN_SRC "./src/*.cpp")
add_executable(${APP} ${MAIN_SRC})

set(CMAKE_BUILD_TYPE Debug)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN_DIR} "inc")

target_link_libraries(${APP} ${OpenCV_LIBS})

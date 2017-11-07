cmake_minimum_required(VERSION 2.8)

project(OMG)

find_package(OpenCV REQUIRED)
find_package( Eigen3 REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set (CMAKE_CXX_FLAGS "-O2")

set(CMAKE_BUILD_TYPE Release)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(test_OMG run_OMG.cpp Dataset.cpp OMGUtilities.hpp)

target_link_libraries(test_OMG ${OpenCV_LIBS} ${Eigen_LIBRARIES})



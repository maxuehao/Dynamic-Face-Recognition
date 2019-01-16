cmake_minimum_required(VERSION 2.6)
project(DFR)

SET(CMAKE_C_COMPILER g++)
SET(CMAK_CXX_COMPILER g++)
add_compile_options(-std=c++11)
add_compile_options(-m64)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})

include_directories(include 
    ${Caffe_INCLUDE_DIRS}
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DFR main.cpp mtcnn.cpp feature.cpp functional.cpp)
target_link_libraries(DFR 
    ${Caffe_LIBRARIES}
    ${OpenCV_LIBS}
)



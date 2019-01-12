cmake_minimum_required(VERSION 2.6)
project(mtcnn_c)

SET(CMAKE_C_COMPILER g++)
SET(CMAK_CXX_COMPILER g++)
add_compile_options(-std=c++11)
add_compile_options(-m64)

#设置caffe的根目录，需改成你的路径
set(Caffe_DIR /home/m/caffe)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})

include_directories(include 
    ${Caffe_INCLUDE_DIRS}
)


include_directories(
    ${mtcnn_c_SOURCE_DIR}/include/ 
    /usr/local/cuda/include)
link_directories(
    ${mtcnn_c_SOURCE_DIR}/lib/
    /usr/local/cuda/lib64)

add_executable(mtcnn_c main.cpp mtcnn.cpp feature.cpp functional.cpp)
target_link_libraries(mtcnn_c 
    ${Caffe_LIBRARIES}
    -lopencv_core 
    -lopencv_highgui 
    -lopencv_imgproc
)



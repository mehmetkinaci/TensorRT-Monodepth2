cmake_minimum_required(VERSION 2.6)
project(destiny_experimental)



add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


#target_link_libraries(cudart nvinfer)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/destiny_padding.cu)
target_link_libraries(decodeplugin nvinfer cudart)


add_executable(experimental ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(experimental nvinfer)
target_link_libraries(experimental cudart)
target_link_libraries(experimental ${OpenCV_LIBS})
target_link_libraries(experimental decodeplugin)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)

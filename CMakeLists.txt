cmake_minimum_required(VERSION 3.22)
project(AD VERSION 1.0 LANGUAGES CUDA CXX)

find_package(CUDA)
find_package(Threads)
find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE SRCS src/*.cu src/*.cpp src/*.h)

add_executable(AD ${SRCS})

set_property(TARGET AD PROPERTY CUDA_ARCHITECTURES OFF)
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

set_target_properties(AD PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -openmp ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo ")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dlto")


# target_compile_options(AD PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)

target_link_libraries(AD ${CUDA_LIBRARIES})
target_link_libraries(AD ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(AD ${CUDA_CUDNN_LIBRARIES})
target_link_libraries(AD ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(AD "C:/Program Files/NVIDIA/CUDNN/v 8.4/lib/cudnn.lib")

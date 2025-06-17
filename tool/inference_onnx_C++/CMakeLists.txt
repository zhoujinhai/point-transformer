cmake_minimum_required(VERSION 3.9)

project(seg VERSION 1.0)

# >>> build type 
# set(CMAKE_BUILD_TYPE "Debug")				# 指定生成的版本
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# <<<


# >>> CXX11 
set(CMAKE_CXX_STANDARD 11)				# C++ 11 编译器
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# <<< CXX11

set(ONNX_ROOT "F:/code/onnxruntime/onnxruntime-win-x64-1.14.1")
include_directories(${ONNX_ROOT}/include)

set(ONNX_LIBRARIES 
	"${ONNX_ROOT}/lib/onnxruntime.lib"
	"${ONNX_ROOT}/lib/onnxruntime_providers_shared.lib"
CACHE INTERNAL "")
set(ONNX_RUNTIME_LIBRARIES
	"${ONNX_ROOT}/lib/onnxruntime.dll"
	"${ONNX_ROOT}/lib/onnxruntime_providers_shared.dll"
CACHE INTERNAL "")

# set(OPENCV_ROOT "E:/opencv-4.5.4/Release")
set(OPENCV_ROOT "E:/opencv-4.5.4/Debug")
include_directories(${OPENCV_ROOT}/include)

set(OPENCV_LIBRARIES 
	"${OPENCV_ROOT}/lib/x64/opencv_world454d.lib" 
CACHE INTERNAL "")
set(OPENCV_RUNTIME_LIBRARIES
	"${OPENCV_ROOT}/bin/opencv_world454d.dll"
CACHE INTERNAL "")


if(WIN32) 
	add_executable(${PROJECT_NAME} support.cpp)
	target_link_libraries(${PROJECT_NAME} ${ONNX_LIBRARIES} ${OPENCV_LIBRARIES})   # release   # debug opencv_world454d
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD   # 绑定到主目标
    COMMAND ${CMAKE_COMMAND} -E copy
        ${ONNX_RUNTIME_LIBRARIES} ${OPENCV_RUNTIME_LIBRARIES}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>   # 自动获取可执行文件目录
    COMMENT "Copying shared library to output directory"
)
endif()

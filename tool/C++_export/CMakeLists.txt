cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(KNNQuery)

set(CMAKE_VERBOSE_MAKEFILE ON)
# >>> build type 
set(CMAKE_BUILD_TYPE "Release")				# 指定生成的版本
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# <<<

if(WIN32)
    # windows10
    # Torch
    set(TORCH_ROOT "D:/Anaconda3/envs/Test374/Lib/site-packages/torch")  # 我这里采用虚拟环境安装的pytorch
    include_directories(${TORCH_ROOT}/include)
    link_directories(${TORCH_ROOT}/lib/)
    set(TORCH_LIBRARIES "${TORCH_ROOT}/lib/*.lib")   # 可以选择需要的库


    # Define our library target
    add_library(${PROJECT_NAME} SHARED warpPerspective.cpp)

    # Enable C++14
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

    # Link against Torch
    target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

     
elseif(UNIX)
    # Ubuntu18.04
    # Torch
    set(TORCH_ROOT "/home/heygears/anaconda3/envs/pt/lib/python3.7/site-packages/torch")   
    include_directories(${TORCH_ROOT}/include)
    link_directories(${TORCH_ROOT}/lib/)
    file(GLOB TORCH_LIBRARIES "${TORCH_ROOT}/lib/*.so")

    # Define our library target
    add_library(${PROJECT_NAME} SHARED knnQuery.cpp)

    # Enable C++14
    add_definitions(-D _GLIBCXX_USE_CXX11_ABI=0)
    target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)
    # libtorch库文件
    target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}"
        # CPU
        # c10 
        # torch_cpu
        # torch
        # GPU
        # c10_cuda 
        # torch_cuda
    )

    
endif()

# windows需要把相关依赖库copy到编译目录下
if (MSVC)
  file(GLOB OPENCV_DLLS "${OPENCV_ROOT}/bin/x64/opencv_world420.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OPENCV_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)

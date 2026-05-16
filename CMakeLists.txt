cmake_minimum_required(VERSION 3.10)
project(MyProject)

# ユーザーが指定しない場合、強制的にReleaseモードにする
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

file(GLOB SOURCES src/*.cpp)

add_executable(Matrix ${SOURCES})

target_include_directories(Matrix 
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

# OpenMPの検索とリンク
find_package(OpenMP REQUIRED)
target_link_libraries(Matrix PRIVATE OpenMP::OpenMP_CXX)

if (MSVC)
    target_compile_options(Matrix PRIVATE
        /O2          # 最適化
        /arch:AVX2   # AVX2命令の有効化
        /fp:fast     # 浮動小数点演算の高速化
        /wd4244   
    )
endif()
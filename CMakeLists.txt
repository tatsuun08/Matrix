cmake_minimum_required(VERSION 3.10)
project(MyProject)

file(GLOB SOURCES src/*.cpp)


add_executable(Matrix ${SOURCES})

target_include_directories(Matrix 
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

# MSVCでインストールしたClang(clang-cl)かどうかの判定
if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT MSVC)

target_compile_options(Matrix PRIVATE -fopenmp)
    get_filename_component(COMPILE_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(LLVM_ROOT_DIR "${COMPILER_DIR}/..")
    target_link_directories(Matrix PRIVATE "${LLVM_ROOT_DIR}/lib")
    # リンクオプション：MSVC付属の libomp.lib をリンクする
    target_link_libraries(Matrix PRIVATE libomp.lib)
else()
    # 通常のコンパイラ（GCCや本来のMSVC）向け
    find_package(OpenMP REQUIRED)
    target_link_libraries(Matrix PRIVATE OpenMP::OpenMP_CXX)
endif()
cmake_minimum_required(VERSION 3.3)
project(SUSAN LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)

include_directories("${CMAKE_SOURCE_DIR}/../dependencies/eigen")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=20011")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=20014")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --display_error_number")
#string(APPEND CMAKE_CUDA_FLAGS " -lineinfo")

#add_executable(averager src/averager.cu src/averager.h)
#target_link_libraries(averager cufft)

#add_executable(aligner src/aligner.cu src/aligner.h)
#target_link_libraries(aligner cufft)

add_executable(susan_estimate_ctf src/estimate_ctf.cu src/estimate_ctf.h)
target_link_libraries(susan_estimate_ctf cufft)
set_property(TARGET susan_estimate_ctf PROPERTY CUDA_ARCHITECTURES 52 60 70)

cmake_minimum_required(VERSION 3.3)
project(SUSAN LANGUAGES CXX CUDA)

#set(CMAKE_BUILD_TYPE Debug)

include_directories("${CMAKE_SOURCE_DIR}/../dependencies/eigen")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=20011")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=20014")
#string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=2976")
#string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=2979")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --display_error_number")
#string(APPEND CMAKE_CUDA_FLAGS " -lineinfo")

add_executable(susan_estimate_ctf src/estimate_ctf.cu)
target_link_libraries(susan_estimate_ctf cufft)
#set_property(TARGET susan_estimate_ctf PROPERTY CUDA_ARCHITECTURES 52 60 70)
set_property(TARGET susan_estimate_ctf PROPERTY CUDA_ARCHITECTURES 52)

add_executable(susan_reconstruct src/reconstruct.cu)
target_link_libraries(susan_reconstruct cufft)
#set_property(TARGET susan_reconstruct PROPERTY CUDA_ARCHITECTURES 52 60 70)
set_property(TARGET susan_reconstruct PROPERTY CUDA_ARCHITECTURES 52)

add_executable(susan_subtomos_rec src/rec_subtomos.cu)
target_link_libraries(susan_subtomos_rec cufft)
#set_property(TARGET susan_subtomos_rec PROPERTY CUDA_ARCHITECTURES 52 60 70)
set_property(TARGET susan_subtomos_rec PROPERTY CUDA_ARCHITECTURES 52)

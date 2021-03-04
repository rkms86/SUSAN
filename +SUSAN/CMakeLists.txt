cmake_minimum_required(VERSION 3.3)
project(SUSAN LANGUAGES CXX CUDA)
find_package(MPI 3)

set(CMAKE_BUILD_TYPE Debug)

include_directories("${CMAKE_SOURCE_DIR}/../dependencies/eigen")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_ARCHS 52)
#set(CUDA_ARCHS 52 60 70)


string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
if( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 10.0 )
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=20011")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=20014")
elseif( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0 )
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=2976")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=2979")
endif()
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --display_error_number")
#string(APPEND CMAKE_CUDA_FLAGS " -lineinfo")

add_executable(susan_estimate_ctf src/estimate_ctf.cu)
target_link_libraries(susan_estimate_ctf cufft)
set_property(TARGET susan_estimate_ctf PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})

add_executable(susan_reconstruct src/reconstruct.cu)
target_link_libraries(susan_reconstruct cufft)
set_property(TARGET susan_reconstruct PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})

add_executable(susan_subtomos_rec src/rec_subtomos.cu)
target_link_libraries(susan_subtomos_rec cufft)
set_property(TARGET susan_subtomos_rec PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})

add_executable(susan_aligner src/aligner.cu)
target_link_libraries(susan_aligner cufft)
set_property(TARGET susan_aligner PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})

add_executable(susan_refs_aligner src/refs_aligner.cu)
target_link_libraries(susan_refs_aligner cufft)
set_property(TARGET susan_refs_aligner PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})

add_executable(susan_refs_fsc src/refs_fsc.cu)
target_link_libraries(susan_refs_fsc cufft)
set_property(TARGET susan_refs_fsc PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})

if( MPI_FOUND )
add_executable(susan_reconstruct_mpi src/reconstruct_mpi.cu)
target_compile_definitions(susan_reconstruct_mpi PUBLIC OPENMPI)
target_link_libraries(susan_reconstruct_mpi cufft ${MPI_CXX_LIBRARIES})
target_include_directories(susan_reconstruct_mpi PUBLIC ${MPI_CXX_INCLUDE_DIRS})
set_property(TARGET susan_reconstruct_mpi PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})

add_executable(susan_aligner_mpi src/aligner_mpi.cu)
target_compile_definitions(susan_aligner_mpi PUBLIC OPENMPI)
target_link_libraries(susan_aligner_mpi cufft ${MPI_CXX_LIBRARIES})
target_include_directories(susan_aligner_mpi PUBLIC ${MPI_CXX_INCLUDE_DIRS})
set_property(TARGET susan_aligner_mpi PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})

endif( MPI_FOUND )

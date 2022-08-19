cmake_minimum_required(VERSION 3.3)

# Define Project
project(SUSAN LANGUAGES CXX CUDA)
set(CMAKE_BUILD_TYPE Release)

# Configure EIGEN
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/extern/eigen_lib)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Configure CUDA
set(CUDA_ARCHS 52 60 61 70 75)
string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --display_error_number")

# Define Executables
set(target_list estimate_ctf reconstruct rec_subtomos aligner refs_aligner ctf_refiner)
foreach(target IN LISTS target_list)
	set(bin susan_${target})
	add_executable(${bin} src/${target}.cu)
	target_link_libraries(${bin} cufft Eigen3::Eigen)
	set_property(TARGET ${bin} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})
endforeach()

# Conditionally build if MPI is found
find_package(MPI)
if( MPI_FOUND )
	set(target_list reconstruct_mpi aligner_mpi)
	foreach(target IN LISTS target_list)
		set(bin susan_${target})
		add_executable(${bin} src/${target}.cu)
		target_compile_definitions(${bin} PUBLIC OPENMPI)
		target_link_libraries(${bin} cufft ${MPI_CXX_LIBRARIES} Eigen3::Eigen)
		target_include_directories(${bin} PUBLIC ${MPI_CXX_INCLUDE_DIRS})
		set_property(TARGET ${bin} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHS})
	endforeach()
endif( MPI_FOUND )

# Conditionally build if MATLAB is found
find_package(Matlab)
if( Matlab_FOUND )
	include_directories(${CMAKE_SOURCE_DIR}/src)
	file(GLOB_RECURSE mex_cpp LIST_DIRECTORIES false "${CMAKE_SOURCE_DIR}/+SUSAN/*.cpp")
	foreach(src IN LISTS mex_cpp)
		get_filename_component(folder ${src} DIRECTORY)
		get_filename_component(target ${src} NAME_WE)
		set(binary "${target}.${Matlab_MEX_EXTENSION}")
		matlab_add_mex(NAME ${target} SRC ${src} LINK_TO Eigen3::Eigen R2018a)
		file(COPY "${CMAKE_CURRENT_BINARY_DIR}/${binary}" DESTINATION "${folder}/")
	endforeach()
endif( Matlab_FOUND )

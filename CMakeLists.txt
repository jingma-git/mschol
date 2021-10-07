project(mschol)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE RELEASE)
list(APPEND CMAKE_CXX_FLAGS "-fopenmp -std=c++14 -fpermissive")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")

include_directories(${PROJECT_SOURCE_DIR})

# BOOST
find_package(Boost REQUIRED)
if(Boost_FOUND)
  message("-- Boost @ ${Boost_INCLUDE_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)

# BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
set(BLAS_LAPACK_LIBS ${LAPACK_LIBARIES} ${BLAS_LIBRARIES})

# SuiteSparse
find_package(SuiteSparse REQUIRED)
if(SuiteSparse_FOUND)
  message("-- SuiteSparse @ ${SuiteSparse_INCLUDE_DIRS}")
  message("-- SuiteSparse libs @ ${SuiteSparse_LIBRARIES}")
  include_directories(${SuiteSparse_INCLUDE_DIRS})
endif(SuiteSparse_FOUND)

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  message("-- OpenCV @ ${OpenCV_INCLUDE_DIRS}")
  message("-- OpenCV @ ${OpenCV_LIBRARIES}")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)


# HEADER-ONLY LIBS
include_directories(external/amgcl/)
include_directories(external/eigen/)
include_directories(external/CGAL/include)
include_directories(external/spdlog/include)
include_directories(external/egl/include)

add_subdirectory(src)
add_subdirectory(examples)

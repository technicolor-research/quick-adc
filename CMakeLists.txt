cmake_minimum_required(VERSION 3.2)
project(qadc)

include(LibsPath.cmake)

# GCC Flags
set(GCC_FLAGS "-g -O3 -m64 -march=native -ffast-math -fopenmp -Wall -Wno-ignored-attributes -Werror")

execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
if (GCC_VERSION VERSION_GREATER 4.9 OR GCC_VERSION VERSION_EQUAL 4.9)
    set(GCC_FLAGS "-fdiagnostics-color=auto ${GCC_FLAGS}")
endif()

link_directories("${LIBS_DIR}")
include_directories("${LIBS_INCLUDE_DIR}")

# C++ Flags
set(CXX_STD "-std=c++14")
set(CMAKE_CXX_FLAGS "${CXX_STD} ${GCC_FLAGS}")

# Object files
add_library(vector_io OBJECT vector_io.cpp)
add_library(quantizers OBJECT quantizers.cpp)
add_library(databases OBJECT databases.cpp)
add_library(distances OBJECT distances.cpp)
add_library(neighbors OBJECT neighbors.cpp)

add_executable(flatdb_create
    flatdb_create.cpp
    $<TARGET_OBJECTS:databases>
    $<TARGET_OBJECTS:quantizers>
    $<TARGET_OBJECTS:vector_io>
    $<TARGET_OBJECTS:distances>
    $<TARGET_OBJECTS:neighbors>)
target_link_libraries(flatdb_create 
    ${LIBS_DIR}/lib${SEQUENTIAL_BLAS_LIB}.so
    ${OPENCV_CORE_LIB})

add_executable(indexdb_create1
    indexdb_create1.cpp
    $<TARGET_OBJECTS:databases>
    $<TARGET_OBJECTS:quantizers>
    $<TARGET_OBJECTS:vector_io>
    $<TARGET_OBJECTS:distances>
    $<TARGET_OBJECTS:neighbors>)
target_link_libraries(indexdb_create1 
    ${LIBS_DIR}/lib${SEQUENTIAL_BLAS_LIB}.so
    ${OPENCV_CORE_LIB})

add_executable(indexdb_create2
    indexdb_create2.cpp
    $<TARGET_OBJECTS:databases>
    $<TARGET_OBJECTS:quantizers>
    $<TARGET_OBJECTS:vector_io>
    $<TARGET_OBJECTS:distances>
    $<TARGET_OBJECTS:neighbors>)
target_link_libraries(indexdb_create2 
    ${LIBS_DIR}/lib${SEQUENTIAL_BLAS_LIB}.so
    ${OPENCV_CORE_LIB})

add_executable(db_add
    db_add.cpp
    $<TARGET_OBJECTS:databases>
    $<TARGET_OBJECTS:quantizers>
    $<TARGET_OBJECTS:vector_io>
    $<TARGET_OBJECTS:distances>
    $<TARGET_OBJECTS:neighbors>)
target_link_libraries(db_add 
    ${LIBS_DIR}/lib${SEQUENTIAL_BLAS_LIB}.so
    ${OPENCV_CORE_LIB})

add_executable(db_query
    db_query.cpp
    $<TARGET_OBJECTS:databases>
    $<TARGET_OBJECTS:quantizers>
    $<TARGET_OBJECTS:vector_io>
    $<TARGET_OBJECTS:distances>
    $<TARGET_OBJECTS:neighbors>)
target_link_libraries(db_query 
    ${LIBS_DIR}/lib${SEQUENTIAL_BLAS_LIB}.so
    ${OPENCV_CORE_LIB})

add_executable(db_query_4
    db_query_4.cpp
    $<TARGET_OBJECTS:databases>
    $<TARGET_OBJECTS:quantizers>
    $<TARGET_OBJECTS:vector_io>
    $<TARGET_OBJECTS:distances>
    $<TARGET_OBJECTS:neighbors>)
target_link_libraries(db_query_4 
    ${LIBS_DIR}/lib${SEQUENTIAL_BLAS_LIB}.so
    ${OPENCV_CORE_LIB})

add_executable(split_vecs
    split_vecs.cpp)
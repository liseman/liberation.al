# An example C++ worker on the Continuum C ABI (continuum.h) and its C++20 wrapper (continuum.hpp).
#
#   cmake -S . -B build -DCONTINUUM_LIB_DIR=<dir with libcontinuum> && cmake --build build
#
# `mise run continuum-ffi-example` builds the library with cargo, then this, and runs it against a real broker.
cmake_minimum_required(VERSION 3.20)
project(continuum_cpp_worker LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

set(CONTINUUM_LIB_DIR "" CACHE PATH "Where cargo put libcontinuum (continuum/target/debug or release)")
set(CONTINUUM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include" CACHE PATH "continuum.h and continuum.hpp")
if(NOT CONTINUUM_LIB_DIR)
  message(FATAL_ERROR "set CONTINUUM_LIB_DIR to the directory holding libcontinuum")
endif()

add_library(continuum SHARED IMPORTED)
set_target_properties(continuum PROPERTIES
  IMPORTED_LOCATION "${CONTINUUM_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}continuum${CMAKE_SHARED_LIBRARY_SUFFIX}"
  INTERFACE_INCLUDE_DIRECTORIES "${CONTINUUM_INCLUDE_DIR}")

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_executable(cpp_worker main.cpp)
target_link_libraries(cpp_worker PRIVATE continuum)
set_target_properties(cpp_worker PROPERTIES BUILD_RPATH "${CONTINUUM_LIB_DIR}")

# The C header has to stand on its own in plain C too.
add_executable(c_check c_check.c)
target_link_libraries(c_check PRIVATE continuum)
set_target_properties(c_check PROPERTIES BUILD_RPATH "${CONTINUUM_LIB_DIR}")

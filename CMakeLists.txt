cmake_minimum_required(VERSION 3.20)
project(lemmakernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Generated files are committed; this target keeps them honest at configure time.
execute_process(
  COMMAND ${CMAKE_SOURCE_DIR}/tools/manifest.py check
  RESULT_VARIABLE manifest_result OUTPUT_VARIABLE manifest_output ERROR_VARIABLE manifest_output)
if(NOT manifest_result EQUAL 0)
  message(FATAL_ERROR "manifest check failed:\n${manifest_output}")
endif()

add_library(lemmakernel SHARED
  runtime/src/lk.cpp
  runtime/src/object.cpp
  runtime/src/family.cpp
  runtime/generated/manifest.cpp)
if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS compile_pool=4 link_pool=1)
  set_property(TARGET lemmakernel PROPERTY JOB_POOL_COMPILE compile_pool)
  set_property(TARGET lemmakernel PROPERTY JOB_POOL_LINK link_pool)
endif()
target_include_directories(lemmakernel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(lemmakernel PRIVATE -O3 -Wall -Wextra -fno-plt)
find_package(Threads REQUIRED)
target_link_libraries(lemmakernel PRIVATE Threads::Threads)
set_target_properties(lemmakernel PROPERTIES VERSION 0.1.0 SOVERSION 0)

include(${CMAKE_SOURCE_DIR}/runtime/generated/modules.cmake)

include(GNUInstallDirs)
install(TARGETS lemmakernel LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY runtime/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

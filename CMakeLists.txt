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
  runtime/src/group_catalogue_data.cpp
  runtime/generated/manifest.cpp)
if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS compile_pool=4 link_pool=1)
  set_property(TARGET lemmakernel PROPERTY JOB_POOL_COMPILE compile_pool)
  set_property(TARGET lemmakernel PROPERTY JOB_POOL_LINK link_pool)
endif()
target_include_directories(lemmakernel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime/include>
  $<INSTALL_INTERFACE:include>)
# Backends lean on popcount, ctz and wide ORs; without an ISA baseline gcc emits library calls
# for them. `native` targets the build machine; set a level such as x86-64-v3 for a binary that
# must run elsewhere, or an empty string for the compiler default.
set(LEMMAKERNEL_MARCH "native" CACHE STRING "-march value for the library")
target_compile_options(lemmakernel PRIVATE -O3 -Wall -Wextra -fno-plt)
if(LEMMAKERNEL_MARCH)
  target_compile_options(lemmakernel PRIVATE -march=${LEMMAKERNEL_MARCH})
endif()
find_package(Threads REQUIRED)
target_link_libraries(lemmakernel PRIVATE Threads::Threads)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(NAUTY_TLS QUIET IMPORTED_TARGET libnautyT)
  if(NOT TARGET PkgConfig::NAUTY_TLS)
    pkg_check_modules(NAUTY QUIET IMPORTED_TARGET nauty)
  endif()
endif()
if(TARGET PkgConfig::NAUTY_TLS)
  target_link_libraries(lemmakernel PRIVATE PkgConfig::NAUTY_TLS)
  target_compile_definitions(lemmakernel PRIVATE LEMMAKERNEL_HAVE_NAUTY=1)
elseif(TARGET PkgConfig::NAUTY)
  target_link_libraries(lemmakernel PRIVATE PkgConfig::NAUTY)
  target_compile_definitions(lemmakernel PRIVATE
    LEMMAKERNEL_HAVE_NAUTY=1 LEMMAKERNEL_SERIALIZE_NAUTY=1)
endif()
set_target_properties(lemmakernel PROPERTIES
  VERSION 0.1.0 SOVERSION 0 BUILD_RPATH "$ORIGIN")

include(${CMAKE_SOURCE_DIR}/runtime/generated/modules.cmake)

# GPU device code is a separate library built by hipcc (CMake's HIP language cannot locate the
# ROCm device libraries on NixOS) and loaded by its backend with dlopen from liblemmakernel's
# directory, so the main library carries no ROCm dependency and the backend is simply
# unavailable where the device library or a device is missing. The architecture list defaults to
# the machine's devices (rocm_agent_enumerator), else gfx1151.
option(LEMMAKERNEL_HIP "Build the HIP device library when hipcc is present" ON)
find_program(LEMMAKERNEL_HIPCC hipcc)
if(LEMMAKERNEL_HIP AND LEMMAKERNEL_HIPCC)
  if(NOT LEMMAKERNEL_HIP_ARCHS)
    find_program(LEMMAKERNEL_ROCM_AGENTS rocm_agent_enumerator)
    if(LEMMAKERNEL_ROCM_AGENTS)
      execute_process(COMMAND ${LEMMAKERNEL_ROCM_AGENTS} OUTPUT_VARIABLE agents OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REPLACE "\n" ";" agents "${agents}")
      list(REMOVE_ITEM agents gfx000)
      list(REMOVE_DUPLICATES agents)
      string(REPLACE ";" "," LEMMAKERNEL_HIP_ARCHS "${agents}")
    endif()
    if(NOT LEMMAKERNEL_HIP_ARCHS)
      set(LEMMAKERNEL_HIP_ARCHS gfx1151)
    endif()
  endif()
  set(LEMMAKERNEL_HIP_ARCHS "${LEMMAKERNEL_HIP_ARCHS}" CACHE STRING "comma-separated offload architectures for hipcc")
  string(REPLACE "," ";" hip_arch_list "${LEMMAKERNEL_HIP_ARCHS}")
  set(hip_arch_flags)
  foreach(arch ${hip_arch_list})
    list(APPEND hip_arch_flags --offload-arch=${arch})
  endforeach()
  set(hip_sources ${CMAKE_SOURCE_DIR}/modules/heat_dirichlet/backends/hip/phase_kernel.hip)
  set(hip_library ${CMAKE_BINARY_DIR}/liblemmakernel_hip.so)
  add_custom_command(OUTPUT ${hip_library}
    COMMAND ${LEMMAKERNEL_HIPCC} -O3 -std=c++20 -fPIC -shared ${hip_arch_flags} ${hip_sources} -o ${hip_library}
    DEPENDS ${hip_sources} ${CMAKE_SOURCE_DIR}/modules/heat_dirichlet/backends/hip/phase_gpu_abi.hpp
    COMMENT "hipcc liblemmakernel_hip.so (${LEMMAKERNEL_HIP_ARCHS})" VERBATIM)
  add_custom_target(lemmakernel_hip ALL DEPENDS ${hip_library})
  install(FILES ${hip_library} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  message(STATUS "HIP device library: ${LEMMAKERNEL_HIPCC} for ${LEMMAKERNEL_HIP_ARCHS}")
else()
  message(STATUS "HIP device library: not built (hipcc not found or LEMMAKERNEL_HIP off)")
endif()

include(GNUInstallDirs)
install(TARGETS lemmakernel LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY runtime/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

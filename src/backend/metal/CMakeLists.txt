# Copyright (c) 2017, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

include(InternalUtils)
include(FileToString)

if(NOT APPLE)
  message(FATAL_ERROR "The ArrayFire Metal backend is only supported on Apple platforms.")
endif()

find_path(METALCPP_INCLUDE_DIR
  NAMES Metal.hpp
  HINTS
    "$ENV{METALCPP_ROOT}"
  PATH_SUFFIXES
    include
    metal-cpp
    metal-cpp/Metal
  DOC "Directory containing Apple's metal-cpp Metal.hpp header")

if(NOT METALCPP_INCLUDE_DIR)
  message(FATAL_ERROR
    "AF_BUILD_METAL requires Apple's metal-cpp headers. "
    "Set METALCPP_INCLUDE_DIR to the directory containing Metal.hpp.")
endif()

find_library(AF_METAL_FRAMEWORK Metal REQUIRED)
find_library(AF_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(AF_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(AF_MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)

enable_language(OBJCXX)

generate_product_version(af_metal_ver_res_file
  FILE_NAME "afmetal"
  FILE_DESCRIPTION "Metal Backend Dynamic-link library"
)

add_library(afmetal "")
add_library(ArrayFire::afmetal ALIAS afmetal)

set(metal_kernel_src
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/copy.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/anisotropic_diffusion.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/Array.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/approx.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/assign.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/bilateral.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/canny.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/convolve.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/exampleFunction.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/range.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/random_engine.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/reduce.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/iota.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/ireduce.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/identity.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/iir.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/index.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/tile.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/topk.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/shift.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/sift.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/reorder.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/regions.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/rotate.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/resize.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/scan.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/scan_by_key.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/select.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/join.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/lookup.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/lu.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/solve.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/match_template.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/moments.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/morph.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/meanshift.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/mean.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/medfilt.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/nearest_neighbour.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/orb.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/qr.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/pad_borders.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/diagonal.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/diff.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/dot.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/cholesky.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/flood_fill.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/fast.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/fftconvolve.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/triangle.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/gradient.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/hsv_rgb.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/histogram.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/harris.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/sobel.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/sort_helper.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/sort.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/sort_by_key.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/sparse.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/sparse_blas.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/sparse_arith.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/susan.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/svd.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/transpose.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/transform.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/unwrap.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/wrap.metal)

set(metal_jit_kernel_src
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/metal/jit.metal)

file_to_string(
  SOURCES ${metal_kernel_src} ${metal_jit_kernel_src}
  VARNAME metal_kernel_files
  EXTENSION "hpp"
  OUTPUT_DIR metal_kernel_headers
  TARGETS metal_kernel_targets
  NAMESPACE "arrayfire metal kernel"
  NULLTERM
  NO_INCLUDE)

add_dependencies(afmetal ${metal_kernel_targets})

option(AF_METAL_PRECOMPILE_KERNELS
  "Compile built-in Metal kernels to an embedded metallib at build time"
  ON)
set(AF_METAL_KERNEL_LIBRARY "" CACHE FILEPATH
  "Existing metallib to embed instead of compiling ArrayFire's Metal kernels")

set(af_metal_embedded_library "")
set(af_metal_normalized_library
  "${CMAKE_CURRENT_BINARY_DIR}/arrayfire_metal_kernels.metallib")

if(AF_METAL_KERNEL_LIBRARY)
  if(NOT EXISTS "${AF_METAL_KERNEL_LIBRARY}")
    message(FATAL_ERROR
      "AF_METAL_KERNEL_LIBRARY does not exist: ${AF_METAL_KERNEL_LIBRARY}")
  endif()
  add_custom_command(
    OUTPUT "${af_metal_normalized_library}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${AF_METAL_KERNEL_LIBRARY}"
      "${af_metal_normalized_library}"
    DEPENDS "${AF_METAL_KERNEL_LIBRARY}"
    VERBATIM)
  set(af_metal_embedded_library "${af_metal_normalized_library}")
elseif(AF_METAL_PRECOMPILE_KERNELS)
  set(af_metal_find_result 1)
  set(af_metallib_find_result 1)
  set(af_metal_compiler_works 1)
  find_program(AF_XCRUN_EXECUTABLE xcrun)
  if(AF_XCRUN_EXECUTABLE)
    execute_process(
      COMMAND "${AF_XCRUN_EXECUTABLE}" -sdk macosx --find metal
      RESULT_VARIABLE af_metal_find_result
      OUTPUT_VARIABLE af_metal_compiler
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    execute_process(
      COMMAND "${AF_XCRUN_EXECUTABLE}" -sdk macosx --find metallib
      RESULT_VARIABLE af_metallib_find_result
      OUTPUT_VARIABLE af_metallib_linker
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()

  if(af_metal_find_result EQUAL 0 AND
     af_metallib_find_result EQUAL 0)
    execute_process(
      COMMAND "${af_metal_compiler}" --version
      RESULT_VARIABLE af_metal_compiler_works
      OUTPUT_QUIET
      ERROR_QUIET)
  endif()

  if(af_metal_compiler_works EQUAL 0)
    set(af_metal_combined_source
      "${CMAKE_CURRENT_BINARY_DIR}/arrayfire_metal_kernels.metal")
    set(af_metal_air
      "${CMAKE_CURRENT_BINARY_DIR}/arrayfire_metal_kernels.air")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      ${metal_kernel_src})
    file(WRITE "${af_metal_combined_source}" "")
    foreach(af_metal_source IN LISTS metal_kernel_src)
      file(READ "${af_metal_source}" af_metal_source_contents)
      file(APPEND "${af_metal_combined_source}"
        "${af_metal_source_contents}\n")
    endforeach()

    set(af_metal_deployment_flag "")
    if(CMAKE_OSX_DEPLOYMENT_TARGET)
      set(af_metal_deployment_flag
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
    add_custom_command(
      OUTPUT "${af_metal_normalized_library}"
      COMMAND "${af_metal_compiler}"
        -c
        ${af_metal_deployment_flag}
        "${af_metal_combined_source}"
        -o "${af_metal_air}"
      COMMAND "${af_metallib_linker}"
        "${af_metal_air}"
        -o "${af_metal_normalized_library}"
      DEPENDS "${af_metal_combined_source}" ${metal_kernel_src}
      COMMENT "Compiling ArrayFire Metal kernels to a metallib"
      VERBATIM)
    set(af_metal_embedded_library "${af_metal_normalized_library}")
  else()
    message(STATUS
      "Metal compiler tools unavailable; built-in kernels will use runtime source compilation")
  endif()
endif()

if(af_metal_embedded_library)
  file_to_string(
    SOURCES "${af_metal_embedded_library}"
    VARNAME metal_kernel_library
    EXTENSION "hpp"
    OUTPUT_DIR metal_kernel_library
    TARGETS metal_kernel_library_target
    NAMESPACE "arrayfire metal kernel"
    BINARY TRUE
    NO_INCLUDE)
  add_dependencies(afmetal ${metal_kernel_library_target})
  target_compile_definitions(afmetal PRIVATE AF_METAL_PRECOMPILED_KERNELS)
endif()

# Metal backend needs to use MKL LP64 interface
set(MKL_INTERFACE_INTEGER_SIZE 4)
set(MKL_INTERFACE "lp64")

# Metal backend source files
target_sources(afmetal
  PRIVATE
    $<$<PLATFORM_ID:Windows>:${af_metal_ver_res_file}>
    Array.cpp
    Array.hpp
    anisotropic_diffusion.cpp
    anisotropic_diffusion.hpp
    approx.cpp
    approx.hpp
    arith.hpp
    assign.cpp
    assign.hpp
    backend.hpp
    binary.hpp
    bilateral.cpp
    bilateral.hpp
    blas.cpp
    blas.hpp
    canny.cpp
    canny.hpp
    cast.hpp
    cholesky.cpp
    cholesky.hpp
    complex.hpp
    convolve.cpp
    convolve.hpp
    copy.cpp
    copy.hpp
    compile_module.cpp
    device_manager.cpp
    device_manager.hpp
    diagonal.cpp
    diagonal.hpp
    diff.cpp
    diff.hpp
    err_metal.hpp
    Event.cpp
    Event.hpp
    exampleFunction.cpp
    exampleFunction.hpp
    fast.cpp
    fast.hpp
    fft.cpp
    fft.hpp
    metal_fft.mm
    metal_fft.hpp
    metal_inverse.mm
    metal_inverse.hpp
    fftconvolve.cpp
    fftconvolve.hpp
    flood_fill.hpp
    flood_fill.cpp
    gradient.cpp
    gradient.hpp
    harris.cpp
    harris.hpp
    hist_graphics.cpp
    hist_graphics.hpp
    histogram.cpp
    histogram.hpp
    homography.cpp
    homography.hpp
    hsv_rgb.cpp
    hsv_rgb.hpp
    identity.cpp
    identity.hpp
    iir.cpp
    iir.hpp
    image.cpp
    image.hpp
    index.cpp
    index.hpp
    inverse.cpp
    inverse.hpp
    iota.cpp
    iota.hpp
    ireduce.cpp
    ireduce.hpp
    jit.cpp
    jit.hpp
    join.cpp
    join.hpp
    lapack_helper.hpp
    logic.hpp
    lookup.cpp
    lookup.hpp
    lu.cpp
    lu.hpp
    match_template.cpp
    match_template.hpp
    math.cpp
    math.hpp
    mean.cpp
    mean.hpp
    meanshift.cpp
    meanshift.hpp
    medfilt.cpp
    medfilt.hpp
    kernel/dispatch_algorithm.cpp
    kernel/dispatch_common.cpp
    kernel/dispatch_data.cpp
    kernel/dispatch_image.cpp
    kernel/metal_runtime.cpp
    Kernel.hpp
    memory.cpp
    memory.hpp
    Module.hpp
    moments.cpp
    moments.hpp
    morph.cpp
    morph.hpp
    nearest_neighbour.cpp
    nearest_neighbour.hpp
    orb.cpp
    orb.hpp
    platform.cpp
    platform.hpp
    Param.hpp
    plot.cpp
    plot.hpp
    print.hpp
    qr.cpp
    qr.hpp
    queue.hpp
    random_engine.cpp
    random_engine.hpp
    range.cpp
    range.hpp
    reduce.cpp
    reduce.hpp
    regions.cpp
    regions.hpp
    reorder.cpp
    reorder.hpp
    resize.cpp
    resize.hpp
    reshape.cpp
    rotate.cpp
    rotate.hpp
    scan.cpp
    scan.hpp
    scan_by_key.cpp
    scan_by_key.hpp
    select.cpp
    select.hpp
    set.cpp
    set.hpp
    shift.cpp
    shift.hpp
    sift.cpp
    sift.hpp
    sobel.cpp
    sobel.hpp
    solve.cpp
    solve.hpp
    sort.cpp
    sort.hpp
    sort_by_key.cpp
    sort_by_key.hpp
    sort_index.cpp
    sort_index.hpp
    sparse.cpp
    sparse.hpp
    sparse_arith.cpp
    sparse_arith.hpp
    sparse_blas.cpp
    sparse_blas.hpp
    surface.cpp
    surface.hpp
    susan.cpp
    susan.hpp
    svd.cpp
    svd.hpp
    tile.cpp
    tile.hpp
    topk.cpp
    topk.hpp
    traits.hpp
    transform.cpp
    transform.hpp
    transpose.cpp
    transpose.hpp
    triangle.cpp
    triangle.hpp
    types.hpp
    unary.hpp
    unwrap.cpp
    unwrap.hpp
    utility.hpp
    vector_field.cpp
    vector_field.hpp
    where.cpp
    where.hpp
    wrap.cpp
    wrap.hpp
  )

set_source_files_properties(metal_fft.mm metal_inverse.mm PROPERTIES
  COMPILE_FLAGS "-fobjc-arc")

target_sources(afmetal
  PRIVATE
    jit/BufferNode.hpp
    jit/ShiftNode.hpp
    jit/kernel_generators.hpp
    kernel/KParam.hpp
    kernel/cholesky.hpp
    kernel/dispatch_common.hpp
    kernel/metal_runtime.hpp
  )

# Metal backend kernel files
target_sources(afmetal
  PRIVATE
    kernel/Array.hpp
    kernel/anisotropic_diffusion.hpp
    kernel/approx.hpp
    kernel/assign.hpp
    kernel/bilateral.hpp
    kernel/canny.hpp
    kernel/convolve.hpp
    kernel/copy.hpp
    kernel/cholesky.hpp
    kernel/diagonal.hpp
    kernel/diff.hpp
    kernel/dot.hpp
    kernel/exampleFunction.hpp
    kernel/fast.hpp
    kernel/fftconvolve.hpp
    kernel/flood_fill.hpp
    kernel/gradient.hpp
    kernel/harris.hpp
    kernel/histogram.hpp
    kernel/hsv_rgb.hpp
    kernel/identity.hpp
    kernel/iir.hpp
    kernel/index.hpp
    kernel/iota.hpp
    kernel/ireduce.hpp
    kernel/join.hpp
    kernel/lookup.hpp
    kernel/lu.hpp
    kernel/match_template.hpp
    kernel/meanshift.hpp
    kernel/medfilt.hpp
    kernel/moments.hpp
    kernel/morph.hpp
    kernel/nearest_neighbour.hpp
    kernel/orb.hpp
    kernel/pad_array_borders.hpp
    kernel/random_engine.hpp
    kernel/random_engine_mersenne.hpp
    kernel/random_engine_philox.hpp
    kernel/random_engine_threefry.hpp
    kernel/range.hpp
    kernel/reduce.hpp
    kernel/regions.hpp
    kernel/reorder.hpp
    kernel/resize.hpp
    kernel/rotate.hpp
    kernel/scan.hpp
    kernel/scan_by_key.hpp
    kernel/select.hpp
    kernel/shift.hpp
    kernel/sift.hpp
    kernel/sobel.hpp
    kernel/sort.hpp
    kernel/sort_by_key.hpp
    kernel/sort_helper.hpp
    kernel/sparse.hpp
    kernel/sparse_blas.hpp
    kernel/sparse_arith.hpp
    kernel/susan.hpp
    kernel/tile.hpp
    kernel/topk.hpp
    kernel/transform.hpp
    kernel/transpose.hpp
    kernel/triangle.hpp
    kernel/unwrap.hpp
    kernel/wrap.hpp
  )

target_include_directories(afmetal
  PUBLIC
    $<BUILD_INTERFACE:${ArrayFire_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ArrayFire_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${AF_INSTALL_INC_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${METALCPP_INCLUDE_DIR})

target_include_directories(afmetal
  SYSTEM PRIVATE
    ${CBLAS_INCLUDE_DIR})

target_compile_definitions(afmetal
  PRIVATE
    AF_METAL
    METALCPP_SYMBOL_VISIBILITY_HIDDEN
  )

target_compile_options(afmetal
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fblocks>)

target_link_libraries(afmetal
  PRIVATE
    c_api_interface
    cpp_api_interface
    afcommon_interface
    Threads::Threads
    ${AF_METAL_FRAMEWORK}
    ${AF_FOUNDATION_FRAMEWORK}
    ${AF_QUARTZCORE_FRAMEWORK}
    ${AF_MPSGRAPH_FRAMEWORK}
  )
if(BUILD_WITH_MKL)
  target_compile_definitions(afmetal PRIVATE USE_MKL)
  target_compile_definitions(afmetal PRIVATE AF_MKL_INTERFACE_SIZE=${MKL_INTERFACE_INTEGER_SIZE})

  if(MKL_BATCH)
    target_compile_definitions(afmetal PRIVATE AF_USE_MKL_BATCH)
  endif()

  if(AF_WITH_STATIC_MKL)
      target_link_libraries(afmetal PRIVATE MKL::Static)
      target_compile_definitions(afmetal PRIVATE USE_STATIC_MKL)
  else()
      target_link_libraries(afmetal PRIVATE MKL::RT)
  endif()
else()
  target_link_libraries(afmetal
    PRIVATE
      ${CBLAS_LIBRARIES}
      FFTW::FFTW
      FFTW::FFTWF
    )
  if(LAPACK_FOUND AND LAPACKE_FOUND)
    target_link_libraries(afmetal PRIVATE LAPACKE::LAPACKE ${LAPACK_LIBRARIES})
  endif()
endif()

if(LAPACK_FOUND OR BUILD_WITH_MKL)
  target_compile_definitions(afmetal PRIVATE WITH_LINEAR_ALGEBRA)
endif()

af_split_debug_info(afmetal ${AF_INSTALL_LIB_DIR})

install(TARGETS afmetal
  EXPORT ArrayFireMetalTargets
  COMPONENT metal
  PUBLIC_HEADER DESTINATION af
  RUNTIME DESTINATION ${AF_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${AF_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${AF_INSTALL_LIB_DIR}
  FRAMEWORK DESTINATION framework
  INCLUDES DESTINATION ${AF_INSTALL_INC_DIR}
  )

source_group(include REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/include/*)
source_group(api\\cpp REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/cpp/*)
source_group(api\\c   REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/api/c/*)
source_group(backend  REGULAR_EXPRESSION ${ArrayFire_SOURCE_DIR}/src/backend/common/*|${CMAKE_CURRENT_SOURCE_DIR}/*)
source_group(backend\\kernel  REGULAR_EXPRESSION ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*)
source_group("generated files" FILES ${ArrayFire_BINARY_DIR}/src/backend/build_version.hpp ${ArrayFire_BINARY_DIR}/include/af/version.h)
source_group("" FILES CMakeLists.txt)

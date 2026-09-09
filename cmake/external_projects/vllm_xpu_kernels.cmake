include(FetchContent)

set(
  VLLM_XPU_KERNELS_GIT_REPOSITORY
  "https://github.com/vllm-project/vllm-xpu-kernels.git"
  CACHE STRING "vLLM XPU kernels Git repository")
set(
  VLLM_XPU_KERNELS_GIT_TAG
  "7e85c904edd4eecc34f5c8ceea43aacdfbfd3a8a"
  CACHE STRING "vLLM XPU kernels Git revision")

if(DEFINED ENV{VLLM_XPU_KERNELS_SRC_DIR})
  set(VLLM_XPU_KERNELS_SRC_DIR "$ENV{VLLM_XPU_KERNELS_SRC_DIR}")
endif()

if(VLLM_XPU_KERNELS_SRC_DIR)
  cmake_path(ABSOLUTE_PATH VLLM_XPU_KERNELS_SRC_DIR
             NORMALIZE OUTPUT_VARIABLE vllm_xpu_kernels_SOURCE_DIR)
  set(vllm_xpu_kernels_BINARY_DIR
      "${CMAKE_BINARY_DIR}/vllm-xpu-kernels-build")
else()
  FetchContent_Populate(
    vllm_xpu_kernels
    SUBBUILD_DIR "${FETCHCONTENT_BASE_DIR}/vllm-xpu-kernels-subbuild"
    SOURCE_DIR "${FETCHCONTENT_BASE_DIR}/vllm-xpu-kernels-src"
    BINARY_DIR "${CMAKE_BINARY_DIR}/vllm-xpu-kernels-build"
    GIT_REPOSITORY "${VLLM_XPU_KERNELS_GIT_REPOSITORY}"
    GIT_TAG "${VLLM_XPU_KERNELS_GIT_TAG}"
    GIT_SHALLOW FALSE
    GIT_PROGRESS TRUE)
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}"
    "-DSOURCE_DIR=${vllm_xpu_kernels_SOURCE_DIR}"
    "-DPATCH_DIR=${CMAKE_CURRENT_LIST_DIR}/vllm_xpu_kernels/patches"
    -P "${CMAKE_CURRENT_LIST_DIR}/vllm_xpu_kernels/ApplyPatches.cmake"
  COMMAND_ERROR_IS_FATAL ANY)

set(VLLM_XPU_ENABLE_XE_DEFAULT OFF CACHE BOOL "" FORCE)
set(VLLM_XPU_ENABLE_XE3P OFF CACHE BOOL "" FORCE)
set(
  VLLM_CHUNK_PREFILL_CONFIG
  "chunk_prefill_default.conf"
  CACHE STRING "XPU chunk prefill kernel selection")
set(
  VLLM_PAGED_DECODE_CONFIG
  "paged_decode_default.conf"
  CACHE STRING "XPU paged decode kernel selection")
add_subdirectory(
  "${vllm_xpu_kernels_SOURCE_DIR}"
  "${vllm_xpu_kernels_BINARY_DIR}"
  EXCLUDE_FROM_ALL)

install(
  DIRECTORY "${vllm_xpu_kernels_SOURCE_DIR}/vllm_xpu_kernels/"
  DESTINATION vllm_xpu_kernels
  COMPONENT _C
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "py.typed")

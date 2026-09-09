if(NOT SOURCE_DIR OR NOT IS_DIRECTORY "${SOURCE_DIR}")
  message(FATAL_ERROR "SOURCE_DIR must name the vllm-xpu-kernels source tree")
endif()
if(NOT PATCH_DIR OR NOT IS_DIRECTORY "${PATCH_DIR}")
  message(FATAL_ERROR "PATCH_DIR must name the vllm-xpu-kernels patch directory")
endif()

find_package(Git REQUIRED)
file(GLOB patches LIST_DIRECTORIES FALSE "${PATCH_DIR}/*.patch")
list(SORT patches)

foreach(patch IN LISTS patches)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --check --unidiff-zero "${patch}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE apply_check
    OUTPUT_QUIET
    ERROR_QUIET)
  if(apply_check EQUAL 0)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" apply --unidiff-zero "${patch}"
      WORKING_DIRECTORY "${SOURCE_DIR}"
      COMMAND_ERROR_IS_FATAL ANY)
    continue()
  endif()

  execute_process(
    COMMAND
      "${GIT_EXECUTABLE}" apply --check --reverse --unidiff-zero "${patch}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE reverse_check
    OUTPUT_QUIET
    ERROR_QUIET)
  if(NOT reverse_check EQUAL 0)
    get_filename_component(patch_name "${patch}" NAME)
    message(FATAL_ERROR
      "${patch_name} neither applies nor is already applied to ${SOURCE_DIR}")
  endif()
endforeach()

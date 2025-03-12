# Loads and runs CMake code from the file given. Variable reads and writes access the scope of the caller.
# see https://cmake.org/cmake/help/latest/command/include.html for more info
include(FindPackageHandleStandardArgs)

# Include directories.
# find_path: find a directory in PATHS containing files in NAMES, store the directory found into <VAR>.
# if PATHS not set, cmake will search from cache entries (e.g.,<PackageName>_Dir, <PackageName>_ROOT),
#   if still not, then search from system default: CMAKE_PREFIX_PATH including /usr/include, /usr/local/include, /usr/lib ...
# here is to find "torch_npu/csrc/include/ops.h" 
# see https://cmake.org/cmake/help/latest/command/find_path.html#find-path for more info
find_path(TORCH_NPU_INCLUDE_DIRS NAMES torch_npu/csrc/include/ops.h)

# Library dependencies.
# find a library in NAMES from PATHS, store the library found in <VAR>
# if PATHS not set, cmake will search from cache entries (e.g.,<PackageName>_Dir, <PackageName>_ROOT),
#   if still not, then search from system default: CMAKE_PREFIX_PATH including /usr/include, /usr/local/include, /usr/lib ...
# here <VAR> is TORCH_NPU_LIBRARY, NAMES is torch_npu and npu_profiler. cmake first searches libtorch_npu.so from Torch_npu_ROOT.
#   Return directly if exists and leave libnpu_profiler.so behind.
# see https://cmake.org/cmake/help/latest/command/find_library.html#find-library for more info
find_library(TORCH_NPU_LIBRARY NAMES torch_npu npu_profiler)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'")
endif()
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_LIBRARY})

# torch/csrc/python_headers depends Python.h
# see https://cmake.org/cmake/help/latest/module/FindPython.html for more info
find_package(Python COMPONENTS Interpreter Development)
message(STATUS "Python_FOUND: ${Python_FOUND}")
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")

#TODO (chenchiyu): construct modern cmake target for Torch_npu
message(STATUS "Found Torch_npu: TORCH_NPU_LIBRARY: ${TORCH_NPU_LIBRARY}, TORCH_NPU_INCLUDE_DIRS: ${TORCH_NPU_INCLUDE_DIRS}")

# the Torch_npu library is considered to be found if both TORCH_NPU_LIBRARY and TORCH_NPU_INCLUDE_DIRS are valid.
# Then also Torch_npu_FOUND is set to be TRUE.
# This module is almost only used with find_package.
# `find_package(Torch_npu REQUIRED)` in ascend.cmake used to find FindTorch_npu.cmake(or Torch_npuConfig.cmake) and execute all commands written in it.
# `find_package_handle_standard_args` is set in FindTorch_npu.cmake to deliver a signal that Torch_npu is found and set Torch_npu_FOUND to be TRUE.
# see https://cmake.org/cmake/help/latest/module/FindPackageHandleStandardArgs.html#command:find_package_handle_standard_args for more info
find_package_handle_standard_args(Torch_npu DEFAULT_MSG TORCH_NPU_LIBRARY TORCH_NPU_INCLUDE_DIRS)

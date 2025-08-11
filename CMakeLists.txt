cmake_minimum_required(VERSION 3.18)
project(npunotebooks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# append cmake/*.cmake to CMAKE_MODULE_PATH. Useful in command find_package
# see https://cmake.org/cmake/help/latest/variable/CMAKE_MODULE_PATH.html#cmake-module-path for more info
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# the default CMAKE_BUILD_TYPE is Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# set(DEVICE "" CACHE STRING "device string, default empty string")
# string(TOLOWER "${DEVICE}" DEVICE)

# list(APPEND SUPPORTED_DEVICE "ascend" "maca" "camb")

# if(NOT DEVICE)
#   message(FATAL_ERROR "Please specify variable DEVICE of dlinfer!")
# elseif(NOT DEVICE IN_LIST SUPPORTED_DEVICE)
#   message(FATAL_ERROR "Device ${DEVICE} is not supported! Supported devices: ${SUPPORTED_DEVICE}")
# endif()

# Adds a subdirectory to the build.
# The ./npunotebooks specifies the directory in which the source CMakeLists.txt and code files are located.
# see https://cmake.org/cmake/help/latest/command/add_subdirectory.html#add-subdirectory for more info
add_subdirectory(./npunotebooks)
project(ascend_learning)

include(ascend)

# 一般大型项目会用`set`显式设置文件列表，更加可控
# set(CSRC_FILES 
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/init.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attention.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe_gating_topk_softmax.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/op_api_common.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/csrc/torch_npu_utils.cpp
# )

# 非大型项目可以直接用`file`,
# file(GLOB <variable> [LIST_DIRECTORIES true|false] [RELATIVE <path>] [CONFIGURE_DEPENDS] <globbing-expressions>...)
# Generate a list of files that match the <globbing-expressions> and store it into the <variable>
# more info for `file`: https://cmake.org/cmake/help/latest/command/file.html#file
file(GLOB LEARNING_FILES ${CMAKE_CURRENT_SOURCE_DIR}/learn_npu/*.cpp)

# add_library(<name> <type> <sources>)
# Add a library target called <name> to be built from the source files listed in the command invocation.
# see https://cmake.org/cmake/help/latest/command/add_library.html#add-library for more info
add_library(
    ${PROJECT_NAME} SHARED
    ${LEARNING_FILES}
)

set (EXES 
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_executable(test ${EXES})

# Targets can have properties that affect how they are built.
# here is set file_prefix of shared_lib file after being built to be "", which will be "lib" by default.
# see https://cmake.org/cmake/help/latest/command/set_target_properties.html#set-target-properties for more info
set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    PREFIX ""
)

# target_compile_definitions
# 在 GCC 5.1 及以后版本中，libstdc++ 引入了一个新的 C++11 ABI。这个新的 ABI 改变了部分标准库类的内部表示和符号命名。
# 例如 std::string 和 std::list。这样做的原因是为了改进性能和标准兼容性。
# 然而，引入新的 ABI 会打破与现有二进制组件的兼容性，包括以前版本的编译输出和依赖旧 ABI 的库。
# 因此，为了在新旧 ABI 之间提供兼容性，libstdc++ 使用了 _GLIBCXX_USE_CXX11_ABI 宏来切换 ABI。
# _GLIBCXX_USE_CXX11_ABI=0：使用旧的 ABI，这与 GCC 5 之前的版本二进制兼容。
# _GLIBCXX_USE_CXX11_ABI=1：使用新的 C++11 ABI，这是 GCC 5.1 及以后版本的默认设置。
target_compile_definitions(
    ${PROJECT_NAME} PUBLIC
    GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}
)

# Specifies include directories to use when compiling a given target.
# see https://cmake.org/cmake/help/latest/command/target_include_directories.html#target-include-directories for more info
target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_NPU_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
)

target_include_directories(
    test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_NPU_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    
)

# target_include_directories(
#     test PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     )

# Specify libraries or flags to use when linking a given target and/or its dependents.
# Be aware that all library <item>s included should be a variable(type:list) of library files or or single library file.
# see https://cmake.org/cmake/help/latest/command/target_link_libraries.html#target-link-libraries for more info
target_link_libraries(
    # if PRIVATE here, dependent libraries will not be transmitted to any other file which depends on ${PROJECT_NAME}.
    # Here I should set PUBLIC for executable file `test` depends on all ${PROJECT_NAME} depends.
    ${PROJECT_NAME} PUBLIC
    ${TORCH_NPU_LIBRARY}
    torch # it's defined in <path_to_torch>/share/cmake/Torch/TorchConfig.cmake
    # Python::Python must be in this postion for torch depends on Python::Python and ${TORCH_NPU_LIBRARY} depends on torch and Python::Python.
    # 链接器按顺序解析符号，依赖的库应放在后面
    Python::Python # defined in find_package(Python COMPONENTS Interpreter Development)
    ${CANN_LIBRARY}

)

target_link_libraries(
    test PRIVATE
    ${PROJECT_NAME}
    # Python::Python
    # torch
    # ${TORCH_NPU_LIBRARY}
    # ${CANN_LIBRARY}
)

# This command generates installation rules for a project.
# Install rules specified by calls to the install() command within a source directory are executed in order during installation.
# It's useful when execute `cmake --install <dir>`
install(
    TARGETS ${PROJECT_NAME} test
    # DESTINATION ${CMAKE_SOURCE_DIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_SOURCE_DIR}/lib
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
    ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/lib
    RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin
)
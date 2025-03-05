# save COMMAND output into Torch_DIR
# any variables like <PackageName>_DIR is a special variable called cache entry.
# find_package(<PackageName>) this command exactly searches <PackageName>Config.cmake in <PackageName>_DIR
# if <PackageName>_DIR is not set, cmake will search CMAKE_PREFIX_PATH when `find_package` and some system paths by default.
# see https://cmake.org/cmake/help/latest/command/find_package.html for more info
execute_process(
    COMMAND python -c "from torch.utils import cmake_prefix_path; \
    print(cmake_prefix_path + '/Torch', end='')"
    OUTPUT_VARIABLE Torch_DIR
)

# save COMMAND output into Torch_npu_ROOT
# cache entry like <PackageName>_ROOT stores paths to search for find_package, find_library.
# find_library: searchs libraries from <PackageName>/lib or <PackageName>/lib64 or <PackgeName>/lib/<arch>.
# 注意：如果定义了<PackageName>_DIR cmake变量，那么<PackageName>_ROOT 不起作用
# see https://cmake.org/cmake/help/latest/command/find_package.html
# and https://cmake.org/cmake/help/latest/envvar/PackageName_ROOT.html#envvar:%3CPackageName%3E_ROOT
# and https://cmake.org/cmake/help/latest/command/find_library.html for more info
execute_process(
    COMMAND python -c "from importlib.metadata import distribution; \
    print(str(distribution('torch_npu').locate_file('torch_npu')), end='')"
    OUTPUT_VARIABLE Torch_npu_ROOT
)

execute_process(
    COMMAND python -c "import torch; \
    print('1' if torch.compiled_with_cxx11_abi() else '0', end='')"
    OUTPUT_VARIABLE _GLIBCXX_USE_CXX11_ABI
)

# find <PackageName>Config.cmake or Find<PackageName>.cmake to load external cmake config.
# see https://cmake.org/cmake/help/latest/command/find_package.html for more info
find_package(Torch REQUIRED)

# Attention! There's no any Torch_npuConfig.cmake in Torch_npu_ROOT. This command searches cmake/FindTorch_npu.cmake
# So do the next two `find_package` for CANNToolkit and ATB.
find_package(Torch_npu REQUIRED)
find_package(CANNToolkit REQUIRED)
find_package(ATB)

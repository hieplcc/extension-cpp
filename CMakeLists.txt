cmake_minimum_required(VERSION 3.12)
project(fast_sigmoid_cpp_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Automatically find PyTorch installation path
if(NOT DEFINED CMAKE_PREFIX_PATH)
    execute_process(
        COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_FOUND
    )
    
    if(TORCH_FOUND EQUAL 0)
        set(CMAKE_PREFIX_PATH ${TORCH_PREFIX_PATH})
        message(STATUS "Found PyTorch at: ${TORCH_PREFIX_PATH}")
    else()
        message(FATAL_ERROR "Could not find PyTorch installation. Make sure PyTorch is installed and accessible via python3.")
    endif()
endif()

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add include directories
include_directories(extension_cpp/csrc/include)

# Add the example executable
add_executable(fast_sigmoid_cpp_test test/fast_sigmoid/fast_sigmoid_cpp_test.cpp)

# Set output directory for the test executable
set_target_properties(fast_sigmoid_cpp_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/fast_sigmoid"
)

# Add the fast_sigmoid implementation
add_library(fast_sigmoid_lib STATIC 
    extension_cpp/csrc/fast_sigmoid.cpp
)

# Link libraries
target_link_libraries(fast_sigmoid_lib "${TORCH_LIBRARIES}")
target_link_libraries(fast_sigmoid_cpp_test fast_sigmoid_lib "${TORCH_LIBRARIES}")
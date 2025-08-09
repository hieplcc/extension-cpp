cmake_minimum_required(VERSION 3.12)
project(external_app_demo)

# Detect PyTorch CMake prefix path using Python
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FOUND
)

# Check if PyTorch was found
if(NOT TORCH_FOUND EQUAL 0)
    message(FATAL_ERROR "Could not find PyTorch. Make sure it's installed and accessible via python3.")
endif()

# Add PyTorch installation path (from Python)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX_PATH}")

# Append Python virtual env path if exists
if(DEFINED ENV{VIRTUAL_ENV})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}")
endif()

# Add fast_sigmoid install path
if(DEFINED FAST_SIGMOID_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${FAST_SIGMOID_PREFIX_PATH}")
endif()

message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

# Find fast_sigmoid package
find_package(fast_sigmoid REQUIRED)

# Build external app
add_executable(external_app main.cpp)
target_link_libraries(external_app PRIVATE fast_sigmoid_lib)

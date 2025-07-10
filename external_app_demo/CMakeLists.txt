cmake_minimum_required(VERSION 3.12)
project(external_app_demo)

# Detect PyTorch CMake prefix path using Python
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FOUND
)

if(NOT TORCH_FOUND EQUAL 0)
    message(FATAL_ERROR "Could not find PyTorch. Make sure it's installed and accessible via python3.")
endif()

# Clear or initialize CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH "")

# Add PyTorch installation path (from Python)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_PREFIX_PATH}")

# Add fast_sigmoid install path
list(APPEND CMAKE_PREFIX_PATH "../install_demo/usr/local")

message(STATUS "CMAKE_PREFIX_PATH = ${CMAKE_PREFIX_PATH}")

# Find fast_sigmoid package
find_package(fast_sigmoid REQUIRED)

# Build external app
add_executable(external_app main.cpp)
target_link_libraries(external_app PRIVATE fast_sigmoid_lib)

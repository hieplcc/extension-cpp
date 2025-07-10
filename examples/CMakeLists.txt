cmake_minimum_required(VERSION 3.12)
project(fast_sigmoid_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the fast_sigmoid library
set(FAST_SIGMOID_TARGET fast_sigmoid_lib)

# Build the fast_sigmoid test
add_executable(fast_sigmoid_cpp_test fast_sigmoid_cpp_test.cpp)
set_target_properties(fast_sigmoid_cpp_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples"
)
target_link_libraries(fast_sigmoid_cpp_test ${FAST_SIGMOID_TARGET})

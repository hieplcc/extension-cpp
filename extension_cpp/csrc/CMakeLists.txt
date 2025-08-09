cmake_minimum_required(VERSION 3.12)
project(libfast_sigmoid VERSION 1.0.0)

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

# Create the fast_sigmoid library
add_library(fast_sigmoid_lib STATIC 
    fast_sigmoid.cpp
)

# Set library properties
set_target_properties(fast_sigmoid_lib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "fast_sigmoid"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Link PyTorch libraries
target_link_libraries(fast_sigmoid_lib "${TORCH_LIBRARIES}")

# Set include directories for the library
target_include_directories(fast_sigmoid_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../utils>
    $<INSTALL_INTERFACE:include>
)

# Installation rules
include(GNUInstallDirs)

install(TARGETS fast_sigmoid_lib
    EXPORT fast_sigmoid_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Create and install config files for find_package support
install(EXPORT fast_sigmoid_targets
    FILE fast_sigmoid_targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fast_sigmoid
)

# Create version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/fast_sigmoid_config_version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Create config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fast_sigmoid_config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/fast_sigmoidConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fast_sigmoid
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/fast_sigmoidConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/fast_sigmoid_config_version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fast_sigmoid
)

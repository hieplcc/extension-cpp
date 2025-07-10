cmake_minimum_required(VERSION 3.12)
project(fast_sigmoid_project)

# Add subdirectories
add_subdirectory(extension_cpp/csrc)
add_subdirectory(examples)

include(CPack)
# General CPack metadata
set(CPACK_PACKAGE_NAME "fast_sigmoid")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})

# Select generators based on platform
if(WIN32)
    set(CPACK_GENERATOR "ZIP")
elseif(UNIX)
    set(CPACK_GENERATOR "TGZ;DEB")
endif()
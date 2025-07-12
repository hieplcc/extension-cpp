cmake_minimum_required(VERSION 3.12)
project(fast_sigmoid_project)

# Add subdirectories
add_subdirectory(extension_cpp/csrc)
add_subdirectory(examples)

# Option to enable/disable CPack
option(ENABLE_CPACK "Enable CPack packaging" OFF)

# If CPack is enabled, include CPack configuration
if(ENABLE_CPACK)
    # General CPack metadata
    set(CPACK_PACKAGE_NAME "fast_sigmoid")
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})

    # Select generators based on platform
    if(WIN32)
        set(CPACK_GENERATOR "ZIP")
    elseif(UNIX)
        set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Aiden")
        set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Fast Sigmoid C++ library")

        set(CPACK_GENERATOR "TGZ;DEB")
    endif()

    include(CPack)
endif()
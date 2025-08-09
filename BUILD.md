## Build and Install Workflow
### Prerequisites

- **C++ Version**: Minimum C++17 
- **CMake**: Version 3.12 or higher
- **Python 3**, **PyTorch**
---
### Build the `fast_sigmoid` library and examples
- Clone and navigate to the project directory:
    ```bash
    git clone https://github.com/hieplcc/extension-cpp.git
    cd /extension-cpp
    ```
- Create and enter the build directory:
    ```bash
    mkdir build
    cd build
    ```
- Configure with `CMake` and build:
    ```bash
    cmake ..
    make
    ```
- Check the example executable:
    ```bash
    ls examples/fast_sigmoid_cpp_test
    ```

---

### Install the Library
- From inside the build directory, configure the install prefix for `fast_sigmoid` library:
    ```bash
    # Configure the install prefix, where the library will be installed.
    cmake .. -DCMAKE_INSTALL_PREFIX=/your_install_path
    ```
- **Recommended:** install the library into your Python virtual environment if active:
    ```bash
    cmake .. -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV
    ```
- Install the library:
    ```bash
    make install
    ```
#### Notes:
 After running `make install`, the library files will be placed in the specified install path, typically under:
- `include/extension_cpp/fast_sigmoid/` for headers
- `lib/libfast_sigmoid.a` for compiled libraries
- `lib/cmake/fast_sigmoid/` for CMake configuration files

---

### Demo: Use the library in an External App with `find_package(fast_sigmoid)`
- Go to the `externa_app_demo` folder:
    ```bash
    cd extension_cpp/external_app_demo
    ```
- Create and enter the build directory:
    ```bash
    mkdir build
    cd build
    ```
- Configure where to find the `fast_sigmoid` library:
    ```bash
    cmake .. -DFAST_SIGMOID_PREFIX_PATH=/your_install_path
    ```
    **Note:** if you installed the library into your Python virtual environment, just run: `cmake ..`
-  Build and run the external app:
    ```bash
    # Build
    make

    # Run the external app
    ./external_app
    ```
---

### Packaging with CPack

You can generate distributable packages (such as `.tar.gz`, `.deb`, or `.zip`) using `CPack`.
- In the `extension_cpp` build folder, configure the project with `CPack` enabled:
   ```bash
   cd extension-cpp
   mkdir -p build && cd build
   cmake .. -DENABLE_CPACK=ON
   ```
- Generate the package(s):
   ```bash
   cpack
   ```
#### Notes:
The generated package files will appear in your `build/` directory. The formats depend on your platform and the CPack generators enabled in `CMakeLists.txt`.
- On Linux, you may get `.tar.gz` and `.deb` files.
- On Windows, you may get `.zip` files.

## Build and Install Workflow

### 1. Build the Library and Examples

```bash
# Go to the extension-cpp folder
cd extension-cpp

# Create and enter the build directory
mkdir build
cd build

# Configure and build
cmake ..
make

# Check the example executable
ls examples/fast_sigmoid_cpp_test
```

---

### 2. Install the Library

```bash
# From inside the build directory
make install DESTDIR=../install_demo
```
- This will stage the install tree under `extension-cpp/install_demo/usr/local`.

---

### 3. Demo: Use the Library in an External App with find_package(fast_sigmoid)

```bash
# Go to the external app demo folder
cd ../external_app_demo

# Create and enter the build directory
mkdir build
cd build

# Configure (CMakeLists.txt will set CMAKE_PREFIX_PATH automatically)
cmake ..

# Build the external app
make

# Run the external app
./external_app
```

---

**Summary:**  
- Build and install the library and examples from the main project.
- Use `make install DESTDIR=../install_demo` to stage the install.
- Build and run an external app that uses `find_package(fast_sigmoid)` to link to your installed library.
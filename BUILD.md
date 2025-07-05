# Fast Sigmoid C++ Extension - Build Guide

### Prerequisites

- **C++ Version**: Minimum C++17 
- **CMake**: Version 3.12 or higher
- **Python 3**: For automatic PyTorch path detection
- **PyTorch**: Installed and accessible via Python (for libtorch libraries)

### Build Instructions

#### Step 1: Clone and navigate to Project Directory
```bash
git clone https://github.com/hieplcc/extension-cpp.git
cd /path/to/extension-cpp
```

#### Step 2: Create Build Directory
```bash
mkdir build
cd build
```

#### Step 3: Configure with CMake
```bash
cmake ..
```
#### Step 4: Build the Project
```bash
make
```
#### Step 5: Run the Test Application
```bash
./test/fast_sigmoid/fast_sigmoid_cpp_test
```

#### Build Output

After successful build, you'll find:
- **Library**: `libfast_sigmoid_lib.a` (static library)
- **Test Executable**: `test/fast_sigmoid/fast_sigmoid_cpp_test`

### Expected Output

The test application should produce output similar to:
```
Input: -15.0000
-11.6667
 -8.3333
 -5.0000
 -1.6667
  1.6667
  5.0000
  8.3333
 11.6667
 15.0000
[ CPUFloatType{10} ]
Fast Sigmoid:  0.0000
 0.0000
 0.0002
 0.0067
 0.1589
 0.8411
 0.9933
 0.9998
 1.0000
 1.0000
[ CPUFloatType{10} ]
``` 
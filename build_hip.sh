#!/bin/bash

# Build script for Qwen600 AMD ROCm/HIP version

echo "Building Qwen600 for AMD ROCm/HIP..."

# Check if ROCm is installed
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found. Please install ROCm/HIP first."
    echo "Visit: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    exit 1
fi

# Check if required ROCm packages are available
echo "Checking ROCm installation..."

# Check hipblas
if ! pkg-config --exists hipblas; then
    echo "Warning: hipblas not found via pkg-config. Trying alternative detection..."
fi

# Check hipcub
if ! pkg-config --exists hipcub; then
    echo "Warning: hipcub not found via pkg-config. Trying alternative detection..."
fi

# Create build directory
mkdir -p build_hip
cd build_hip

# Configure with CMake
echo "Configuring with CMake..."
cmake -f ../CMakeLists.hip.txt \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DCMAKE_HIP_ARCHITECTURES="gfx906;gfx908;gfx90a;gfx1030;gfx1100" \
    ..

# Build
echo "Building..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Executable: build_hip/qwen600_hip"
    echo ""
    echo "Usage:"
    echo "  ./build_hip/qwen600_hip <model_dir> [options]"
    echo ""
    echo "Example:"
    echo "  ./build_hip/qwen600_hip ./models/qwen600 -t 0.7 -p 0.9"
else
    echo "Build failed!"
    exit 1
fi

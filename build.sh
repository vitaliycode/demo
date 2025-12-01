#!/bin/bash

# Build script for SmolVLM Demo
# Usage: ./build.sh [cpu|gpu]

set -e

MODE=${1:-cpu}

echo "SmolVLM Demo Builder"
echo ""

# Check if mode is valid
if [ "$MODE" != "cpu" ] && [ "$MODE" != "gpu" ]; then
    echo "ERROR: Invalid mode: $MODE"
    echo "Usage: ./build.sh [cpu|gpu]"
    exit 1
fi

echo "Build mode: $MODE"
echo ""

# Check Docker installation
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi
echo "Docker found: $(docker --version)"

if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed"
    exit 1
fi
echo "Docker Compose found: $(docker-compose --version)"
echo ""

# Record start time
BUILD_START=$(date +%s)

# Build based on mode
if [ "$MODE" = "gpu" ]; then
    echo "Building GPU version..."
    echo "This may take several minutes..."
    echo ""
    
    if docker-compose -f docker-compose.gpu.yml build --progress=plain; then
        echo ""
        echo "GPU build completed"
    else
        echo ""
        echo "ERROR: GPU build failed"
        exit 1
    fi
else
    echo "Building CPU version..."
    echo "This may take several minutes..."
    echo ""
    
    if docker-compose build --progress=plain; then
        echo ""
        echo "CPU build completed"
    else
        echo ""
        echo "ERROR: CPU build failed"
        exit 1
    fi
fi

# Calculate build time
BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))
echo ""
echo "Build time: ${BUILD_TIME} seconds"

# Show built images
echo ""
echo "Built images:"
docker images | grep smolvlm-demo || echo "No images found with name 'smolvlm-demo'"

echo ""
echo "Build complete!"
echo ""
echo "To start: ./start.sh $MODE"
echo ""


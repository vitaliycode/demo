#!/bin/bash

# Build script for SmolVLM Demo
# Usage: ./build.sh [cpu|gpu]

set -e

MODE=${1:-cpu}

echo "==================================="
echo "SmolVLM Demo Builder"
echo "==================================="
echo ""

# Check if mode is valid
if [ "$MODE" != "cpu" ] && [ "$MODE" != "gpu" ]; then
    echo "❌ Invalid mode: $MODE"
    echo "Usage: ./build.sh [cpu|gpu]"
    exit 1
fi

# Build based on mode
if [ "$MODE" = "gpu" ]; then
    echo "🔨 Building GPU version..."
    docker-compose -f docker-compose.gpu.yml build
else
    echo "🔨 Building CPU version..."
    docker-compose build
fi

echo ""
echo "✅ Build complete!"
echo ""
echo "To start: ./start.sh $MODE"
echo ""


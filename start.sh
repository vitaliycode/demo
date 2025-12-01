#!/bin/bash

# Start script for SmolVLM Demo
# Usage: ./start.sh [cpu|gpu]

set -e

MODE=${1:-cpu}

echo "==================================="
echo "SmolVLM Demo Launcher"
echo "==================================="
echo ""

# Check if mode is valid
if [ "$MODE" != "cpu" ] && [ "$MODE" != "gpu" ]; then
    echo "❌ Invalid mode: $MODE"
    echo "Usage: ./start.sh [cpu|gpu]"
    exit 1
fi

# Create necessary directories
mkdir -p hf-cache
mkdir -p outputs/chat
mkdir -p outputs/ocr

echo "✅ Created output directories"

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env from .env.example"
    else
        echo "⚠️  .env.example not found. Using default settings."
    fi
fi

# Start based on mode
if [ "$MODE" = "gpu" ]; then
    echo ""
    echo "🚀 Starting SmolVLM Demo in GPU mode..."
    echo ""
    
    # Check for NVIDIA Docker support
    if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "❌ GPU support not available or NVIDIA Docker not installed"
        echo "Please ensure:"
        echo "  1. You have an NVIDIA GPU"
        echo "  2. NVIDIA drivers are installed"
        echo "  3. NVIDIA Docker runtime is installed"
        echo ""
        echo "Falling back to CPU mode..."
        MODE="cpu"
    else
        echo "✅ GPU support detected"
        docker-compose -f docker-compose.gpu.yml up -d --build
    fi
fi

if [ "$MODE" = "cpu" ]; then
    echo ""
    echo "🚀 Starting SmolVLM Demo in CPU mode..."
    echo ""
    docker-compose up -d --build
fi

echo ""
echo "==================================="
echo "✅ SmolVLM Demo is starting!"
echo "==================================="
echo ""
echo "📝 Logs: docker-compose logs -f"
echo "🌐 Frontend: http://localhost:8000"
echo "🎨 Gradio UI: http://localhost:8000/ui"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🏥 Health: http://localhost:8000/api/health"
echo ""
echo "⏸️  To stop: docker-compose down"
echo ""


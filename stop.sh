#!/bin/bash

# Stop script for SmolVLM Demo

set -e

echo "==================================="
echo "Stopping SmolVLM Demo"
echo "==================================="
echo ""

# Stop CPU version
if docker-compose ps | grep -q smolvlm-demo-cpu; then
    echo "🛑 Stopping CPU version..."
    docker-compose down
fi

# Stop GPU version
if docker-compose -f docker-compose.gpu.yml ps | grep -q smolvlm-demo-gpu; then
    echo "🛑 Stopping GPU version..."
    docker-compose -f docker-compose.gpu.yml down
fi

echo ""
echo "✅ SmolVLM Demo stopped successfully"
echo ""


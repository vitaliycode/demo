#!/bin/bash

# Stop script for SmolVLM Demo

set -e

echo "Stopping SmolVLM Demo"

STOPPED_SOMETHING=false

# Check and stop CPU version
echo "Checking for CPU version containers..."
if docker-compose ps 2>/dev/null | grep -q smolvlm-demo-cpu; then
    echo "Found CPU version running"
    echo "Saving last 50 lines of logs..."
    docker-compose logs --tail=50 > "logs_cpu_$(date +%Y%m%d_%H%M%S).txt" 2>&1 || true
    
    echo "Stopping CPU version..."
    if docker-compose down; then
        echo "CPU version stopped"
        STOPPED_SOMETHING=true
    else
        echo "WARNING: Error stopping CPU version"
    fi
else
    echo "No CPU version containers found"
fi

echo ""

# Check and stop GPU version
echo "Checking for GPU version containers..."
if docker-compose -f docker-compose.gpu.yml ps 2>/dev/null | grep -q smolvlm-demo-gpu; then
    echo "Found GPU version running"
    echo "Saving last 50 lines of logs..."
    docker-compose -f docker-compose.gpu.yml logs --tail=50 > "logs_gpu_$(date +%Y%m%d_%H%M%S).txt" 2>&1 || true
    
    echo "Stopping GPU version..."
    if docker-compose -f docker-compose.gpu.yml down; then
        echo "GPU version stopped"
        STOPPED_SOMETHING=true
    else
        echo "WARNING: Error stopping GPU version"
    fi
else
    echo "No GPU version containers found"
fi

echo ""

# Check for any remaining containers
echo "Checking for remaining SmolVLM containers..."
REMAINING=$(docker ps -a | grep smolvlm-demo 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo "WARNING: Some containers may still be present:"
    echo "$REMAINING"
else
    echo "No SmolVLM containers remaining"
fi

echo ""

if [ "$STOPPED_SOMETHING" = true ]; then
    echo "SmolVLM Demo stopped successfully"
else
    echo "INFO: No running containers found"
fi
echo ""


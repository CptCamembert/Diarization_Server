#!/bin/bash

set -e

echo "Building diarization server Docker image..."

# Navigate to the parent directory where Dockerfile is located
cd "$(dirname "$0")/.."

# Build the Docker image
docker build -t diarization-server .

echo "Docker image 'diarization-server' built successfully!"
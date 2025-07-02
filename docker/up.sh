#!/bin/bash

set -e

echo "Starting diarization server..."

# Create host directories for models and embeddings if they don't exist
mkdir -p ./model ./embeddings

# Run the Docker container
# -d: Run in detached mode
# -p: Map host port 8000 to container port 8000
# -v: Mount host directories to container directories
# --name: Give container a name for easy reference
docker run -d \
  -p 8000:8000 \
  -v "$(pwd)/model:/app/model" \
  -v "$(pwd)/embeddings:/app/embeddings" \
  --name diarization-server \
  diarization-server

echo "Diarization server started on http://localhost:8000"
echo "Check logs with: docker logs diarization-server"
echo "Models directory mounted from: $(pwd)/model"
echo "Embeddings directory mounted from: $(pwd)/embeddings"
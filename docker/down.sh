#!/bin/bash

echo "Stopping diarization server container..."

# Check if the container exists
if docker ps -a --format '{{.Names}}' | grep -q '^diarization-server$'; then
  # Stop and remove the container
  docker stop diarization-server
  docker rm diarization-server
  echo "Diarization server container stopped and removed."
else
  echo "Diarization server container is not running."
fi
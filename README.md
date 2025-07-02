# Diarization Server Docker Setup

This guide provides step-by-step instructions to build and run the speaker diarization server using Docker.

## Overview

The diarization server is a FastAPI-based application that performs speaker recognition using SpeechBrain models. It can identify speakers from audio data and learn new speaker embeddings.

## Prerequisites

- Docker installed and running on your system
- At least 2GB of free disk space (for models and dependencies)
- Internet connection (for downloading SpeechBrain models on first run)

## Quick Start

### 1. Navigate to the Server Directory
```bash
cd /home/maximilian/diarization_clean/src/server_side/diarization_server
```

### 2. Build the Docker Image
```bash
./docker/build.sh
```

This will:
- Create a Docker image named `diarization-server`
- Install all Python dependencies
- Download SpeechBrain models (happens on first container run)

### 3. Start the Container
```bash
./docker/up.sh
```

This will:
- Create `model/` and `embeddings/` directories if they don't exist
- Start the container in detached mode
- Mount local directories for persistent storage
- Make the server available at http://localhost:8000

### 4. Verify Health
```bash
curl -X GET http://localhost:8000/health
```

Expected response: `{"status":"ok"}`

### 5. Stop the Container
```bash
./docker/down.sh
```

## API Endpoints

### Health Check
- **GET** `/health`
- Returns: `{"status": "ok"}`

### Speaker Recognition
- **POST** `/diarize`
- Body: Raw audio data (16-bit PCM, 16kHz)
- Query params: `top_n` (optional, default=1, use -1 for all speakers)
- Returns: List of speakers with similarity scores

### Teach Speaker
- **POST** `/diarize_teach`
- Body: Raw audio data (16-bit PCM, 16kHz)
- Query params: `name` (required, speaker name)
- Returns: `{"success": true}`

## Directory Structure

```
diarization_server/
├── Dockerfile                 # Container definition
├── requirements.txt          # Python dependencies
├── main.py                  # FastAPI application
├── speechbrain_helper.py    # Speaker recognition logic
├── docker/
│   ├── build.sh            # Build script
│   ├── up.sh              # Start script
│   └── down.sh            # Stop script
├── model/                  # SpeechBrain models (auto-downloaded)
└── embeddings/            # Speaker embeddings (persistent)
```

## Required Python Packages

The following packages are strictly necessary for the diarization server:

```
fastapi==0.104.1           # Web framework
uvicorn[standard]==0.24.0  # ASGI server
numpy==1.24.3             # Audio data processing
torch==2.1.0              # PyTorch framework
torchaudio==2.1.0         # Audio processing
speechbrain==0.5.16       # Speaker recognition models
pathlib2==2.3.7.post1     # Path utilities
```

## Troubleshooting

### Container Won't Start
1. Check Docker is running: `docker info`
2. Verify port 8000 is available: `netstat -tulpn | grep 8000`
3. Check container logs: `docker logs diarization-server`

### Network Connectivity Issues
If you see errors like "Temporary failure in name resolution" or "Failed to resolve 'cdn-lfs.hf.co'":

1. **Check Internet Connection**: Ensure your system has internet access
   ```bash
   ping huggingface.co
   ```

2. **DNS Resolution Issues**: If DNS is failing, try:
   ```bash
   # Check DNS settings
   cat /etc/resolv.conf
   # Try alternative DNS servers
   echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf
   ```

3. **Corporate Firewall/Proxy**: If behind a corporate firewall:
   - Contact your IT administrator about accessing HuggingFace Hub
   - You may need to configure proxy settings for Docker

4. **Pre-download Models**: Download models manually when you have connectivity:
   ```bash
   # Run container with internet access first to download models
   ./docker/up.sh
   # Models will be cached in the model/ directory for offline use
   ```

5. **Offline Mode**: The server now includes improved offline fallback:
   - If model files exist locally, they will be used automatically
   - Network connectivity is checked before attempting downloads
   - Clear error messages are provided for troubleshooting

### Model Download Issues
- First run requires internet connection to download SpeechBrain models
- Models are cached in the `model/` directory for subsequent runs
- If download fails, delete `model/` directory and restart container when connectivity is restored
- The server now gracefully handles network failures and provides informative error messages

### Audio Processing Warnings
- TorchAudio backend warnings are non-critical
- The server will still function correctly for speaker recognition

### Permission Issues
- Ensure Docker scripts are executable: `chmod +x docker/*.sh`
- Check Docker daemon permissions if build fails

## Performance Notes

- **First startup**: 2-5 minutes (model download)
- **Subsequent startups**: 10-30 seconds
- **Memory usage**: ~1-2GB RAM
- **CPU**: Works on CPU, GPU acceleration not required

## Data Persistence

- **Speaker embeddings**: Stored in `embeddings/` directory
- **Models**: Cached in `model/` directory
- Both directories are mounted as volumes for persistence across container restarts

## Example Usage

### Test with curl:
```bash
# Health check
curl -X GET http://localhost:8000/health

# Speaker recognition (requires audio file)
curl -X POST http://localhost:8000/diarize \
  --data-binary @audio_sample.raw \
  -H "Content-Type: application/octet-stream"

# Teach new speaker
curl -X POST "http://localhost:8000/diarize_teach?name=NewSpeaker" \
  --data-binary @speaker_audio.raw \
  -H "Content-Type: application/octet-stream"
```

## Success Indicators

When the container is running correctly, you should see:
- ✅ Health endpoint returns `{"status":"ok"}`
- ✅ Container status shows "Up" in `docker ps`
- ✅ Logs show "Uvicorn running on http://0.0.0.0:8000"
- ✅ Any existing speaker embeddings are loaded on startup
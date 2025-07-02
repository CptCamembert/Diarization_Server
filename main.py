from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from speechbrain_helper import SpeechBrainRecognizer
import numpy as np
import uvicorn
import os
import json

app = FastAPI()


# Initialize the SpeechBrainRecognizer with environment variable paths
model = SpeechBrainRecognizer(rate=16000)

@app.post("/diarize")
async def diarize(request: Request):
    """
    Endpoint to perform speaker diarization on the provided audio bytes.
    Parameters:
    - audio data as binary in the request body
    - top_n parameter as query parameter to specify how many top speakers to return
      If top_n = -1, return all speakers. If not provided, return only the top speaker.
    """
    try:
        # Read the audio file from the request
        np_buffer = await request.body()
        audio_data = np.frombuffer(np_buffer, dtype=np.int16)
        
        # Get top_n from query parameters
        query_params = request.query_params
        top_n_str = query_params.get("top_n", "1")
        try:
            top_n = int(top_n_str)
        except ValueError:
            top_n = 1

        # Perform diarization
        speakers = model.recognize(audio_data)
        
        # Sort speakers by score in descending order
        sorted_speakers = sorted(speakers.items(), key=lambda x: x[1], reverse=True)
        
        # Return all speakers if top_n is -1, otherwise return the top N speakers
        if top_n == -1:
            result_speakers = sorted_speakers
        else:
            result_speakers = sorted_speakers[:top_n]
            
        # Format the response - always return a dictionary with a list of speakers
        return JSONResponse({
            "speakers": [{"speaker": name, "score": float(score)} for name, score in result_speakers]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/diarize_teach")
async def diarize_teach(request: Request):
    """
    Endpoint to perform speaker diarization on the provided audio bytes.
    Parameters:
    - audio data as binary in the request body
    - top_n parameter as query parameter to specify how many top speakers to return
      If top_n = -1, return all speakers. If not provided, return only the top speaker.
    """
    try:        
        # Read the audio file from the request
        np_buffer = await request.body()
        audio_data = np.frombuffer(np_buffer, dtype=np.int16)
        
        # Get top_n from query parameters
        query_params = request.query_params
        name = query_params.get("name", "")
        for char in [" ", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"]:
            name = str.replace(name, char, "-")

        # create embedding and save it
        model.update_embedding(name, audio_data)    
        model.save_embeddings()    
            
        # Return success response
        return {
            "success": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
    
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "ok"
    }

# This allows the file to be run directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

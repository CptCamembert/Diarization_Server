from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.inference.VAD import VAD
from speechbrain.utils.fetching import LocalStrategy
import glob
import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
from queue import Queue
import time
import requests

# Get paths from environment variables with fallback defaults
MODEL_DIR = os.environ.get('MODEL_DIR', 'tmp_model')
EMBEDDINGS_DIR = os.environ.get('EMBEDDINGS_DIR', 'embeddings')

class SpeechBrainRecognizer:
    """
    SpeechBrainRecognizer class for recognizing speakers in real-time.
    """

    def __init__(self, rate=16000,):
        """Initialize SpeechBrain speaker recognition model."""
        print("#" * 80)
        print(f"Initializing SpeechBrainRecognizer...")
        
        # Ensure embeddings directory exists
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        
        # Load the speaker recognition model
        # Check if model files already exist in the specified directory
        model_path = MODEL_DIR
        print(f"Using model directory: {model_path}")
        
        # Try to load model with network connectivity handling
        self.model = self._load_model_with_fallback(model_path)
        
        self.started = False
        self.sample_rate = rate  # Sample rate from audio streamer
        self.ref_embeddings = {}  # Dictionary to store reference speaker embeddings
        self.load_embeddings()  # Load existing speaker embeddings

        print("SpeechBrain model initialized")

    def _check_network_connectivity(self):
        """Check if we have network connectivity to HuggingFace."""
        try:
            response = requests.get("https://huggingface.co", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _model_files_exist(self, model_path):
        """Check if required model files exist locally."""
        required_files = ['hyperparams.yaml', 'custom.py', 'embedding_model.ckpt']
        return all(os.path.exists(os.path.join(model_path, f)) for f in required_files)

    def _load_model_with_fallback(self, model_path):
        """Load model with network connectivity fallback."""
        try:
            # First, check if model files already exist locally
            if self._model_files_exist(model_path):
                print("Model files found locally, loading from cache...")
                return SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb", 
                    savedir=model_path,            
                    local_strategy=LocalStrategy.COPY_SKIP_CACHE
                )
            
            # Check network connectivity
            if not self._check_network_connectivity():
                raise ConnectionError("No network connectivity to download models")
            
            print("Downloading model from HuggingFace Hub...")
            return SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir=model_path,            
                local_strategy=LocalStrategy.COPY_SKIP_CACHE
            )
            
        except Exception as e:
            print(f"Error loading model: {e}")
            
            # If we have partial model files, try to use them
            if os.path.exists(model_path) and os.listdir(model_path):
                print("Attempting to use existing model files...")
                try:
                    return SpeakerRecognition.from_hparams(
                        source=model_path,  # Use local path directly
                        savedir=model_path
                    )
                except Exception as local_e:
                    print(f"Failed to load local model: {local_e}")
            
            # If all else fails, provide a helpful error message
            raise RuntimeError(
                f"Failed to load SpeechBrain model. "
                f"Network error: {e}. "
                f"Please ensure internet connectivity or download the model manually. "
                f"Model directory: {model_path}"
            )

    def load_embeddings(self):
        """
        Load speaker embeddings from the embeddings directory.
        """
        print(f"Loading embeddings from {EMBEDDINGS_DIR}")
        # Ensure directory exists
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        
        # Load all embeddings from the specified directory
        for filepath in glob.glob(f"{EMBEDDINGS_DIR}/*.pt"):
            # Extract speaker name from the filename
            speaker_name = filepath.split('/')[-1].split('_')[0]
            # Load embedding and store it in the dictionary
            self.ref_embeddings[speaker_name] = torch.load(filepath)
            print(f"Loaded embedding for {speaker_name}")

    def update_embedding(self, name, audio_buffer):
        """
        Update the speaker embedding with new audio data.

        Args:
            name (str): The name of the speaker to update the embedding for.
        """
        # Convert update audio buffer to tensor and generate new embedding
        try:
            wavs = torch.from_numpy(np.array(audio_buffer))
            embedding = self.model.encode_batch(wavs)#, 1, True)
            # Update the existing embedding by averaging with the new embedding or create a new one
            if name in self.ref_embeddings:
                self.ref_embeddings[name] = (self.ref_embeddings[name] + embedding * 0.5) / 1.5
            else:
                self.ref_embeddings[name] = embedding

            print(f"\nUpdated embedding for {name} using the last {len(audio_buffer) // self.sample_rate} seconds of audio")
        except Exception as e:
            print(f"\nError updating embedding for {name}: {e}")

    def recognize(self, buffer):
        """
        Recognize speakers in real-time.

        Args:
            buffer: List of audio samples or bytes buffer

        Returns:
            dict: A dictionary of speaker names mapped to their cosine similarity scores.
        """
        # Recognize speakers
        likely_speakers = {}
        # Convert audio buffer to tensor
        wavs = torch.from_numpy(np.array(buffer))        
        # Generate embedding for the current audio buffer
        embedding = self.model.encode_batch(wavs)
        # Compare the generated embedding to known speakers
        likely_speakers = self.compare_speakers(embedding)

        return likely_speakers

    def delete_embedding(self, name):
        """
        Delete a speaker's embedding.

        Args:
            name (str): The name of the speaker to delete the embedding for.
        """
        if name in self.ref_embeddings:
            del self.ref_embeddings[name]
            print(f"Deleted embedding for {name}")
        else:
            print(f"No embedding found for {name}")

    def compare_speakers(self, embedding):
        """
        Compare the given embedding to the reference embeddings.

        Args:
            embedding (torch.Tensor): The embedding to compare.

        Returns:
            dict: A dictionary of speaker names mapped to their cosine similarity scores.
        """
        likely_speakers = {}
        # Calculate cosine similarity for each reference speaker
        for ref_speaker, ref_embedding in self.ref_embeddings.items():
            likely_speakers[ref_speaker] = abs(torch.nn.functional.cosine_similarity(
                embedding.squeeze(),
                ref_embedding.squeeze(),
                dim=0).item())
        return likely_speakers
    
    def save_embeddings(self):
        """
        Stop the SpeechBrain speaker recognition system.

        This method is called when the system is shut down. It saves the reference
        speaker embeddings to disk using PyTorch's `torch.save()` method.
        """
        # Empty embeddings directory before saving
        for file in glob.glob(f"{EMBEDDINGS_DIR}/*.pt"):
            os.remove(file)

        # Save the reference speaker embeddings to disk
        for ref_speaker, ref_embedding in self.ref_embeddings.items():
            # Create the file path for the embedding
            file_path = f"{EMBEDDINGS_DIR}/{ref_speaker}_embedding.pt"
            # Save the embedding to disk
            torch.save(ref_embedding, file_path)
            print(f"Saved embedding for {ref_speaker} to {file_path}")



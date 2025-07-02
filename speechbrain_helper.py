from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.inference.VAD import VAD
from speechbrain.utils.fetching import LocalStrategy
import glob
import os
import torch
import numpy as np
from scipy.spatial.distance import cdist
from queue import Queue

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
        
        # Load from the specified directory if it exists, otherwise download
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir=model_path,            
            local_strategy=LocalStrategy.COPY_SKIP_CACHE
        )
        
        self.started = False
        self.sample_rate = rate  # Sample rate from audio streamer
        self.ref_embeddings = {}  # Dictionary to store reference speaker embeddings
        self.load_embeddings()  # Load existing speaker embeddings

        print("SpeechBrain model initialized")

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
        # Save the reference speaker embeddings to disk
        for ref_speaker, ref_embedding in self.ref_embeddings.items():
            # Create the file path for the embedding
            file_path = f"{EMBEDDINGS_DIR}/{ref_speaker}_embedding.pt"
            # Save the embedding to disk
            torch.save(ref_embedding, file_path)
            print(f"Saved embedding for {ref_speaker} to {file_path}")



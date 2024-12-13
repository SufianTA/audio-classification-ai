# audio-classification-ai/audio_classification.py

import torch
import torchaudio
from torchaudio.models import wav2vec2

# Load pre-trained Wav2Vec 2.0 model
model = wav2vec2.w2v2_base()
model.eval()

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("path_to_audio.wav")
waveform = waveform.mean(dim=0).unsqueeze(0)  # Convert to mono

# Audio classification
with torch.no_grad():
    features = model(waveform)

print(features)

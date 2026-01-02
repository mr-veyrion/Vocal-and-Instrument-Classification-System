
# Vocal and Instrument Classification System

A comprehensive audio analysis pipeline that separates vocals from instrumental tracks, transcribes lyrics, and identifies musical instruments using state-of-the-art deep learning models.

## Overview

This project implements a complete audio processing workflow that:
- Separates vocals from background music using Demucs
- Transcribes vocal content with OpenAI's Whisper
- Identifies and quantifies musical instruments in the audio
- Visualizes instrument distribution through radar charts

## Features

### 1. Source Separation
- **Model**: Hybrid Transformer Demucs (htdemucs_ft)
- **Capability**: High-quality vocal and instrumental track separation
- **Output**: Isolated vocal and instrumental audio files

### 2. Automatic Speech Recognition (ASR)
- **Model**: Whisper Large-v3
- **Features**:
  - Time-stamped transcription
  - Multi-language support
  - High accuracy on musical vocals

### 3. Instrument Classification
- **Model**: Fine-tuned audio classification transformer
- **Method**: Multi-label classification with chunk-based analysis
- **Output**: Confidence scores for detected instruments

### 4. Visualization
- Radar chart representation of instrument distribution
- Normalized confidence scores for easy interpretation

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Examples](#examples)
- [License](#license)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster processing)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/vocal-instrument-classification.git
cd vocal-instrument-classification
```

### Step 2: Install Dependencies
```bash
pip install demucs
pip install torch torchaudio
pip install torchcodec
pip install -U openai-whisper
pip install transformers
pip install matplotlib numpy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```python
import torch
import whisper
import demucs

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Whisper available:", whisper.__version__)
```

## Quick Start

```python
# 1. Separate vocals from music
!python -m demucs.separate -n htdemucs_ft -d cuda --two-stems vocals your_audio.wav

# 2. Transcribe vocals
import whisper
model = whisper.load_model("large-v3")
result = model.transcribe("separated/htdemucs_ft/your_audio/vocals.wav")

# 3. Identify instruments
from instrument_classifier import classify_instruments
instruments = classify_instruments("separated/htdemucs_ft/your_audio/no_vocals.wav")
print(instruments)
```

## Usage

### 1. Audio Source Separation

Separate vocals from instrumental tracks using Demucs:

```bash
python -m demucs.separate -n htdemucs_ft -d cuda --two-stems vocals /path/to/audio.wav
```

**Parameters:**
- `-n htdemucs_ft`: Specifies the fine-tuned Hybrid Transformer Demucs model
- `-d cuda`: Uses GPU acceleration (use `cpu` if GPU unavailable)
- `--two-stems vocals`: Outputs only vocals and no_vocals stems
- `-o OUTPUT_DIR`: Specify output directory (default: `./separated`)

**Additional Options:**
```bash
# Show all available options
demucs -h

# Use multiple jobs for faster processing
demucs -n htdemucs_ft -j 4 audio.wav

# Output as MP3 instead of WAV
demucs -n htdemucs_ft --mp3 audio.wav

# Process with higher quality (more shifts)
demucs -n htdemucs_ft --shifts 10 audio.wav
```

**Output Structure:**
```
separated/htdemucs_ft/
└── [audio_name]/
    ├── vocals.wav
    └── no_vocals.wav
```

### 2. Vocal Transcription

Transcribe separated vocals with timestamps:

```python
import whisper

# Load model (options: tiny, base, small, medium, large-v3)
model = whisper.load_model("large-v3")

# Transcribe audio
result = model.transcribe("/path/to/vocals.wav")

# Print timestamped transcription
for seg in result["segments"]:
    print(f"[{seg['start']:.2f} → {seg['end']:.2f}] {seg['text']}")

# Access full text
full_text = result["text"]
print("\nFull transcription:", full_text)
```

**Advanced Options:**
```python
# Transcribe with language specification
result = model.transcribe(
    "vocals.wav",
    language="en",  # Specify language
    task="transcribe",  # or "translate" to translate to English
    verbose=True
)

# With temperature fallback for better accuracy
result = model.transcribe(
    "vocals.wav",
    temperature=0.0,  # More deterministic
    beam_size=5,  # Beam search
    best_of=5  # Number of candidates
)
```

**Model Comparison:**

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| tiny | 39M | ~32x | Low | Quick tests |
| base | 74M | ~16x | Medium | Fast processing |
| small | 244M | ~6x | Good | Balanced |
| medium | 769M | ~2x | Very Good | High quality |
| large-v3 | 1550M | ~1x | Best | Production |


## Technical Details

### Audio Processing Pipeline

#### 1. Chunking Strategy
- **Default chunk size**: 3 seconds (48,000 samples at 16kHz)
- **Overlap**: 50% (1.5 seconds)
- **Purpose**: Ensures smooth temporal coverage and captures instrument transitions
- **Trade-offs**: More overlap = better coverage but slower processing

#### 2. Activity Detection
- **Energy threshold**: 1e-4 (configurable)
- **Calculation**: Root Mean Square (RMS) energy
- **Purpose**: Filters silence and extremely low-energy segments
- **Benefits**: Reduces processing time by 20-40%

#### 3. Multi-label Classification
- **Activation**: Sigmoid (not softmax)
- **Reason**: Allows detection of multiple simultaneous instruments
- **Output**: Independent confidence scores per instrument (0-1 range)
- **Threshold**: 0.05 minimum confidence for reporting

### Model Architectures

#### Demucs (Source Separation)
```
Input Audio (44.1kHz stereo)
    ↓
Encoder (U-Net style)
    ↓
Hybrid Processing:
  • Waveform domain (time)
  • Spectrogram domain (frequency)
    ↓
Transformer Layers
    ↓
Decoder
    ↓
Output: Vocals + Instruments
```

**Key Features:**
- Hybrid architecture combining convolutional and transformer layers
- Bag of 4 models ensemble for improved accuracy
- Trained on 800+ hours of music (Musdb, internal datasets)
- 79M parameters per model

#### Whisper (ASR)
```
Input: Mel Spectrogram (80 bins)
    ↓
Encoder: Audio → Embeddings
    ↓
Decoder: Embeddings → Text
    ↓
Output: Transcription + Timestamps
```

**Key Features:**
- Transformer-based encoder-decoder architecture
- 1550M parameters (large-v3)
- Trained on 680,000 hours of multilingual data
- Supports 99 languages

#### Instrument Classifier
```
Input: Audio chunks (16kHz, 3 seconds)
    ↓
Feature Extraction (Wav2Vec2-based)
    ↓
Classification Head (Multi-label)
    ↓
Output: Instrument probabilities
```

**Supported Instruments:**

- Acoustic Guitar
- Bass Guitar
- Drum Set
- Electric Guitar
- Flute
- Hi-Hats
- Keyboard
- Trumpet
- Violin


### Memory Requirements

| Component | GPU Memory | CPU Memory |
|-----------|-----------|-----------|
| Demucs | 4GB | 2GB |
| Whisper large-v3 | 10GB | 6GB |
| Instrument Classifier | 2GB | 1GB |
| **Peak Usage** | 10GB | 8GB |
---
## Examples

### Example 1: Processing a Song

```python
# Input: "Shape of You" by Ed Sheeran
process_audio_file("shape_of_you.wav")

# Output transcription sample:
"""
[12.00 → 17.00] The club isn't the best place to find a lover
[17.00 → 22.00] So the bar is where I go
[22.00 → 27.00] Me and my friends at the table doing shots
"""

# Output instruments:
{
    'guitar': 0.892,
    'drums': 0.765,
    'bass': 0.543,
    'synthesizer': 0.387,
    'piano': 0.234
}
```


## Requirements

```txt
# requirements.txt
torch>=2.0.0
torchaudio>=2.0.0
torchcodec>=0.9.0
demucs>=4.0.0
openai-whisper>=20231117
transformers>=4.30.0
matplotlib>=3.5.0
numpy>=1.21.0
scipy>=1.7.0
soundfile>=0.12.0
librosa>=0.10.0
```

## Citation

If you use this project in your research, please cite:

```bibtex
@software{vocal_instrument_classification_2025,
  title = {Vocal and Instrument Classification System},
  author = {Abhishek Sharma},
  year = {2025},
  url = {https://github.com/mr-veyrion/vocal-instrument-classification},
  note = {Deep learning pipeline for audio source separation, transcription, and classification}
}
```

### Model Citations

**Demucs:**
```bibtex
@article{defossez2019demucs,
  title={Music Source Separation in the Waveform Domain},
  author={Défossez, Alexandre and Usunier, Nicolas and Bottou, Léon and Bach, Francis},
  journal={arXiv preprint arXiv:1911.13254},
  year={2019}
}
```
**Whisper:**
```bibtex
@article{radford2022robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

**Musical Instrument Classification Model:**
```bibtex
@misc{bhaveen2024musical_instruments,
  author = {Bhaveen Patel},
  title = {Musical Instrument Classification},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Bhaveen/Musical-Instrument-Classification}},
  note = {Fine-tuned Wav2Vec2-based model for multi-label instrument classification}
}
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **Demucs**: MIT License
- **Whisper**: MIT License
- **Music Instrument Classification**: MIT License
  
## Acknowledgments

- **Facebook Research** for Demucs source separation models
- **OpenAI** for Whisper speech recognition
- **Bhaveen** (Huggingface): for providing with instrumental classification model. 

- The open-source audio processing community

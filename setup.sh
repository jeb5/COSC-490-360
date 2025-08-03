#!/bin/bash

pip install numpy opencv-python git+https://github.com/google/spatial-media.git pillow line-profiler av matplotlib scipy shapely progressbar2 git+https://github.com/jeb5/ngdsac_horizon.git

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use default PyTorch installation
    echo "Detected macOS, installing PyTorch without CUDA..."
    pip install torch torchvision torchaudio
else
    # Other OS (Linux/Windows) - use CUDA-enabled PyTorch
    echo "Detected non-macOS system, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi

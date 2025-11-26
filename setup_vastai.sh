#!/bin/bash
# ============================================================
# DeepSeek OCR Training Setup Script for Vast.ai (RTX 4090)
# ============================================================
# 
# USAGE: 
#   1. Create a Vast.ai instance with RTX 4090
#   2. Select PyTorch 2.1+ / CUDA 12.1+ template
#   3. Upload your project files
#   4. Run: bash setup_vastai.sh
#
# ============================================================

set -e

echo "=============================================="
echo "🚀 DeepSeek OCR Training Setup for Vast.ai"
echo "=============================================="

# Check GPU
echo ""
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Check Python version
echo ""
echo "🐍 Python version:"
python3 --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip

# Core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ML packages
pip install transformers>=4.40.0 accelerate>=0.27.0 datasets>=2.18.0
pip install peft>=0.10.0 bitsandbytes>=0.43.0

# Image processing
pip install Pillow opencv-python

# Utilities
pip install tqdm numpy scipy scikit-learn
pip install wandb tensorboard

# Install Flash Attention 2 (optional, but highly recommended)
echo ""
echo "⚡ Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention installation failed, will use default attention"

# Install Jupyter
pip install jupyter ipywidgets

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
python3 -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo ""
echo "To start training, run:"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser"
echo ""
echo "Or run the notebook directly:"
echo "  jupyter nbconvert --to script train_deepseek_ocr.ipynb"
echo "  python train_deepseek_ocr.py"
echo "=============================================="

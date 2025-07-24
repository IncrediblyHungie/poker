#!/bin/bash

echo "🚀 Starting setup for GPU-ready Pluribus Poker Bot..."

# Clone your GPU branch if not already present
if [ ! -d "poker" ]; then
    echo "📥 Cloning repository..."
    git clone -b gpu https://github.com/IncrediblyHungie/poker.git
fi

cd poker

# Set up virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core requirements (excluding open_spiel)
echo "📦 Installing Python requirements..."
grep -v open_spiel requirements.txt > temp_requirements.txt
pip install -r temp_requirements.txt
rm temp_requirements.txt

# 🔧 Force install known fixes
echo "🔧 Fixing NumPy compatibility..."
pip install numpy==1.26.4

echo "🧱 Installing missing CLI helper (fire)..."
pip install fire

# Confirm GPU availability
echo "🧠 Checking GPU:"
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'Using: {torch.cuda.get_device_name(0)}')"

# Show live GPU usage
nvidia-smi

# Launch pipeline
echo "🏁 Starting full training pipeline (CFR → ValueNet → Distill)..."
python launch.py pipeline

echo "✅ All done!"

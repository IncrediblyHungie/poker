#!/bin/bash

echo "🚀 Starting setup for GPU-ready Pluribus Poker Bot..."

# Clone your GPU branch
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

# Install dependencies, skipping open_spiel for now
echo "📦 Installing Python requirements..."
# Remove open_spiel temporarily if needed
grep -v open_spiel requirements.txt > temp_requirements.txt
pip install -r temp_requirements.txt
rm temp_requirements.txt

# Confirm CUDA availability
echo "🧠 Checking GPU:"
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'Using: {torch.cuda.get_device_name(0)}')"

# Optional: show GPU stats
nvidia-smi

# Run full training pipeline
echo "🏁 Starting full training pipeline (CFR → ValueNet → Distill)..."
python launch.py pipeline

echo "✅ All done!"

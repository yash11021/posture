#!/bin/bash

# Posture Monitor Setup Script
# This script sets up the project environment with Python 3.11

echo "🧘 Posture Monitor - Environment Setup"
echo "========================================"
echo ""

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv is not installed"
    echo ""
    echo "Installing pyenv..."
    brew install pyenv
    
    echo ""
    echo "⚠️  Please add these lines to your ~/.zshrc (or ~/.bash_profile):"
    echo ""
    echo 'export PYENV_ROOT="$HOME/.pyenv"'
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"'
    echo 'eval "$(pyenv init -)"'
    echo ""
    echo "Then run: source ~/.zshrc"
    echo "And run this script again."
    exit 1
fi

echo "✓ pyenv is installed"
echo ""

# Check if Python 3.11.9 is installed
if ! pyenv versions | grep -q "3.11.9"; then
    echo "📥 Installing Python 3.11.9 (this may take a few minutes)..."
    pyenv install 3.11.9
    echo "✓ Python 3.11.9 installed"
else
    echo "✓ Python 3.11.9 already installed"
fi

echo ""

# Set local Python version
echo "🔧 Setting Python 3.11.9 for this project..."
pyenv local 3.11.9

echo "✓ Python version set (check .python-version file)"
echo ""

# Check current Python version
PYTHON_VERSION=$(python --version)
echo "Current Python version: $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

echo "✓ pip upgraded"
echo ""

# Install requirements
echo "📦 Installing dependencies..."
echo "   - opencv-python"
echo "   - mediapipe"
echo "   - torch"
echo "   - numpy"
echo ""

pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "========================================"
echo "Next steps:"
echo "========================================"
echo ""
echo "1. Activate the virtual environment (if not already):"
echo "   source venv/bin/activate"
echo ""
echo "2. Test with MediaPipe:"
echo "   python monitor.py --test"
echo ""
echo "3. Calibrate your model:"
echo "   python calibrate.py"
echo ""
echo "4. Run the monitor:"
echo "   python monitor.py"
echo ""
echo "========================================"
echo ""
echo "💡 When you're done, deactivate with: deactivate"
echo ""
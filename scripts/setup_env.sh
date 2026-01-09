#!/bin/bash
# Setup Python virtual environment and install vLLM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Creating virtual environment..."
cd "$PROJECT_DIR"
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing vLLM and dependencies..."
pip install vllm

echo "Installing additional utilities..."
pip install huggingface_hub requests

echo "Installing ModelScope (optional download source)..."
pip install modelscope

echo "Installing TUI dependencies..."
pip install textual

echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"

#!/usr/bin/env bash
# Research Agent - Setup Script
# Run this once to set up the development environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Research Agent Setup ==="
echo "Project root: $PROJECT_ROOT"

# Check if Python 3.12 is available
echo ""
echo "Checking for Python 3.12..."

PYTHON_CMD=""

# Check python3.12 first
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    echo "Found: python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    if [[ "$PYTHON_MINOR" -ge 12 ]]; then
        PYTHON_CMD="python3"
        echo "Found: python3 ($PYTHON_VERSION)"
    fi
fi

if [[ -z "$PYTHON_CMD" ]]; then
    echo ""
    echo "ERROR: Python 3.12+ is required but not found."
    echo ""
    echo "To install Python 3.12 on Ubuntu/Pop!_OS:"
    echo ""
    echo "  sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "  sudo apt update"
    echo "  sudo apt install python3.12 python3.12-venv python3.12-dev"
    echo ""
    echo "After installing, run this script again."
    exit 1
fi

echo "Using: $PYTHON_CMD"
$PYTHON_CMD --version

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up virtual environment..."
VENV_DIR="$PROJECT_ROOT/.venv"

if [[ -d "$VENV_DIR" ]]; then
    echo "Removing old virtual environment..."
    rm -rf "$VENV_DIR"
fi

$PYTHON_CMD -m venv "$VENV_DIR"
echo "Created virtual environment at .venv"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip using python3 -m pip (most reliable method)
echo ""
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from pyproject.toml..."
python3 -m pip install -e "$PROJECT_ROOT[dev]"

# Create .env from .env.example if it doesn't exist
echo ""
echo "Checking environment configuration..."
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo "Created .env from .env.example"
        echo "IMPORTANT: Edit .env and add your API keys before running the application."
    else
        echo "WARNING: .env.example not found. Create .env manually."
    fi
else
    echo ".env already exists"
fi

# Create required directories
echo ""
echo "Creating required directories..."
mkdir -p "$PROJECT_ROOT/outputs"
mkdir -p "$PROJECT_ROOT/storage"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -m pip list | grep -E "fastapi|langgraph|pydantic" || true

# Summary
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Installed versions:"
python3 -m pip show fastapi | grep Version || true
echo ""
echo "To use the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Quick commands:"
echo "  Run tests:    python3 -m pytest tests/ -v"
echo "  Start API:    python3 -m uvicorn src.main:app --reload"
echo "  Lint code:    python3 -m ruff check src/"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (OPENAI_API_KEY, TAVILY_API_KEY, etc.)"
echo "  2. source .venv/bin/activate"
echo "  3. python3 -m pytest tests/ -v"
echo ""

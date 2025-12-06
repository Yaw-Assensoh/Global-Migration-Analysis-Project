#!/bin/bash
# setup.sh - Run before pip install on Streamlit Cloud

echo "🚀 Starting installation setup..."

# Update pip
pip install --upgrade pip

# Install system dependencies
apt-get update
apt-get install -y build-essential python3-dev

echo "✅ Setup complete. Installing Python packages..."

#!/bin/bash

# Setup script for TEACh environment integration
# Usage: bash setup_teach.sh

# Source conda
source $(conda info --base)/etc/profile.d/conda.sh

# Clone the TEACh repository
TEACH_REPO_URL="https://github.com/alexa/teach.git"
TEACH_DIR="teach"

if [ ! -d "$TEACH_DIR" ]; then
    echo "Cloning TEACh repository from $TEACH_REPO_URL into $TEACH_DIR..."
    git clone "$TEACH_REPO_URL" "$TEACH_DIR"
else
    echo "TEACh repository already exists at $TEACH_DIR. Pulling latest changes..."
    cd "$TEACH_DIR"
    git pull
    cd ..
fi

echo "Creating 'embench_teach' environment..."
# Check if environment already exists
if conda info --envs | grep -q "embench_teach"; then
    echo "Environment 'embench_teach' already exists. Skipping creation."
else
    conda create -n embench_teach python=3.8 -y
fi

echo "Activating 'embench_teach'..."
conda activate embench_teach

echo "Installing dependencies..."
# Navigate to cloned repo to install requirements
cd "$TEACH_DIR"

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in $TEACH_DIR"
fi

# Install the teach package in editable mode
pip install -e .

echo "Downloading TEACh dataset (optional, can be skipped with CTRL+C)..."
# teach_download is a command provided by the teach package
echo "Run 'teach_download' manually if needed, or uncomment the line below."
# teach_download

cd ..

echo "TEACh environment setup complete."
echo "To verify: conda activate embench_teach && python -c 'import teach; print(teach.__file__)'"

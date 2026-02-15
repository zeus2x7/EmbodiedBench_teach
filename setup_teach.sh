#!/bin/bash

# Setup script for TEACh environment integration
# Usage: bash setup_teach.sh

# Source conda
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
else
    echo "Error: conda not found."
    exit 1
fi

# Set the TEACH directory (using existing integrated folder)
TEACH_DIR="teach_integrate/teach"
REPO_URL="https://github.com/alexa/teach.git"

if [ ! -d "$TEACH_DIR" ]; then
    echo "Cloning TEACh repository from $REPO_URL into $TEACH_DIR..."
    git clone "$REPO_URL" "$TEACH_DIR"
else
    echo "TEACh repository already exists at $TEACH_DIR. Pulling latest changes..."
    cd "$TEACH_DIR"
    git pull
    cd - > /dev/null
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

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export DISPLAY=:1
echo "Exported PYTHONPATH and DISPLAY."

echo "Downloading TEACh dataset..."
# Download to data directory inside TEACH_DIR
DATA_DIR="$(pwd)/data"
mkdir -p "$DATA_DIR"

# Check if data already exists to avoid re-downloading
if [ -d "$DATA_DIR/games" ]; then
    echo "Dataset seems to be present in $DATA_DIR. Skipping download."
else
    echo "Downloading dataset to $DATA_DIR"
    # teach_download is a command provided by the teach package
    # Using -d to specify directory
    teach_download -d "$DATA_DIR"
fi

cd - > /dev/null

echo "TEACh environment setup complete."
echo "Data downloaded to $TEACH_DIR/data"
echo "You can run the random agent with:"
echo "python teach_random_agent.py --data_dir $TEACH_DIR/data"

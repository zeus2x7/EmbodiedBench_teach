#!/bin/bash

# Sequentially runs all 5 EmbodiedBench environments
# (EB-ALFRED, EB-Habitat, EB-Navigation, EB-Manipulation, EB-TEACh)
# Usage: ./run_logger.sh

# Source conda
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
else
    echo "Error: conda not found."
    exit 1
fi

# List of environments to run
ENVS=("EB-ALFRED" "EB-Habitat" "EB-Navigation" "EB-Manipulation" "EB-TEACh")

for ENV_NAME in "${ENVS[@]}"; do
    echo "=========================================="
    echo "Starting process for: $ENV_NAME"
    echo "=========================================="

    if [ "$ENV_NAME" == "EB-ALFRED" ]; then
        echo "Activating 'embench' for EB-ALFRED..."
        conda activate embench
    elif [ "$ENV_NAME" == "EB-Habitat" ]; then
        echo "Activating 'embench' for EB-Habitat..."
        conda activate embench
    elif [ "$ENV_NAME" == "EB-Navigation" ]; then
        echo "Activating 'embench_nav' for EB-Navigation..."
        conda activate embench_nav
    elif [ "$ENV_NAME" == "EB-TEACh" ]; then
        echo "Activating 'embench_teach' for EB-TEACh..."
        conda activate embench_teach
        # TEACh requires DISPLAY for ai2thor and PYTHONPATH for teach modules
        export DISPLAY=:1
        export PYTHONPATH="${PYTHONPATH}:$(pwd)/teach_integrate/teach/src"
    elif [ "$ENV_NAME" == "EB-Manipulation" ]; then
        echo "Activating 'embench_man' for EB-Manipulation..."
        conda activate embench_man
        # Set required env vars for Manipulation
        export EMBODIED_BENCH_ROOT=$(pwd)
        export COPPELIASIM_ROOT="$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04"
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
        export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
    fi

    echo "Running random_agent_logger.py for $ENV_NAME..."
    python random_agent_logger.py --env "$ENV_NAME"

    echo "Completed $ENV_NAME"
    echo ""
done

echo "All environments completed."

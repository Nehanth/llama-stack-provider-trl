#!/bin/bash

# Script to run the TRL Remote FSDP Provider Server
# This server provides distributed DPO training capabilities via HTTP API

echo "Starting TRL Remote FSDP Provider Server..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if required packages are installed
python -c "import torch, trl, transformers" 2>/dev/null || {
    echo "Error: Required packages not found. Please run ./scripts/prepare-env.sh first"
    exit 1
}

# Check for CUDA availability (optional, can run on CPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "CUDA detected - distributed training will use GPUs"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}  # Default to first 2 GPUs
    export CUDA_VISIBLE_DEVICES
    echo "Using CUDA devices: $CUDA_VISIBLE_DEVICES"
else
    echo "Warning: CUDA not available - distributed training will be limited"
fi

# Set environment variables for distributed training
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Default configuration
HOST=${HOST:-"localhost"}
PORT=${PORT:-8322}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

echo "Server configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT" 
echo "  Log Level: $LOG_LEVEL"

# Create log directory
mkdir -p /tmp/trl_remote_logs

# Start the remote server
echo "Starting remote FSDP provider server..."
python -m llama_stack_provider_trl_remote.server \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" 
#!/bin/bash
# Parallel Training Startup Script for WSL2
# Run this in WSL2 Ubuntu terminal

set -e

echo "=========================================="
echo "PARALLEL TRAINING SETUP"
echo "=========================================="

# Navigate to project directory
cd /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness

# Check GPUs are visible
echo ""
echo "Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Check Python environment
echo ""
echo "Python environment:"
which python3
python3 --version

# Check if unsloth is available
echo ""
echo "Checking unsloth installation..."
python3 -c "import unsloth; print(f'Unsloth version: {unsloth.__version__}')" || echo "WARNING: unsloth not found!"

# Start training on GPU 0
echo ""
echo "=========================================="
echo "Starting LLM Training on GPU 0 (RTX 5090)"
echo "=========================================="
cd NanoGPT

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=12 \
    nohup python3 train_100k_production.py > training_100k.log 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Log file: NanoGPT/training_100k.log"

# Wait a moment for training to initialize
echo ""
echo "Waiting 10 seconds for training to initialize..."
sleep 10

# Check if process is still running
if ps -p $TRAIN_PID > /dev/null; then
    echo "✓ Training process is running!"
    echo ""
    echo "Monitor with: tail -f /mnt/c/Users/akbon/OneDrive/Documents/GitHub/harmonic-field-consciousness/NanoGPT/training_100k.log"
    echo "Check GPU usage: watch -n 5 nvidia-smi"
    echo "Kill training: kill $TRAIN_PID"
else
    echo "✗ Training process failed to start. Check the log:"
    tail -30 training_100k.log
    exit 1
fi

# Show initial log output
echo ""
echo "=========================================="
echo "Initial Training Log (last 20 lines):"
echo "=========================================="
tail -20 training_100k.log

echo ""
echo "=========================================="
echo "Training is running in background!"
echo "Expected duration: ~15 hours"
echo "=========================================="

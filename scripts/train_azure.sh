#!/bin/bash
# =============================================================================
# D4RT Azure Training Launch Script
# =============================================================================
#
# This script launches D4RT training on Azure VMs with proper environment setup.
#
# Usage:
#   # Single GPU
#   ./scripts/train_azure.sh single
#
#   # Multi-GPU (single node)
#   ./scripts/train_azure.sh multi
#
#   # Multi-node distributed
#   ./scripts/train_azure.sh distributed
#
# Environment variables (set before running):
#   DATA_ROOT: Path to training data (default: ~/d4rt_data/kubric)
#   OUTPUT_DIR: Path for checkpoints (default: ~/d4rt_outputs)
#   CONFIG: Config file (default: configs/d4rt_rtx5090.yaml)
#
# For multi-node training, also set:
#   MASTER_ADDR: Master node address
#   MASTER_PORT: Master node port (default: 29500)
#   NODE_RANK: This node's rank (0 for master)
#   NNODES: Total number of nodes
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
MODE="${1:-single}"
DATA_ROOT="${DATA_ROOT:-$HOME/d4rt_data/kubric}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/d4rt_outputs}"
CONFIG="${CONFIG:-configs/d4rt_rtx5090.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "=============================================="
echo "D4RT Azure Training"
echo "=============================================="
echo "Mode: $MODE"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Config: $CONFIG"
echo "GPUs detected: $NUM_GPUS"
echo "=============================================="

# =============================================================================
# Environment Setup
# =============================================================================

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate d4rt
fi

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# PyTorch optimizations
export OMP_NUM_THREADS=8
export TORCH_NCCL_BLOCKING_WAIT=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Training Launch
# =============================================================================

cd "$(dirname "$0")/.."

case $MODE in
    "single")
        # Single GPU training
        echo "Starting single GPU training..."
        python train.py \
            --config "$CONFIG" \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            --auto-resume \
            2>&1 | tee "$OUTPUT_DIR/train.log"
        ;;

    "multi")
        # Multi-GPU training on single node
        echo "Starting multi-GPU training on $NUM_GPUS GPUs..."
        torchrun \
            --standalone \
            --nproc_per_node=$NUM_GPUS \
            train.py \
            --config "$CONFIG" \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            --auto-resume \
            2>&1 | tee "$OUTPUT_DIR/train.log"
        ;;

    "distributed")
        # Multi-node distributed training
        if [ -z "$MASTER_ADDR" ]; then
            echo "ERROR: MASTER_ADDR must be set for distributed training"
            exit 1
        fi
        if [ -z "$NODE_RANK" ]; then
            echo "ERROR: NODE_RANK must be set for distributed training"
            exit 1
        fi
        if [ -z "$NNODES" ]; then
            echo "ERROR: NNODES must be set for distributed training"
            exit 1
        fi

        echo "Starting distributed training..."
        echo "  Master: $MASTER_ADDR:$MASTER_PORT"
        echo "  Node rank: $NODE_RANK / $NNODES"
        echo "  GPUs per node: $NUM_GPUS"
        echo "  Total GPUs: $((NNODES * NUM_GPUS))"

        torchrun \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --nproc_per_node=$NUM_GPUS \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            train.py \
            --config "$CONFIG" \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            --auto-resume \
            2>&1 | tee "$OUTPUT_DIR/train_node${NODE_RANK}.log"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [single|multi|distributed]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Training complete!"
echo "Outputs saved to: $OUTPUT_DIR"
echo "=============================================="

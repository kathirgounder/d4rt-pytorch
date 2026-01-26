#!/bin/bash
# =============================================================================
# D4RT Multi-Node Training Launcher for Azure
# =============================================================================
#
# This script helps launch distributed training across multiple Azure VMs.
# Run this on ALL nodes (master and workers).
#
# Prerequisites:
#   1. All nodes must have the same setup (code, data, environment)
#   2. Nodes must be able to communicate over the network
#   3. SSH keys set up between nodes (for monitoring)
#
# Usage:
#   # On master node (node 0):
#   MASTER_ADDR=10.0.0.4 NODE_RANK=0 NNODES=4 ./scripts/azure_multinode_launch.sh
#
#   # On worker node 1:
#   MASTER_ADDR=10.0.0.4 NODE_RANK=1 NNODES=4 ./scripts/azure_multinode_launch.sh
#
#   # On worker node 2:
#   MASTER_ADDR=10.0.0.4 NODE_RANK=2 NNODES=4 ./scripts/azure_multinode_launch.sh
#
#   # On worker node 3:
#   MASTER_ADDR=10.0.0.4 NODE_RANK=3 NNODES=4 ./scripts/azure_multinode_launch.sh
# =============================================================================

set -e

# Required environment variables
if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: MASTER_ADDR must be set (e.g., 10.0.0.4 or hostname)"
    exit 1
fi

if [ -z "$NODE_RANK" ]; then
    echo "ERROR: NODE_RANK must be set (0 for master, 1+ for workers)"
    exit 1
fi

if [ -z "$NNODES" ]; then
    echo "ERROR: NNODES must be set (total number of nodes)"
    exit 1
fi

# Optional configuration
MASTER_PORT="${MASTER_PORT:-29500}"
DATA_ROOT="${DATA_ROOT:-$HOME/d4rt_data/kubric}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/d4rt_outputs}"
CONFIG="${CONFIG:-configs/d4rt_azure_a100.yaml}"

# Detect GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
TOTAL_GPUS=$((NNODES * NUM_GPUS))

echo "=============================================="
echo "D4RT Multi-Node Distributed Training"
echo "=============================================="
echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo "Node rank: $NODE_RANK"
echo "Total nodes: $NNODES"
echo "GPUs per node: $NUM_GPUS"
echo "Total GPUs: $TOTAL_GPUS"
echo "Config: $CONFIG"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

# Activate environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate d4rt
fi

# NCCL settings for Azure InfiniBand
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=eth0

# For Azure NC-series with NVLink
export NCCL_P2P_LEVEL=NVL

# PyTorch settings
export OMP_NUM_THREADS=8
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Create output directory
mkdir -p "$OUTPUT_DIR"

cd "$(dirname "$0")/.."

# Calculate gradient accumulation based on total GPUs
# Target effective batch size = 64 (paper)
# If using 8 GPUs with batch_size=2: accum = 64 / (2 * 8) = 4
# If using 32 GPUs with batch_size=2: accum = 64 / (2 * 32) = 1

echo ""
echo "Starting training on node $NODE_RANK..."
echo ""

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

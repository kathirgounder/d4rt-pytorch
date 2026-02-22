#!/bin/bash
#SBATCH --job-name=d4rt-train
#SBATCH --output=logs/d4rt_%j.out
#SBATCH --error=logs/d4rt_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
# Uncomment and set your PACE account:
# #SBATCH -A your_account_here

# =============================================================================
# D4RT Training on PACE (Georgia Tech)
# =============================================================================
#
# Setup (run once, interactively):
#   module load anaconda3 cuda/12.1
#   conda create -n d4rt python=3.10 -y
#   conda activate d4rt
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   pip install timm transformers pillow pyyaml tensorboard tqdm
#
# Download PointOdyssey sample (run on login/data-transfer node):
#   cd $SCRATCH && mkdir -p d4rt_data/pointodyssey
#   pip install huggingface_hub
#   huggingface-cli download aharley/pointodyssey --repo-type dataset \
#     --include "sample.tar.gz" --local-dir $SCRATCH/d4rt_data/pointodyssey
#   cd $SCRATCH/d4rt_data/pointodyssey && tar -xzf sample.tar.gz
#
# Submit:
#   mkdir -p logs
#   sbatch scripts/train_pace.sh
#
# =============================================================================

set -e

# Load modules
module load anaconda3
module load cuda/12.1
conda activate d4rt

# Directories
DATA_ROOT="${DATA_ROOT:-$SCRATCH/d4rt_data/pointodyssey/sample}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/d4rt_outputs/$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/d4rt_test.yaml}"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "D4RT Training on PACE"
echo "============================================"
echo "Config: $CONFIG"
echo "Data: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

# Single GPU training
python train.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --dataset kubric \
    --output-dir "$OUTPUT_DIR" \
    --amp \
    --gradient-checkpointing \
    --auto-resume \
    --num-workers 16

echo "Training complete! Output: $OUTPUT_DIR"

# =============================================================================
# Multi-GPU variant (uncomment and change SBATCH --gres=gpu:A100:4):
# torchrun --nproc_per_node=4 train.py \
#     --config configs/d4rt_base.yaml \
#     --data-root "$DATA_ROOT" \
#     --dataset kubric \
#     --output-dir "$OUTPUT_DIR" \
#     --amp \
#     --gradient-checkpointing \
#     --auto-resume
# =============================================================================

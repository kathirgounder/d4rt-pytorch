#!/bin/bash
# =============================================================================
# D4RT Azure VM Setup Script
# =============================================================================
# This script sets up an Azure VM for D4RT training with RTX 5090 or other GPUs
#
# Recommended Azure VM:
#   - Standard_NC24ads_A100_v4 (A100 80GB) for paper-like training
#   - Standard_ND96asr_v4 (8x A100) for multi-GPU
#   - For RTX 5090: Use a custom VM or on-prem setup
#
# Usage:
#   chmod +x scripts/setup_azure_vm.sh
#   ./scripts/setup_azure_vm.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "D4RT Azure VM Setup"
echo "=============================================="

# -----------------------------------------------------------------------------
# 1. System Updates
# -----------------------------------------------------------------------------
echo "[1/8] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# -----------------------------------------------------------------------------
# 2. Install NVIDIA Drivers (if not already installed)
# -----------------------------------------------------------------------------
echo "[2/8] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y linux-headers-$(uname -r)

    # Add NVIDIA package repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-driver-535  # Adjust version as needed

    echo "NVIDIA drivers installed. Please reboot and re-run this script."
    exit 0
else
    echo "NVIDIA drivers already installed:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi

# -----------------------------------------------------------------------------
# 3. Install CUDA Toolkit
# -----------------------------------------------------------------------------
echo "[3/8] Installing CUDA Toolkit..."
if ! command -v nvcc &> /dev/null; then
    # CUDA 12.1 (compatible with PyTorch 2.x)
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
    sudo sh cuda_12.1.0_530.30.02_linux.run --toolkit --silent

    # Add to PATH
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc

    rm cuda_12.1.0_530.30.02_linux.run
else
    echo "CUDA already installed: $(nvcc --version | grep release)"
fi

# -----------------------------------------------------------------------------
# 4. Install Miniconda
# -----------------------------------------------------------------------------
echo "[4/8] Setting up Python environment..."
if ! command -v conda &> /dev/null; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh

    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    source ~/.bashrc
else
    echo "Conda already installed"
fi

# Create D4RT environment
echo "Creating D4RT conda environment..."
conda create -n d4rt python=3.10 -y
conda activate d4rt

# -----------------------------------------------------------------------------
# 5. Install PyTorch with CUDA
# -----------------------------------------------------------------------------
echo "[5/8] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# -----------------------------------------------------------------------------
# 6. Install D4RT Dependencies
# -----------------------------------------------------------------------------
echo "[6/8] Installing D4RT dependencies..."

# Clone/update D4RT repository
if [ ! -d "d4rt" ]; then
    git clone https://github.com/jiangyurong609/d4rt-pytorch.git d4rt
fi
cd d4rt

pip install -r requirements.txt

# Install optional dependencies
pip install tensorflow-datasets  # For Kubric
pip install tensorboard  # For logging

# Install Flash Attention (optional, for faster training)
# pip install flash-attn --no-build-isolation

# -----------------------------------------------------------------------------
# 7. Install Azure CLI and azcopy
# -----------------------------------------------------------------------------
echo "[7/8] Installing Azure tools..."

# Azure CLI
if ! command -v az &> /dev/null; then
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
fi

# azcopy
if ! command -v azcopy &> /dev/null; then
    wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
    tar -xvf azcopy.tar.gz
    sudo mv azcopy_linux_amd64_*/azcopy /usr/local/bin/
    rm -rf azcopy.tar.gz azcopy_linux_amd64_*
fi

# -----------------------------------------------------------------------------
# 8. Download Training Data
# -----------------------------------------------------------------------------
echo "[8/8] Setting up training data..."

DATA_DIR="$HOME/d4rt_data"
mkdir -p $DATA_DIR

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Configure Azure credentials:"
echo "   export AZURE_STORAGE_ACCOUNT='your_account'"
echo "   export AZURE_STORAGE_KEY='your_key'"
echo ""
echo "2. Download training data from Azure:"
echo "   azcopy copy 'https://\${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/d4rt-training-data/*' $DATA_DIR/ --recursive"
echo ""
echo "3. Or prepare data locally:"
echo "   python scripts/prepare_data_azure.py --dataset kubric --local-dir $DATA_DIR"
echo ""
echo "4. Start training:"
echo "   conda activate d4rt"
echo "   python train.py --config configs/d4rt_rtx5090.yaml --data-root $DATA_DIR/kubric"
echo ""
echo "5. Monitor training:"
echo "   tensorboard --logdir outputs/"
echo ""
echo "=============================================="

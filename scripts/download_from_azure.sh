#!/bin/bash
# =============================================================================
# Download D4RT Training Data from Azure Blob Storage
# =============================================================================
#
# Prerequisites:
#   - azcopy installed
#   - Azure storage credentials set:
#     export AZURE_STORAGE_ACCOUNT="your_account"
#     export AZURE_STORAGE_KEY="your_key"  # or use SAS token
#
# Usage:
#   ./scripts/download_from_azure.sh [dataset] [local_dir]
#
# Examples:
#   ./scripts/download_from_azure.sh kubric ~/d4rt_data
#   ./scripts/download_from_azure.sh all ~/d4rt_data
# =============================================================================

set -e

# Configuration
CONTAINER="d4rt-training-data"
DATASET="${1:-kubric}"
LOCAL_DIR="${2:-$HOME/d4rt_data}"

# Check prerequisites
if ! command -v azcopy &> /dev/null; then
    echo "ERROR: azcopy not found. Install it first:"
    echo "  Linux: wget https://aka.ms/downloadazcopy-v10-linux && tar -xvf downloadazcopy-v10-linux"
    echo "  macOS: brew install azcopy"
    exit 1
fi

if [ -z "$AZURE_STORAGE_ACCOUNT" ]; then
    echo "ERROR: AZURE_STORAGE_ACCOUNT not set"
    echo "  export AZURE_STORAGE_ACCOUNT='your_account'"
    exit 1
fi

if [ -z "$AZURE_STORAGE_KEY" ] && [ -z "$AZURE_STORAGE_SAS_TOKEN" ]; then
    echo "ERROR: Azure credentials not set"
    echo "  export AZURE_STORAGE_KEY='your_key'"
    echo "  OR"
    echo "  export AZURE_STORAGE_SAS_TOKEN='your_sas_token'"
    exit 1
fi

# Build Azure URL
if [ -n "$AZURE_STORAGE_SAS_TOKEN" ]; then
    BASE_URL="https://${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}?${AZURE_STORAGE_SAS_TOKEN}"
else
    BASE_URL="https://${AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}"
fi

# Create local directory
mkdir -p "$LOCAL_DIR"

echo "=============================================="
echo "D4RT Data Download"
echo "=============================================="
echo "Storage Account: $AZURE_STORAGE_ACCOUNT"
echo "Container: $CONTAINER"
echo "Dataset: $DATASET"
echo "Local Directory: $LOCAL_DIR"
echo "=============================================="

download_dataset() {
    local name=$1
    local blob_prefix=$2

    echo ""
    echo "Downloading $name..."

    local source_url="${BASE_URL}/${blob_prefix}"
    local dest_path="${LOCAL_DIR}/${name}"

    mkdir -p "$dest_path"

    azcopy copy "$source_url/*" "$dest_path/" --recursive

    echo "  ✓ $name downloaded to $dest_path"
}

case $DATASET in
    "kubric")
        download_dataset "kubric" "kubric/movi_f"
        ;;
    "pointodyssey")
        download_dataset "pointodyssey" "pointodyssey"
        ;;
    "sintel")
        download_dataset "sintel" "sintel"
        ;;
    "tartanair")
        download_dataset "tartanair" "tartanair"
        ;;
    "scannet")
        download_dataset "scannet" "scannet"
        ;;
    "all")
        download_dataset "kubric" "kubric/movi_f"
        download_dataset "pointodyssey" "pointodyssey"
        download_dataset "sintel" "sintel"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Available: kubric, pointodyssey, sintel, tartanair, scannet, all"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Download Complete!"
echo "=============================================="
echo ""
echo "Start training with:"
echo "  python train.py --config configs/d4rt_rtx5090.yaml --data-root $LOCAL_DIR/$DATASET"
echo ""

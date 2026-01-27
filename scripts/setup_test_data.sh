#!/bin/bash
# Setup minimal test data for D4RT training
# Downloads PointOdyssey sample (3.1GB) - smallest available dataset

set -e

DATA_DIR="${1:-./data/pointodyssey}"
mkdir -p "$DATA_DIR"

echo "=================================================="
echo "D4RT Test Data Setup"
echo "=================================================="
echo "Data directory: $DATA_DIR"
echo ""

# Check for required tools
check_tool() {
    if command -v "$1" &> /dev/null; then
        echo "[OK] $1 found"
        return 0
    else
        echo "[MISSING] $1 not found"
        return 1
    fi
}

echo "Checking required tools..."
HAS_CURL=$(check_tool curl && echo 1 || echo 0)
HAS_WGET=$(check_tool wget && echo 1 || echo 0)
HAS_HF_CLI=$(check_tool huggingface-cli && echo 1 || echo 0)

echo ""

# Download PointOdyssey sample (3.1GB)
SAMPLE_FILE="$DATA_DIR/sample.tar.gz"
SAMPLE_URL="https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/sample.tar.gz"

if [ -f "$SAMPLE_FILE" ]; then
    echo "Sample archive already exists: $SAMPLE_FILE"
elif [ -d "$DATA_DIR/sample" ]; then
    echo "Sample data already extracted: $DATA_DIR/sample"
else
    echo "Downloading PointOdyssey sample (3.1GB)..."
    echo "This may take a few minutes depending on your connection."
    echo ""

    # Try different download methods
    if [ "$HAS_CURL" = "1" ]; then
        echo "Using curl..."
        curl -L -o "$SAMPLE_FILE" --progress-bar "$SAMPLE_URL"
    elif [ "$HAS_WGET" = "1" ]; then
        echo "Using wget..."
        wget -O "$SAMPLE_FILE" --progress=bar:force "$SAMPLE_URL"
    elif [ "$HAS_HF_CLI" = "1" ]; then
        echo "Using huggingface-cli..."
        huggingface-cli download aharley/pointodyssey sample.tar.gz \
            --repo-type dataset \
            --local-dir "$DATA_DIR"
    else
        echo "ERROR: No download tool available!"
        echo "Please install curl, wget, or huggingface-cli:"
        echo "  pip install huggingface_hub"
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "Download failed!"
        exit 1
    fi
fi

# Extract if needed
if [ -f "$SAMPLE_FILE" ] && [ ! -d "$DATA_DIR/sample" ]; then
    echo ""
    echo "Extracting sample data..."
    cd "$DATA_DIR"
    tar -xzf sample.tar.gz
    echo "Extraction complete."
fi

# Verify the data
echo ""
echo "=================================================="
echo "Verifying data structure..."
echo "=================================================="

SAMPLE_DIR="$DATA_DIR/sample"
if [ -d "$SAMPLE_DIR" ]; then
    NUM_SEQUENCES=$(find "$SAMPLE_DIR" -maxdepth 1 -type d | wc -l)
    NUM_SEQUENCES=$((NUM_SEQUENCES - 1))  # Subtract the parent directory
    echo "Found $NUM_SEQUENCES sequences in $SAMPLE_DIR"

    # Check first sequence
    FIRST_SEQ=$(find "$SAMPLE_DIR" -maxdepth 1 -type d | head -2 | tail -1)
    if [ -d "$FIRST_SEQ" ]; then
        echo ""
        echo "First sequence: $(basename $FIRST_SEQ)"
        [ -d "$FIRST_SEQ/rgbs" ] && echo "  [OK] rgbs/ directory found"
        [ -d "$FIRST_SEQ/depths" ] && echo "  [OK] depths/ directory found"
        [ -f "$FIRST_SEQ/anno.npz" ] && echo "  [OK] anno.npz found"
        [ -f "$FIRST_SEQ/intrinsics.npy" ] && echo "  [OK] intrinsics.npy found"
    fi
else
    echo "WARNING: Sample directory not found at $SAMPLE_DIR"
    echo "You may need to manually extract the archive."
fi

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "To test training, run:"
echo ""
echo "  python train.py \\"
echo "    --config configs/d4rt_test.yaml \\"
echo "    --data-root $DATA_DIR/sample \\"
echo "    --output-dir outputs/test_run"
echo ""
echo "For a more thorough test with default settings:"
echo ""
echo "  python train.py \\"
echo "    --config configs/d4rt_base.yaml \\"
echo "    --data-root $DATA_DIR/sample \\"
echo "    --num-frames 16 \\"
echo "    --img-size 128 \\"
echo "    --num-queries 1024 \\"
echo "    --steps 1000"
echo ""

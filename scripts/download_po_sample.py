"""Download PointOdyssey sample (3.1GB) using Python."""
import os
import sys
import tarfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pointodyssey")
SAMPLE_URL = "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/sample.tar.gz"
ARCHIVE_PATH = os.path.join(DATA_DIR, "sample.tar.gz")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")

os.makedirs(DATA_DIR, exist_ok=True)

if os.path.isdir(SAMPLE_DIR):
    entries = os.listdir(SAMPLE_DIR)
    if len(entries) > 0:
        print(f"Already extracted: {SAMPLE_DIR} ({len(entries)} entries)")
        sys.exit(0)

if not os.path.isfile(ARCHIVE_PATH):
    print(f"Downloading PointOdyssey sample (3.1GB)...")
    print(f"URL: {SAMPLE_URL}")
    print(f"Destination: {ARCHIVE_PATH}")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(SAMPLE_URL, ARCHIVE_PATH, reporthook=report)
    print(f"\nDownload complete: {os.path.getsize(ARCHIVE_PATH) / 1024 / 1024:.1f} MB")
else:
    print(f"Archive already exists: {ARCHIVE_PATH}")

print(f"Extracting to {DATA_DIR}...")
with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
    tar.extractall(DATA_DIR)
print("Extraction complete.")

# Verify
if os.path.isdir(SAMPLE_DIR):
    sequences = [d for d in os.listdir(SAMPLE_DIR) if os.path.isdir(os.path.join(SAMPLE_DIR, d))]
    print(f"Found {len(sequences)} sequences in {SAMPLE_DIR}")
    if sequences:
        first = os.path.join(SAMPLE_DIR, sorted(sequences)[0])
        for name in ["rgbs", "depths", "normals", "anno.npz", "intrinsics.npy"]:
            path = os.path.join(first, name)
            exists = os.path.exists(path)
            print(f"  {'[OK]' if exists else '[MISSING]'} {name}")

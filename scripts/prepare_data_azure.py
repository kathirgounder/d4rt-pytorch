#!/usr/bin/env python3
"""
D4RT Data Preparation and Azure Upload Script

This script prepares training datasets and uploads them to Azure Blob Storage.

Supported datasets:
- Kubric MOVi-F (synthetic, primary training data)
- PointOdyssey (long-term tracking)
- ScanNet (indoor RGB-D)
- Sintel (optical flow benchmark)
- TartanAir (simulation)

Usage:
    # Setup Azure credentials first
    export AZURE_STORAGE_ACCOUNT="your_account"
    export AZURE_STORAGE_KEY="your_key"
    # OR
    export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"

    # Prepare and upload Kubric dataset
    python scripts/prepare_data_azure.py --dataset kubric --upload

    # Prepare multiple datasets
    python scripts/prepare_data_azure.py --dataset all --upload

    # Download only (no upload)
    python scripts/prepare_data_azure.py --dataset kubric --local-only
"""

import argparse
import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


# ============================================================================
# Configuration
# ============================================================================

DATASETS_CONFIG = {
    "kubric": {
        "name": "Kubric MOVi-F",
        "description": "Synthetic multi-object video dataset with 3D annotations",
        "source": "tensorflow_datasets",
        "tfds_name": "movi_f",
        "size_gb": 150,
        "container": "d4rt-training-data",
        "blob_prefix": "kubric/movi_f",
    },
    "pointodyssey": {
        "name": "PointOdyssey",
        "description": "Long-term point tracking dataset",
        "source": "url",
        "urls": [
            "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz",
            "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/val.tar.gz",
        ],
        "size_gb": 200,
        "container": "d4rt-training-data",
        "blob_prefix": "pointodyssey",
    },
    "scannet": {
        "name": "ScanNet",
        "description": "Indoor RGB-D dataset (requires agreement)",
        "source": "manual",
        "instructions": """
ScanNet requires signing a terms of use agreement.
1. Go to http://www.scan-net.org/
2. Fill out the agreement form
3. Download using the provided script
4. Place data in --local-dir/scannet/
        """,
        "size_gb": 1500,
        "container": "d4rt-training-data",
        "blob_prefix": "scannet",
    },
    "sintel": {
        "name": "MPI Sintel",
        "description": "Optical flow benchmark with depth",
        "source": "url",
        "urls": [
            "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
        ],
        "size_gb": 25,
        "container": "d4rt-training-data",
        "blob_prefix": "sintel",
    },
    "tartanair": {
        "name": "TartanAir",
        "description": "AirSim simulation dataset",
        "source": "url",
        "urls": [
            # TartanAir is large, download specific environments
            "https://tartanair.blob.core.windows.net/tartanair-release1/abandonedfactory/Easy/image_left.zip",
            "https://tartanair.blob.core.windows.net/tartanair-release1/abandonedfactory/Easy/depth_left.zip",
        ],
        "size_gb": 500,
        "container": "d4rt-training-data",
        "blob_prefix": "tartanair",
    },
}


# ============================================================================
# Azure Storage Utilities
# ============================================================================

class AzureUploader:
    """Handles uploads to Azure Blob Storage using azcopy."""

    def __init__(
        self,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
        sas_token: Optional[str] = None,
    ):
        self.account_name = account_name or os.environ.get("AZURE_STORAGE_ACCOUNT")
        self.account_key = account_key or os.environ.get("AZURE_STORAGE_KEY")
        self.connection_string = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.sas_token = sas_token or os.environ.get("AZURE_STORAGE_SAS_TOKEN")

        self._check_azcopy()
        self._validate_credentials()

    def _check_azcopy(self):
        """Check if azcopy is installed."""
        if shutil.which("azcopy") is None:
            print("ERROR: azcopy not found. Please install it:")
            print("  macOS: brew install azcopy")
            print("  Linux: https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10")
            print("  Windows: https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10")
            sys.exit(1)

    def _validate_credentials(self):
        """Validate Azure credentials are available."""
        if not any([self.connection_string, self.sas_token,
                    (self.account_name and self.account_key)]):
            print("ERROR: Azure credentials not found. Set one of:")
            print("  export AZURE_STORAGE_CONNECTION_STRING='...'")
            print("  export AZURE_STORAGE_ACCOUNT='...' && export AZURE_STORAGE_KEY='...'")
            print("  export AZURE_STORAGE_SAS_TOKEN='...'")
            sys.exit(1)

    def _get_blob_url(self, container: str, blob_prefix: str = "") -> str:
        """Get the blob URL for azcopy."""
        if self.sas_token:
            return f"https://{self.account_name}.blob.core.windows.net/{container}/{blob_prefix}?{self.sas_token}"
        elif self.account_name:
            return f"https://{self.account_name}.blob.core.windows.net/{container}/{blob_prefix}"
        else:
            # Parse from connection string
            # DefaultEndpointsProtocol=https;AccountName=xxx;AccountKey=xxx;EndpointSuffix=core.windows.net
            parts = dict(p.split("=", 1) for p in self.connection_string.split(";") if "=" in p)
            account = parts.get("AccountName")
            return f"https://{account}.blob.core.windows.net/{container}/{blob_prefix}"

    def create_container(self, container: str):
        """Create container if it doesn't exist."""
        print(f"Creating container: {container}")

        env = os.environ.copy()
        if self.account_key:
            env["AZCOPY_AUTO_LOGIN_TYPE"] = "SPN"

        cmd = [
            "azcopy", "make",
            self._get_blob_url(container, "").rstrip("/")
        ]

        if self.account_key:
            # Use environment variable for key
            env["AZURE_STORAGE_KEY"] = self.account_key

        try:
            subprocess.run(cmd, env=env, check=False, capture_output=True)
        except subprocess.CalledProcessError:
            pass  # Container might already exist

    def upload_directory(
        self,
        local_path: Path,
        container: str,
        blob_prefix: str,
        recursive: bool = True,
        overwrite: bool = False,
    ) -> bool:
        """Upload a directory to Azure Blob Storage."""
        print(f"Uploading {local_path} -> {container}/{blob_prefix}")

        blob_url = self._get_blob_url(container, blob_prefix)

        cmd = [
            "azcopy", "copy",
            str(local_path),
            blob_url,
            "--recursive" if recursive else "",
            "--overwrite" if overwrite else "--overwrite=ifSourceNewer",
        ]
        cmd = [c for c in cmd if c]  # Remove empty strings

        env = os.environ.copy()
        if self.account_key:
            env["AZURE_STORAGE_KEY"] = self.account_key

        try:
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR uploading: {e.stderr}")
            return False

    def upload_file(
        self,
        local_path: Path,
        container: str,
        blob_name: str,
    ) -> bool:
        """Upload a single file to Azure Blob Storage."""
        blob_url = self._get_blob_url(container, blob_name)

        cmd = ["azcopy", "copy", str(local_path), blob_url]

        env = os.environ.copy()
        if self.account_key:
            env["AZURE_STORAGE_KEY"] = self.account_key

        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR uploading {local_path}: {e}")
            return False


# ============================================================================
# Dataset Downloaders
# ============================================================================

def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress."""
    import urllib.request

    print(f"Downloading: {url}")
    print(f"Destination: {dest_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Progress: {pct:.1f}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="")

            print()
        return True
    except Exception as e:
        print(f"ERROR downloading {url}: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract tar.gz, zip, or other archives."""
    print(f"Extracting: {archive_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = "".join(archive_path.suffixes)

    try:
        if suffix in [".tar.gz", ".tgz"]:
            import tarfile
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(dest_dir)
        elif suffix == ".tar":
            import tarfile
            with tarfile.open(archive_path, "r:") as tar:
                tar.extractall(dest_dir)
        elif suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
        else:
            print(f"Unknown archive format: {suffix}")
            return False
        return True
    except Exception as e:
        print(f"ERROR extracting {archive_path}: {e}")
        return False


def prepare_kubric(local_dir: Path) -> Path:
    """Download and prepare Kubric MOVi-F dataset."""
    print("\n" + "="*60)
    print("Preparing Kubric MOVi-F Dataset")
    print("="*60)

    kubric_dir = local_dir / "kubric" / "movi_f"
    kubric_dir.mkdir(parents=True, exist_ok=True)

    # Check if TensorFlow Datasets is available
    try:
        import tensorflow_datasets as tfds
        print("Using TensorFlow Datasets to download Kubric...")

        # Download using tfds
        ds_builder = tfds.builder("movi_f", data_dir=str(kubric_dir))
        ds_builder.download_and_prepare()

        print(f"Kubric MOVi-F downloaded to: {kubric_dir}")

    except ImportError:
        print("TensorFlow Datasets not available.")
        print("Installing tensorflow-datasets...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow-datasets"], check=True)

        # Retry
        import tensorflow_datasets as tfds
        ds_builder = tfds.builder("movi_f", data_dir=str(kubric_dir))
        ds_builder.download_and_prepare()

    # Convert to D4RT format (optional preprocessing)
    convert_kubric_to_d4rt_format(kubric_dir)

    return kubric_dir


def convert_kubric_to_d4rt_format(kubric_dir: Path):
    """Convert Kubric TFDS format to D4RT-friendly format."""
    print("Converting Kubric to D4RT format...")

    output_dir = kubric_dir / "d4rt_format"
    output_dir.mkdir(exist_ok=True)

    try:
        import tensorflow_datasets as tfds
        import numpy as np
        from PIL import Image

        ds = tfds.load("movi_f", data_dir=str(kubric_dir), split="train")

        for i, example in enumerate(ds.take(100)):  # Process first 100 for testing
            sample_dir = output_dir / f"sample_{i:06d}"
            sample_dir.mkdir(exist_ok=True)

            # Save video frames
            video = example["video"].numpy()  # (T, H, W, 3)
            frames_dir = sample_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

            for t, frame in enumerate(video):
                img = Image.fromarray(frame)
                img.save(frames_dir / f"frame_{t:04d}.png")

            # Save depth
            if "depth" in example:
                depth = example["depth"].numpy()
                np.save(sample_dir / "depth.npy", depth)

            # Save segmentation
            if "segmentations" in example:
                seg = example["segmentations"].numpy()
                np.save(sample_dir / "segmentation.npy", seg)

            # Save camera info
            if "camera" in example:
                camera_info = {
                    "positions": example["camera"]["positions"].numpy().tolist(),
                    "quaternions": example["camera"]["quaternions"].numpy().tolist(),
                }
                with open(sample_dir / "camera.json", "w") as f:
                    json.dump(camera_info, f)

            # Save point tracks if available
            if "target_points" in example:
                tracks = example["target_points"].numpy()
                np.save(sample_dir / "tracks.npy", tracks)

            if i % 10 == 0:
                print(f"  Processed {i+1} samples...")

        print(f"Converted samples saved to: {output_dir}")

    except Exception as e:
        print(f"Warning: Could not convert Kubric format: {e}")
        print("You may need to implement custom conversion for your TFDS version.")


def prepare_pointodyssey(local_dir: Path) -> Path:
    """Download and prepare PointOdyssey dataset."""
    print("\n" + "="*60)
    print("Preparing PointOdyssey Dataset")
    print("="*60)

    po_dir = local_dir / "pointodyssey"
    po_dir.mkdir(parents=True, exist_ok=True)

    config = DATASETS_CONFIG["pointodyssey"]

    for url in config["urls"]:
        filename = url.split("/")[-1]
        archive_path = po_dir / filename

        if not archive_path.exists():
            download_file(url, archive_path)

        # Extract
        extract_archive(archive_path, po_dir)

    return po_dir


def prepare_sintel(local_dir: Path) -> Path:
    """Download and prepare MPI Sintel dataset."""
    print("\n" + "="*60)
    print("Preparing MPI Sintel Dataset")
    print("="*60)

    sintel_dir = local_dir / "sintel"
    sintel_dir.mkdir(parents=True, exist_ok=True)

    config = DATASETS_CONFIG["sintel"]

    for url in config["urls"]:
        filename = url.split("/")[-1]
        archive_path = sintel_dir / filename

        if not archive_path.exists():
            download_file(url, archive_path)

        # Extract
        extract_archive(archive_path, sintel_dir)

    return sintel_dir


def prepare_tartanair(local_dir: Path) -> Path:
    """Download and prepare TartanAir dataset (partial)."""
    print("\n" + "="*60)
    print("Preparing TartanAir Dataset")
    print("="*60)

    ta_dir = local_dir / "tartanair"
    ta_dir.mkdir(parents=True, exist_ok=True)

    config = DATASETS_CONFIG["tartanair"]

    for url in config["urls"]:
        filename = url.split("/")[-1]
        # Include environment name in path
        env_name = url.split("/")[-3]  # e.g., "abandonedfactory"
        archive_path = ta_dir / env_name / filename

        if not archive_path.exists():
            download_file(url, archive_path)
            extract_archive(archive_path, archive_path.parent)

    return ta_dir


# ============================================================================
# Main Script
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare D4RT training data and upload to Azure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="kubric",
        choices=["all", "kubric", "pointodyssey", "sintel", "tartanair", "scannet"],
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path.home() / "d4rt_data",
        help="Local directory for downloading data"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Azure Blob Storage after download"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only download locally, don't upload"
    )
    parser.add_argument(
        "--container",
        type=str,
        default="d4rt-training-data",
        help="Azure Blob Storage container name"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only upload existing data"
    )
    parser.add_argument(
        "--azure-account",
        type=str,
        help="Azure Storage account name (or set AZURE_STORAGE_ACCOUNT)"
    )
    parser.add_argument(
        "--azure-key",
        type=str,
        help="Azure Storage account key (or set AZURE_STORAGE_KEY)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("D4RT Data Preparation Script")
    print("="*60)
    print(f"Local directory: {args.local_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Upload to Azure: {args.upload and not args.local_only}")
    print()

    # Create local directory
    args.local_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to prepare
    if args.dataset == "all":
        datasets_to_prepare = ["kubric", "pointodyssey", "sintel"]
    else:
        datasets_to_prepare = [args.dataset]

    prepared_paths = {}

    # Download/prepare datasets
    if not args.skip_download:
        for ds_name in datasets_to_prepare:
            config = DATASETS_CONFIG[ds_name]

            if config["source"] == "manual":
                print(f"\n{config['name']} requires manual download:")
                print(config["instructions"])
                continue

            # Prepare dataset
            if ds_name == "kubric":
                prepared_paths[ds_name] = prepare_kubric(args.local_dir)
            elif ds_name == "pointodyssey":
                prepared_paths[ds_name] = prepare_pointodyssey(args.local_dir)
            elif ds_name == "sintel":
                prepared_paths[ds_name] = prepare_sintel(args.local_dir)
            elif ds_name == "tartanair":
                prepared_paths[ds_name] = prepare_tartanair(args.local_dir)
    else:
        # Use existing paths
        for ds_name in datasets_to_prepare:
            config = DATASETS_CONFIG[ds_name]
            ds_path = args.local_dir / ds_name
            if ds_path.exists():
                prepared_paths[ds_name] = ds_path

    # Upload to Azure
    if args.upload and not args.local_only:
        print("\n" + "="*60)
        print("Uploading to Azure Blob Storage")
        print("="*60)

        uploader = AzureUploader(
            account_name=args.azure_account,
            account_key=args.azure_key,
        )

        # Create container
        uploader.create_container(args.container)

        # Upload each dataset
        for ds_name, ds_path in prepared_paths.items():
            config = DATASETS_CONFIG[ds_name]

            print(f"\nUploading {config['name']}...")
            success = uploader.upload_directory(
                local_path=ds_path,
                container=args.container,
                blob_prefix=config["blob_prefix"],
            )

            if success:
                print(f"  ✓ {ds_name} uploaded successfully")
            else:
                print(f"  ✗ {ds_name} upload failed")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for ds_name, ds_path in prepared_paths.items():
        config = DATASETS_CONFIG[ds_name]
        print(f"\n{config['name']}:")
        print(f"  Local path: {ds_path}")
        if args.upload and not args.local_only:
            print(f"  Azure path: {args.container}/{config['blob_prefix']}")

    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("""
1. To train locally:
   python train.py --data-root {local_dir}/kubric --dataset kubric

2. To train on Azure VM:
   # Mount blob storage or download from Azure
   azcopy copy "https://{account}.blob.core.windows.net/{container}/*" ./data/ --recursive
   python train.py --data-root ./data/kubric --dataset kubric

3. For multi-GPU training:
   torchrun --nproc_per_node=8 train.py --data-root ./data/kubric --dataset kubric
""".format(
        local_dir=args.local_dir,
        account=os.environ.get("AZURE_STORAGE_ACCOUNT", "your_account"),
        container=args.container,
    ))


if __name__ == "__main__":
    main()

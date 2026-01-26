#!/usr/bin/env python3
"""
Download D4RT training datasets and upload to Azure Storage.

This handles datasets NOT available on HuggingFace:
- Kubric MOVi-F (via TensorFlow Datasets)
- MPI Sintel (direct download)
- TartanAir (via tartanair package)
- PointOdyssey sample (direct URL)

Usage:
    # Download and upload to Azure
    python scripts/download_datasets.py --dataset sintel --upload-azure
    python scripts/download_datasets.py --dataset all --upload-azure

    # Download locally only
    python scripts/download_datasets.py --dataset kubric --output-dir ~/d4rt_data
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import zipfile
import tarfile
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =============================================================================
# AZURE CONFIGURATION
# =============================================================================

HF_TOKEN = os.getenv("HF_TOKEN", "")
AZURE_CONN_STRING = os.getenv("AZURE_CONN_STRING", "")
CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "d4rt-training-data")
MAX_WORKERS = 4


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASETS = {
    "kubric": {
        "name": "Kubric MOVi-F",
        "description": "Synthetic multi-object video with 3D annotations",
        "size_gb": 150,
        "method": "tfds",
        "tfds_name": "movi_f",
    },
    "sintel": {
        "name": "MPI Sintel",
        "description": "CGI film with optical flow and depth",
        "size_gb": 7,
        "method": "url",
        "urls": [
            ("MPI-Sintel-complete.zip", "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"),
            ("MPI-Sintel-depth-training.zip", "http://files.is.tue.mpg.de/sintel/MPI-Sintel-depth-training-20150305.zip"),
        ],
    },
    "tartanair": {
        "name": "TartanAir",
        "description": "Large-scale synthetic environments",
        "size_gb": 50,
        "method": "tartanair",
        "environments": ["abandonedfactory", "amusement", "carwelding"],
    },
    "pointodyssey_sample": {
        "name": "PointOdyssey Sample",
        "description": "Small sample for quick testing (3.1GB)",
        "size_gb": 3.1,
        "method": "url",
        "urls": [
            ("sample.tar.gz", "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/sample.tar.gz"),
        ],
    },
}


# =============================================================================
# DOWNLOAD UTILITIES
# =============================================================================

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: Path, desc: str = None):
    """Download a file with progress bar. Handles SSL issues."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  Already exists: {dest_path}")
        return True

    print(f"  Downloading: {url}")

    # Try multiple methods
    # Method 1: Try curl (most reliable, handles SSL well)
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", str(dest_path), "--progress-bar", url],
            check=True,
            capture_output=False
        )
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 2: Try wget
    try:
        result = subprocess.run(
            ["wget", "-O", str(dest_path), "--progress=bar:force", url],
            check=True,
            capture_output=False
        )
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Method 3: Python with SSL context (fallback)
    try:
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc or dest_path.name) as t:
            with urllib.request.urlopen(url, context=ssl_context) as response:
                total_size = int(response.headers.get('content-length', 0))
                t.total = total_size
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        t.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract tar.gz or zip archives."""
    print(f"  Extracting: {archive_path.name}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = "".join(archive_path.suffixes)

    try:
        if suffix in [".tar.gz", ".tgz"]:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(dest_dir)
        elif suffix == ".tar":
            with tarfile.open(archive_path, "r:") as tar:
                tar.extractall(dest_dir)
        elif suffix == ".zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
        else:
            print(f"  Unknown archive format: {suffix}")
            return False
        return True
    except Exception as e:
        print(f"  ERROR extracting: {e}")
        return False


# =============================================================================
# AZURE UPLOAD
# =============================================================================

def get_azure_client():
    """Initialize Azure Blob Storage client."""
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        print("Installing azure-storage-blob...")
        subprocess.run([sys.executable, "-m", "pip", "install", "azure-storage-blob"], check=True)
        from azure.storage.blob import BlobServiceClient

    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_CONN_STRING,
        connection_timeout=600,
        read_timeout=600
    )

    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    if not container_client.exists():
        print(f"Creating container: {CONTAINER_NAME}")
        container_client.create_container()

    return container_client


def upload_file_to_azure(container_client, local_path: Path, blob_name: str):
    """Upload a single file to Azure."""
    blob_client = container_client.get_blob_client(blob_name)
    file_size = local_path.stat().st_size

    # Skip if already exists with same size
    if blob_client.exists():
        try:
            props = blob_client.get_blob_properties()
            if props.size == file_size:
                return "Skipped"
        except:
            pass

    try:
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True, max_concurrency=2, timeout=600)
        return "Uploaded"
    except Exception as e:
        return f"Failed: {e}"


def upload_directory_to_azure(local_dir: Path, azure_prefix: str):
    """Upload entire directory to Azure Blob Storage."""
    print(f"\n  Uploading {local_dir} -> {azure_prefix}")

    container_client = get_azure_client()

    # Get all files
    all_files = list(local_dir.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]

    if not all_files:
        print("  No files to upload")
        return

    print(f"  Found {len(all_files)} files")

    stats = {"uploaded": 0, "skipped": 0, "failed": 0}

    with tqdm(total=len(all_files), unit="file", desc="  Uploading") as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for file_path in all_files:
                rel_path = file_path.relative_to(local_dir)
                blob_name = f"{azure_prefix}/{rel_path}"
                futures[executor.submit(upload_file_to_azure, container_client, file_path, blob_name)] = blob_name

            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                if result == "Skipped":
                    stats["skipped"] += 1
                elif result == "Uploaded":
                    stats["uploaded"] += 1
                else:
                    stats["failed"] += 1

    print(f"  Results: {stats['uploaded']} uploaded, {stats['skipped']} skipped, {stats['failed']} failed")
    gc.collect()


# =============================================================================
# DATASET DOWNLOADERS
# =============================================================================

def download_kubric(output_dir: Path, upload_azure: bool = False):
    """Download Kubric MOVi-F from Google Cloud Storage."""
    print("\n" + "="*60)
    print("Downloading Kubric MOVi-F")
    print("="*60)

    kubric_dir = output_dir / "kubric"
    kubric_dir.mkdir(parents=True, exist_ok=True)

    # MOVi-F is hosted on Google Cloud Storage
    # We use gsutil to download it
    GCS_PATH = "gs://kubric-public/tfds/movi_f/1.0.0"

    print("Checking for gsutil...")

    # Check if gsutil is available
    gsutil_available = subprocess.run(
        ["which", "gsutil"], capture_output=True
    ).returncode == 0

    if not gsutil_available:
        print("Installing google-cloud-storage CLI...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "google-cloud-storage", "gcsfs"],
                check=True
            )
        except:
            pass

        # Try installing gsutil via snap or apt
        try:
            subprocess.run(["sudo", "snap", "install", "google-cloud-cli", "--classic"], check=False)
        except:
            pass

    print(f"Downloading from {GCS_PATH}...")
    print("This is a large dataset (~150GB), it will take a while...")

    try:
        # Method 1: Try gsutil
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", GCS_PATH, str(kubric_dir)],
            capture_output=False
        )
        if result.returncode == 0:
            print(f"  Downloaded to: {kubric_dir}")
            if upload_azure:
                upload_directory_to_azure(kubric_dir, "datasets/kubric")
            return True
    except FileNotFoundError:
        pass

    # Method 2: Try gcsfs (Python)
    try:
        import gcsfs
        print("Using gcsfs to download...")

        fs = gcsfs.GCSFileSystem(token='anon')
        files = fs.ls("kubric-public/tfds/movi_f/1.0.0")

        print(f"  Found {len(files)} files")

        for gcs_file in tqdm(files, desc="Downloading"):
            local_file = kubric_dir / Path(gcs_file).name
            if not local_file.exists():
                fs.get(gcs_file, str(local_file))

        print(f"  Downloaded to: {kubric_dir}")

        if upload_azure:
            upload_directory_to_azure(kubric_dir, "datasets/kubric")

        return True

    except ImportError:
        print("  Installing gcsfs...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gcsfs"], check=True)
        import gcsfs

        fs = gcsfs.GCSFileSystem(token='anon')
        files = fs.ls("kubric-public/tfds/movi_f/1.0.0")

        for gcs_file in tqdm(files, desc="Downloading"):
            local_file = kubric_dir / Path(gcs_file).name
            if not local_file.exists():
                fs.get(gcs_file, str(local_file))

        if upload_azure:
            upload_directory_to_azure(kubric_dir, "datasets/kubric")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        print("\n  Manual download instructions:")
        print("  1. Install gsutil: https://cloud.google.com/storage/docs/gsutil_install")
        print(f"  2. Run: gsutil -m cp -r {GCS_PATH} {kubric_dir}")
        return False


def download_sintel(output_dir: Path, upload_azure: bool = False):
    """Download MPI Sintel dataset."""
    print("\n" + "="*60)
    print("Downloading MPI Sintel")
    print("="*60)

    sintel_dir = output_dir / "sintel"
    sintel_dir.mkdir(parents=True, exist_ok=True)

    config = DATASETS["sintel"]
    success = True

    for filename, url in config["urls"]:
        archive_path = sintel_dir / filename
        if download_file(url, archive_path):
            if not extract_archive(archive_path, sintel_dir):
                success = False
        else:
            success = False

    if success:
        print(f"  Sintel downloaded to: {sintel_dir}")
        if upload_azure:
            upload_directory_to_azure(sintel_dir, "datasets/sintel")

    return success


def download_tartanair(output_dir: Path, upload_azure: bool = False):
    """Download TartanAir subset."""
    print("\n" + "="*60)
    print("Downloading TartanAir")
    print("="*60)

    tartanair_dir = output_dir / "tartanair"
    tartanair_dir.mkdir(parents=True, exist_ok=True)

    try:
        import tartanair
    except ImportError:
        print("Installing tartanair...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tartanair"], check=True)
        import tartanair

    print("Downloading TartanAir environments...")
    config = DATASETS["tartanair"]

    try:
        tartanair.init(str(tartanair_dir))
        for env in config["environments"]:
            print(f"  Downloading environment: {env}")
            tartanair.download(
                env=env,
                difficulty=['easy'],
                modality=['image', 'depth'],
                camera_name=['lcam_front']
            )
        print(f"  TartanAir downloaded to: {tartanair_dir}")

        if upload_azure:
            upload_directory_to_azure(tartanair_dir, "datasets/tartanair")

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  You may need to download manually from https://tartanair.org/")
        return False


def download_pointodyssey_sample(output_dir: Path, upload_azure: bool = False):
    """Download PointOdyssey sample for quick testing."""
    print("\n" + "="*60)
    print("Downloading PointOdyssey Sample (3.1GB)")
    print("="*60)

    po_dir = output_dir / "pointodyssey"
    po_dir.mkdir(parents=True, exist_ok=True)

    config = DATASETS["pointodyssey_sample"]

    for filename, url in config["urls"]:
        archive_path = po_dir / filename
        if download_file(url, archive_path):
            extract_archive(archive_path, po_dir)

    print(f"  PointOdyssey sample downloaded to: {po_dir}")

    if upload_azure:
        upload_directory_to_azure(po_dir, "datasets/pointodyssey")

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download D4RT training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets available:
  kubric              Kubric MOVi-F (~150GB) - Primary synthetic training
  sintel              MPI Sintel (~7GB) - Evaluation with depth/flow
  tartanair           TartanAir (~50GB) - Synthetic environments
  pointodyssey_sample PointOdyssey sample (3.1GB) - Quick test
  all                 Download all datasets

For full PointOdyssey (~170GB), use transfer_hf_to_azure.py with:
  python scripts/transfer_hf_to_azure.py --datasets-only
        """
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        choices=["kubric", "sintel", "tartanair", "pointodyssey_sample", "all"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path.home() / "d4rt_data",
        help="Output directory (default: ~/d4rt_data)"
    )
    parser.add_argument(
        "--upload-azure",
        action="store_true",
        help="Upload to Azure after download (requires AZURE_CONN_STRING)"
    )
    args = parser.parse_args()

    print("="*60)
    print("D4RT Dataset Downloader")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Upload to Azure: {args.upload_azure}")
    if args.upload_azure:
        print(f"Azure container: {CONTAINER_NAME}")
    print("="*60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = []
    if args.dataset == "all":
        datasets_to_download = ["pointodyssey_sample", "sintel", "kubric"]
    else:
        datasets_to_download = [args.dataset]

    results = {}

    for ds_name in datasets_to_download:
        if ds_name == "kubric":
            results[ds_name] = download_kubric(args.output_dir, args.upload_azure)
        elif ds_name == "sintel":
            results[ds_name] = download_sintel(args.output_dir, args.upload_azure)
        elif ds_name == "tartanair":
            results[ds_name] = download_tartanair(args.output_dir, args.upload_azure)
        elif ds_name == "pointodyssey_sample":
            results[ds_name] = download_pointodyssey_sample(args.output_dir, args.upload_azure)

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for ds_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {DATASETS[ds_name]['name']}: {status}")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    if args.upload_azure:
        print(f"""
Data has been uploaded to Azure container: {CONTAINER_NAME}

1. For full PointOdyssey (~170GB) from HuggingFace:
   python scripts/transfer_hf_to_azure.py --datasets-only

2. On Azure VM, download data:
   export AZURE_STORAGE_ACCOUNT=signiqlab
   export AZURE_STORAGE_KEY=<your_key>
   ./scripts/download_from_azure.sh all ~/d4rt_data

3. Start training:
   python train.py --config configs/d4rt_azure_a100.yaml --data-root ~/d4rt_data
""")
    else:
        print(f"""
1. For full PointOdyssey (~170GB) from HuggingFace:
   python scripts/transfer_hf_to_azure.py --datasets-only

2. Upload local data to Azure:
   python scripts/download_datasets.py -d all --upload-azure

3. Start training locally:
   python train.py --config configs/d4rt_rtx5090.yaml --data-root {args.output_dir}
""")


if __name__ == "__main__":
    main()

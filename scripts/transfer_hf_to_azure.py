#!/usr/bin/env python3
"""
Transfer D4RT training data and pretrained weights from HuggingFace to Azure Storage.

This script transfers:
1. VideoMAE pretrained weights (for encoder initialization)
2. Training/evaluation datasets

Prerequisites:
    pip install huggingface_hub azure-storage-blob tqdm

Usage:
    python scripts/transfer_hf_to_azure.py
    python scripts/transfer_hf_to_azure.py --checkpoints-only
    python scripts/transfer_hf_to_azure.py --dry-run
"""

import os
import sys
import time
import logging
import gc
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

HF_TOKEN = os.getenv("HF_TOKEN", "")
AZURE_CONN_STRING = os.getenv("AZURE_CONN_STRING", "")
CONTAINER_NAME = "d4rt-training-data"

# =============================================================================
# TRANSFER TARGETS
# =============================================================================
# Format: (HuggingFace repo ID, Azure destination folder, repo_type)
# repo_type: "model" or "dataset"

TRANSFER_TARGETS = {
    "checkpoints": [
        # VideoMAE v1 - Pretrained weights for encoder initialization
        ("MCG-NJU/videomae-base", "checkpoints/videomae-base", "model"),
        ("MCG-NJU/videomae-large", "checkpoints/videomae-large", "model"),
        ("MCG-NJU/videomae-huge-finetuned-kinetics", "checkpoints/videomae-huge", "model"),
    ],

    "datasets": [
        # =================================================================
        # PointOdyssey - PRIMARY TRAINING DATA (Recommended)
        # Synthetic with complete 3D annotations: RGB, Depth, 3D tracks,
        # Normals, Camera intrinsics/extrinsics, Visibility
        # Paper: ICCV 2023 - "PointOdyssey: A Large-Scale Synthetic Dataset"
        # =================================================================
        ("aharley/pointodyssey", "datasets/pointodyssey", "dataset"),

        # =================================================================
        # TAP-Vid - Point tracking benchmark
        # Includes TAP-Vid-Kubric, TAP-Vid-DAVIS, TAP-Vid-RGB-Stacking
        # =================================================================
        ("google-deepmind/tapvid", "datasets/tapvid", "dataset"),

        # =================================================================
        # CoTracker training data
        # =================================================================
        ("facebook/co-tracker", "datasets/cotracker", "dataset"),
    ],
}

# Concurrency settings
MAX_WORKERS = 4
MAX_RETRIES = 8
SMALL_FILE_LIMIT = 10 * 1024 * 1024  # 10MB

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    filename='d4rt_transfer_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(message)s'
)

# =============================================================================
# AZURE CLIENT
# =============================================================================

def get_azure_client(container_name: str = CONTAINER_NAME):
    """Initialize Azure Blob Storage client."""
    if not AZURE_CONN_STRING:
        print("ERROR: AZURE_CONN_STRING must be set")
        sys.exit(1)

    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_CONN_STRING,
        connection_timeout=600,
        read_timeout=600
    )

    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        print(f"Creating container: {container_name}")
        container_client.create_container()

    return container_client


# =============================================================================
# FILE TRANSFER
# =============================================================================

def upload_file_to_azure(container_client, local_path: Path, blob_name: str):
    """Upload a single file to Azure Blob Storage."""
    blob_client = container_client.get_blob_client(blob_name)

    file_size = local_path.stat().st_size

    # Check if already exists with same size
    if blob_client.exists():
        try:
            props = blob_client.get_blob_properties()
            if props.size == file_size:
                return "Skipped"
        except Exception:
            pass

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(local_path, "rb") as f:
                if file_size < SMALL_FILE_LIMIT:
                    blob_client.upload_blob(f, overwrite=True, timeout=600)
                else:
                    blob_client.upload_blob(f, overwrite=True, max_concurrency=2, timeout=600)
            return "Uploaded"
        except Exception as e:
            if attempt == MAX_RETRIES:
                logging.error(f"FAIL {blob_name}: {e}")
                return f"Failed: {e}"
            time.sleep(5 * (2 ** (attempt - 1)))

    return "Failed-Unknown"


def process_repo(container_client, repo_id: str, dest_folder: str, repo_type: str,
                 dry_run: bool = False, max_workers: int = 4):
    """Download repo from HuggingFace and upload to Azure."""
    print(f"\n{'='*60}")
    print(f"Processing: {repo_id}")
    print(f"Type: {repo_type}")
    print(f"Destination: {dest_folder}")
    print(f"{'='*60}")

    stats = {"skipped": 0, "uploaded": 0, "failed": 0}

    try:
        # List files in the repo
        print("  Listing files...")
        files = list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            token=HF_TOKEN
        )
        print(f"  Found {len(files)} files")

        if dry_run:
            print("  [DRY RUN] Would transfer:")
            for f in files[:10]:
                print(f"    {f}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more files")
            return stats

        # Download entire repo to temp location
        print("  Downloading from HuggingFace...")
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            token=HF_TOKEN,
            local_dir=f"/tmp/hf_cache/{repo_id.replace('/', '_')}"
        )
        local_path = Path(local_dir)
        print(f"  Downloaded to: {local_path}")

        # Get all files recursively
        all_files = list(local_path.rglob("*"))
        all_files = [f for f in all_files if f.is_file()]
        print(f"  Uploading {len(all_files)} files to Azure...")

        # Upload to Azure
        with tqdm(total=len(all_files), unit="file", desc="Uploading") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for file_path in all_files:
                    rel_path = file_path.relative_to(local_path)
                    blob_name = f"{dest_folder}/{rel_path}"
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
                        pbar.set_description(f"Err: {result[:20]}")

        print(f"  Results: {stats['uploaded']} uploaded, {stats['skipped']} skipped, {stats['failed']} failed")

    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            print(f"  Repository not found: {repo_id}")
            print(f"  Check if repo exists and you have access")
        else:
            print(f"  Error: {e}")
        logging.error(f"Error processing {repo_id}: {e}")
        stats["failed"] = 1

    # Cleanup
    gc.collect()

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Transfer D4RT data from HuggingFace to Azure")
    parser.add_argument("--checkpoints-only", action="store_true", help="Only transfer checkpoints")
    parser.add_argument("--datasets-only", action="store_true", help="Only transfer datasets")
    parser.add_argument("--dry-run", action="store_true", help="List files without transferring")
    parser.add_argument("--repos", nargs="+", help="Specific repos to transfer")
    parser.add_argument("--container", type=str, default=CONTAINER_NAME, help="Azure container name")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    args = parser.parse_args()

    print("="*60)
    print("D4RT HuggingFace to Azure Transfer")
    print("="*60)
    print(f"Container: {args.container}")
    print(f"HF Token: {'Set' if HF_TOKEN else 'NOT SET'}")
    print(f"Azure Connection: {'Set' if AZURE_CONN_STRING else 'NOT SET'}")
    print(f"Workers: {args.workers}")
    print("="*60)

    container_client = get_azure_client(args.container)

    # Determine targets
    targets = []

    if args.repos:
        for repo in args.repos:
            # Default to model type
            targets.append((repo, f"custom/{repo.replace('/', '_')}", "model"))
    else:
        if not args.datasets_only:
            targets.extend(TRANSFER_TARGETS["checkpoints"])
        if not args.checkpoints_only:
            targets.extend(TRANSFER_TARGETS["datasets"])

    print(f"\nWill process {len(targets)} repositories:")
    for repo_id, dest, rtype in targets:
        print(f"  [{rtype}] {repo_id} -> {dest}")

    if args.dry_run:
        print("\n*** DRY RUN MODE ***\n")

    # Process
    total_stats = {"skipped": 0, "uploaded": 0, "failed": 0}

    for repo_id, dest_folder, repo_type in targets:
        stats = process_repo(container_client, repo_id, dest_folder, repo_type,
                           args.dry_run, args.workers)
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    # Summary
    print("\n" + "="*60)
    print("TRANSFER COMPLETE")
    print("="*60)
    print(f"Total uploaded:  {total_stats['uploaded']}")
    print(f"Total skipped:   {total_stats['skipped']}")
    print(f"Total failed:    {total_stats['failed']}")
    print("="*60)

    if total_stats['failed'] > 0:
        print("\nCheck 'd4rt_transfer_errors.log' for details.")


if __name__ == "__main__":
    main()

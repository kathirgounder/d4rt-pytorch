#!/usr/bin/env python3
"""Evaluation script for D4RT model."""

import argparse
import os
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models import D4RT, create_d4rt
from data import VideoDataset, SintelDataset, ScanNetDataset, KubricDataset, collate_fn
from utils.metrics import (
    compute_depth_metrics,
    compute_pose_metrics,
    compute_tracking_metrics,
    compute_point_cloud_metrics
)
from utils.camera import estimate_camera_pose, estimate_intrinsics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate D4RT model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to evaluation data')
    parser.add_argument('--dataset', type=str, default='sintel',
                        choices=['sintel', 'scannet', 'kubric', 'kitti', 'bonn', 'tapvid3d'],
                        help='Evaluation dataset')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'depth', 'pose', 'tracking', 'pointcloud'],
                        help='Evaluation task')
    parser.add_argument('--split', type=str, default='test',
                        help='Data split to evaluate')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num-frames', type=int, default=48,
                        help='Number of frames per clip')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size')

    parser.add_argument('--output-dir', type=str, default='eval_results',
                        help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to disk')

    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get model config from checkpoint
    args = checkpoint.get('args', {})
    encoder_variant = args.get('encoder', 'base')
    img_size = args.get('img_size', 256)
    num_frames = args.get('num_frames', 48)
    decoder_depth = args.get('decoder_depth', 8)
    patch_size = args.get('patch_size', 9)

    model = create_d4rt(
        variant=encoder_variant,
        img_size=img_size,
        temporal_size=num_frames,
        decoder_depth=decoder_depth,
        query_patch_size=patch_size
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def create_eval_dataloader(args):
    """Create evaluation dataloader."""
    if args.dataset == 'sintel':
        dataset = SintelDataset(
            args.data_root,
            split='training',  # Sintel uses training for eval
            pass_name='final',
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=0,  # No query sampling for eval
            transform=None
        )
    elif args.dataset == 'scannet':
        dataset = ScanNetDataset(
            args.data_root,
            split=args.split,
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=0,
            transform=None
        )
    elif args.dataset == 'kubric':
        dataset = KubricDataset(
            args.data_root,
            split=args.split,
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=0,
            transform=None
        )
    else:
        dataset = VideoDataset(
            args.data_root,
            split=args.split,
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=0,
            transform=None
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return dataloader


@torch.no_grad()
def evaluate_depth(model, dataloader, device, scale_invariant=True, shift_invariant=False):
    """Evaluate depth estimation."""
    all_metrics = []

    for batch in tqdm(dataloader, desc='Evaluating depth'):
        video = batch['video'].to(device)  # (B, T, H, W, C)
        gt_depth = batch.get('depth')

        if gt_depth is None:
            continue

        gt_depth = gt_depth.to(device)

        # Predict depth
        pred_depth = model.predict_depth(video)

        # Compute metrics per frame
        B, T = pred_depth.shape[:2]
        for b in range(B):
            for t in range(T):
                metrics = compute_depth_metrics(
                    pred_depth[b, t],
                    gt_depth[b, t],
                    scale_invariant=scale_invariant,
                    shift_invariant=shift_invariant
                )
                all_metrics.append({k: v.item() for k, v in metrics.items()})

    # Aggregate metrics
    if not all_metrics:
        return {}

    result = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        result[key] = np.mean(values) if values else float('nan')

    return result


@torch.no_grad()
def evaluate_pose(model, dataloader, device, with_alignment=True):
    """Evaluate camera pose estimation."""
    all_metrics = []

    for batch in tqdm(dataloader, desc='Evaluating pose'):
        video = batch['video'].to(device)
        gt_poses = batch.get('extrinsics')

        if gt_poses is None:
            continue

        gt_poses = gt_poses.to(device)

        B, T = video.shape[:2]

        # Encode video
        video_input = video.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        encoder_features = model.encode(video_input)

        # Prepare frames for decoder
        frames = video.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

        for b in range(B):
            # Estimate poses relative to first frame
            pred_poses = [torch.eye(4, device=device)]

            for t in range(1, T):
                R, trans = estimate_camera_pose(
                    model.decoder,
                    encoder_features[b:b+1],
                    frames[b:b+1],
                    frame_i=0,
                    frame_j=t
                )
                pose = torch.eye(4, device=device)
                pose[:3, :3] = R
                pose[:3, 3] = trans
                pred_poses.append(pose)

            pred_poses = torch.stack(pred_poses)

            # Compute metrics
            metrics = compute_pose_metrics(pred_poses, gt_poses[b], align=with_alignment)
            all_metrics.append({k: v.item() for k, v in metrics.items()})

    # Aggregate
    if not all_metrics:
        return {}

    result = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        result[key] = np.mean(values)

    return result


@torch.no_grad()
def evaluate_tracking(model, dataloader, device, use_gt_intrinsics=False):
    """Evaluate 3D point tracking."""
    all_metrics = []

    for batch in tqdm(dataloader, desc='Evaluating tracking'):
        video = batch['video'].to(device)
        gt_tracks = batch.get('tracks')
        visibility = batch.get('visibility')

        if gt_tracks is None:
            continue

        gt_tracks = gt_tracks.to(device)
        visibility = visibility.to(device) if visibility is not None else None

        B, T = video.shape[:2]
        N = gt_tracks.shape[1]

        # Sample query points from first frame
        # Use a grid or random sampling
        query_points = torch.rand(B, N, 2, device=device)  # Random points
        query_frames = torch.zeros(B, N, device=device, dtype=torch.long)

        # Predict tracks
        track_outputs = model.predict_point_tracks(video, query_points, query_frames)
        pred_tracks = track_outputs['tracks_3d']

        # Compute metrics
        for b in range(B):
            metrics = compute_tracking_metrics(
                pred_tracks[b],
                gt_tracks[b],
                visibility[b] if visibility is not None else torch.ones(N, T, device=device)
            )
            all_metrics.append({k: v.item() for k, v in metrics.items()})

    # Aggregate
    if not all_metrics:
        return {}

    result = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        result[key] = np.mean(values)

    return result


@torch.no_grad()
def evaluate_point_cloud(model, dataloader, device, reference_frame=0):
    """Evaluate point cloud reconstruction."""
    all_metrics = []

    for batch in tqdm(dataloader, desc='Evaluating point cloud'):
        video = batch['video'].to(device)
        gt_points = batch.get('point_cloud')

        if gt_points is None:
            # Generate GT point cloud from depth if available
            gt_depth = batch.get('depth')
            if gt_depth is None:
                continue
            # Skip for now - would need intrinsics to generate GT point cloud
            continue

        gt_points = gt_points.to(device)

        # Predict point cloud
        pc_outputs = model.predict_point_cloud(video, reference_frame=reference_frame)
        pred_points = pc_outputs['points']

        # Compute metrics
        B = pred_points.shape[0]
        for b in range(B):
            metrics = compute_point_cloud_metrics(pred_points[b], gt_points[b])
            all_metrics.append({k: v.item() for k, v in metrics.items()})

    # Aggregate
    if not all_metrics:
        return {}

    result = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        result[key] = np.mean(values) if values else float('nan')

    return result


def main():
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Create dataloader
    print(f"Loading {args.dataset} dataset from {args.data_root}")
    dataloader = create_eval_dataloader(args)

    results = {}

    # Run evaluations
    if args.task in ['all', 'depth']:
        print("\nEvaluating depth estimation...")
        depth_results_s = evaluate_depth(model, dataloader, device, scale_invariant=True, shift_invariant=False)
        depth_results_ss = evaluate_depth(model, dataloader, device, scale_invariant=True, shift_invariant=True)

        results['depth_scale'] = depth_results_s
        results['depth_scale_shift'] = depth_results_ss

        print("Depth (scale):", depth_results_s)
        print("Depth (scale+shift):", depth_results_ss)

    if args.task in ['all', 'pose']:
        print("\nEvaluating camera pose estimation...")
        pose_results = evaluate_pose(model, dataloader, device)
        results['pose'] = pose_results
        print("Pose:", pose_results)

    if args.task in ['all', 'tracking']:
        print("\nEvaluating 3D tracking...")
        tracking_results = evaluate_tracking(model, dataloader, device)
        results['tracking'] = tracking_results
        print("Tracking:", tracking_results)

    if args.task in ['all', 'pointcloud']:
        print("\nEvaluating point cloud reconstruction...")
        pc_results = evaluate_point_cloud(model, dataloader, device)
        results['point_cloud'] = pc_results
        print("Point Cloud:", pc_results)

    # Save results
    results_file = output_dir / f'results_{args.dataset}_{args.task}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Inference script for D4RT model."""

import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2

from models import create_d4rt
from utils.visualization import (
    visualize_depth,
    visualize_point_cloud,
    visualize_tracks,
    save_point_cloud_ply
)


def parse_args():
    parser = argparse.ArgumentParser(description='D4RT Inference')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video or image directory')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')

    parser.add_argument('--task', type=str, default='depth',
                        choices=['depth', 'tracking', 'pointcloud', 'all'],
                        help='Inference task')

    parser.add_argument('--num-frames', type=int, default=48,
                        help='Number of frames to process')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--fps', type=int, default=24,
                        help='Output video FPS')

    # Tracking options
    parser.add_argument('--num-tracks', type=int, default=100,
                        help='Number of points to track')
    parser.add_argument('--track-frame', type=int, default=0,
                        help='Frame to sample tracking points from')

    # Point cloud options
    parser.add_argument('--ref-frame', type=int, default=0,
                        help='Reference frame for point cloud')
    parser.add_argument('--pc-stride', type=int, default=2,
                        help='Spatial stride for point cloud generation')

    return parser.parse_args()


def load_video(video_path, num_frames, img_size):
    """Load video from file or image directory."""
    video_path = Path(video_path)

    frames = []

    if video_path.is_dir():
        # Load from image directory
        image_files = sorted(video_path.glob('*.png')) + sorted(video_path.glob('*.jpg'))
        for img_file in image_files[:num_frames]:
            img = Image.open(img_file).convert('RGB')
            img = img.resize((img_size, img_size), Image.BILINEAR)
            frames.append(np.array(img))
    else:
        # Load from video file
        cap = cv2.VideoCapture(str(video_path))
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
        cap.release()

    if not frames:
        raise ValueError(f"Could not load frames from {video_path}")

    # Pad if necessary
    while len(frames) < num_frames:
        frames.append(frames[-1])

    video = np.stack(frames[:num_frames], axis=0)  # (T, H, W, 3)
    video = torch.from_numpy(video).float() / 255.0

    return video


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

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


@torch.no_grad()
def run_depth_inference(model, video, output_dir):
    """Run depth inference and save results."""
    print("Running depth inference...")

    video = video.unsqueeze(0)  # (1, T, H, W, 3)
    depth = model.predict_depth(video)  # (1, T, H, W)
    depth = depth.squeeze(0).cpu()  # (T, H, W)

    T = depth.shape[0]
    video_np = (video.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    # Save depth frames
    depth_dir = output_dir / 'depth'
    depth_dir.mkdir(exist_ok=True)

    for t in range(T):
        depth_vis = visualize_depth(depth[t])
        Image.fromarray(depth_vis).save(depth_dir / f'depth_{t:04d}.png')

        # Save raw depth
        np.save(depth_dir / f'depth_{t:04d}.npy', depth[t].numpy())

    # Create side-by-side video
    print("Creating depth video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    H, W = depth.shape[1:]
    writer = cv2.VideoWriter(
        str(output_dir / 'depth_video.mp4'),
        fourcc, 24, (W * 2, H)
    )

    for t in range(T):
        rgb = video_np[t]
        depth_vis = visualize_depth(depth[t])

        combined = np.concatenate([rgb, depth_vis], axis=1)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        writer.write(combined_bgr)

    writer.release()
    print(f"Saved depth results to {depth_dir}")

    return depth


@torch.no_grad()
def run_tracking_inference(model, video, output_dir, num_tracks, track_frame):
    """Run 3D point tracking and save results."""
    print(f"Running tracking inference with {num_tracks} points from frame {track_frame}...")

    video = video.unsqueeze(0)  # (1, T, H, W, 3)
    T, H, W = video.shape[1:4]

    # Sample random query points
    query_points = torch.rand(1, num_tracks, 2)  # Normalized coords
    query_frames = torch.full((1, num_tracks), track_frame, dtype=torch.long)

    # Run tracking
    outputs = model.predict_point_tracks(video, query_points, query_frames)
    tracks_3d = outputs['tracks_3d'].squeeze(0).cpu()  # (N, T, 3)
    tracks_2d = outputs['tracks_2d'].squeeze(0).cpu()  # (N, T, 2)
    visibility = outputs['visibility'].squeeze(0).cpu()  # (N, T)

    video_np = video.squeeze(0).cpu()

    # Visualize tracks
    print("Visualizing tracks...")
    fig = visualize_tracks(video_np, tracks_2d, visibility, num_tracks=min(50, num_tracks))
    fig.savefig(output_dir / 'tracks_2d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3D track visualization
    from utils.visualization import visualize_3d_tracks
    fig = visualize_3d_tracks(tracks_3d)
    fig.savefig(output_dir / 'tracks_3d.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save track data
    np.save(output_dir / 'tracks_3d.npy', tracks_3d.numpy())
    np.save(output_dir / 'tracks_2d.npy', tracks_2d.numpy())
    np.save(output_dir / 'visibility.npy', visibility.numpy())

    print(f"Saved tracking results to {output_dir}")

    return tracks_3d, tracks_2d, visibility


@torch.no_grad()
def run_pointcloud_inference(model, video, output_dir, ref_frame, stride):
    """Run point cloud reconstruction and save results."""
    print(f"Running point cloud inference (ref_frame={ref_frame}, stride={stride})...")

    video = video.unsqueeze(0)  # (1, T, H, W, 3)

    # Predict point cloud
    outputs = model.predict_point_cloud(video, reference_frame=ref_frame, stride=stride)
    points = outputs['points'].squeeze(0).cpu()  # (N, 3)
    colors = outputs['colors'].squeeze(0).cpu()  # (N, 3)
    normals = outputs['normals'].squeeze(0).cpu()  # (N, 3)

    # Filter by depth (remove far points)
    max_depth = points[:, 2].quantile(0.99)
    valid = points[:, 2] < max_depth
    points = points[valid]
    colors = colors[valid]
    normals = normals[valid]

    # Save PLY file
    ply_path = output_dir / 'pointcloud.ply'
    save_point_cloud_ply(str(ply_path), points, colors, normals)
    print(f"Saved point cloud to {ply_path}")

    # Visualization
    fig = visualize_point_cloud(points, colors, point_size=0.5)
    fig.savefig(output_dir / 'pointcloud.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    return points, colors, normals


def main():
    args = parse_args()

    # Import matplotlib here to avoid issues on headless servers
    global plt
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Load video
    print(f"Loading video from {args.video}")
    video = load_video(args.video, args.num_frames, args.img_size)
    video = video.to(device)

    # Save input frames
    input_dir = output_dir / 'input'
    input_dir.mkdir(exist_ok=True)
    video_np = (video.cpu().numpy() * 255).astype(np.uint8)
    for t in range(len(video_np)):
        Image.fromarray(video_np[t]).save(input_dir / f'frame_{t:04d}.png')

    # Run inference
    if args.task in ['all', 'depth']:
        run_depth_inference(model, video, output_dir)

    if args.task in ['all', 'tracking']:
        run_tracking_inference(
            model, video, output_dir,
            args.num_tracks, args.track_frame
        )

    if args.task in ['all', 'pointcloud']:
        run_pointcloud_inference(
            model, video, output_dir,
            args.ref_frame, args.pc_stride
        )

    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()

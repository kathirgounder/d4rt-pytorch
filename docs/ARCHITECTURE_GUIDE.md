# D4RT: Architecture Deep Dive, Training Guide, and MLE Learning Roadmap

> A comprehensive guide for understanding the D4RT codebase, knowing what to expect during training, building a sanity-checking discipline, and charting a career path in mapping/photogrammetry/perception.

---

## Table of Contents

- [Part 1: How D4RT Works](#part-1-how-d4rt-works)
  - [1.1 The Core Idea](#11-the-core-idea)
  - [1.2 Encoder: Video to Global Scene Representation](#12-encoder-video-to-global-scene-representation)
  - [1.3 Query Embedding: The 5-Tuple Interface](#13-query-embedding-the-5-tuple-interface)
  - [1.4 Decoder: Queries x Scene to 3D Predictions](#14-decoder-queries-x-scene-to-3d-predictions)
  - [1.5 Loss: 6 Weighted Components](#15-loss-6-weighted-components)
  - [1.6 Query Sampling: How Training Data is Constructed](#16-query-sampling-how-training-data-is-constructed)
  - [1.7 Camera Pose via Umeyama](#17-camera-pose-via-umeyama)
  - [1.8 Full Data Flow](#18-full-data-flow)
- [Part 2: What to Expect During Training](#part-2-what-to-expect-during-training)
- [Part 3: Sanity Check Roadmap](#part-3-sanity-check-roadmap)
- [Part 4: 2026 New-Grad MLE Roadmap](#part-4-2026-new-grad-mle-roadmap)

---

## Part 1: How D4RT Works

### 1.1 The Core Idea

D4RT is a **feedforward transformer** that takes a video and answers arbitrary 3D questions about it. You give it a **query** `(u, v, t_src, t_tgt, t_cam)` meaning:

> "What is the 3D position of pixel (u,v) from frame t_src, at time t_tgt, expressed in the camera coordinate system of frame t_cam?"

...and it returns a 3D point. By varying the query parameters, a **single model** does depth estimation, point tracking, point cloud reconstruction, and camera pose estimation. No per-task head, no iterative optimization, no test-time fitting.

**How the query interface encodes different tasks:**

| Task | u, v | t_src | t_tgt | t_cam | What you get |
|------|------|-------|-------|-------|-------------|
| **Depth map** | Grid over image | t | t | t | Per-pixel depth (Z of pos_3d) |
| **Point tracking** | Fixed point | Fixed | Varies (1..T) | = t_tgt | 3D trajectory of a single point |
| **Point cloud** | Grid over image | Varies | = t_src | Fixed ref | All points in one coordinate frame |
| **Camera extrinsics** | Grid | Fixed | Varies | Varies | Pose via Umeyama on correspondences |
| **Intrinsics** | Grid | t | t | t | Focal length from pinhole inversion |

This is the key insight: **one architecture, one set of weights, five different capabilities**.

---

### 1.2 Encoder: Video to Global Scene Representation

**File**: `models/encoder.py`

The encoder takes raw video `(B, 3, T=48, H=256, W=256)` and compresses it into a flat sequence of tokens `(B, 6144, 768)` called the **Global Scene Representation** F.

#### Step-by-step tensor flow

| Step | Operation | Shape | Where in code |
|------|-----------|-------|---------------|
| Input | Raw video | `(B, 3, 48, 256, 256)` | -- |
| Patch embed | Conv3d(kernel=2x16x16, stride=2x16x16) | `(B, 6144, 768)` | `encoder.py` lines 42-68 |
| + Pos embed | Learned positional embedding added | `(B, 6144, 768)` | `encoder.py` lines 256-260 |
| + AR token | Aspect ratio linear projection, prepended | `(B, 6145, 768)` | `encoder.py` lines 369-372 |
| 12 blocks | Alternating local/global self-attention | `(B, 6145, 768)` | `encoder.py` lines 375-382 |
| Remove AR | Strip the first token | `(B, 6144, 768)` | `encoder.py` lines 386-387 |

#### Why 6144 tokens?

```
48 frames / 2 (temporal patch size) = 24 temporal slots
256 pixels / 16 (spatial patch size) = 16 patches per dimension
16 x 16 = 256 spatial patches per temporal slot
24 x 256 = 6144 total tokens
```

Each token represents a 2x16x16 spatio-temporal volume of the input video.

#### Why alternating local/global attention?

This is one of the most important design decisions in the encoder:

**Local attention** (`encoder.py` lines 122-146):
```python
# Reshape: (B, 6144, 768) -> (B*24, 256, 768)
# Each of the 24 time slots gets its own attention computation
x = x.reshape(B * num_frames, patches_per_frame, C)
x = self.attn(x)  # attention within 256 spatial patches only
x = x.reshape(B, num_frames * patches_per_frame, C)
```
- Cost: O(256^2) per time slot x 24 slots = manageable
- What it does: Patches in the **same frame** attend to each other
- Why it matters: Builds spatial understanding -- nearby patches exchange information about local geometry, texture boundaries, object extents

**Global attention** (`encoder.py` lines 71-119):
```python
# All 6144 tokens attend to all 6144 tokens
x = self.attn(x)  # full O(6144^2) attention
```
- Cost: O(6144^2) -- expensive but necessary
- What it does: Any patch can attend to **any other patch in any frame**
- Why it matters: Cross-frame reasoning. A patch showing a car in frame 0 can attend to the same car in frame 47. This is how the model learns temporal correspondence.

**Why alternate instead of all-global?**
- Pure global would be prohibitively expensive for all 12 layers
- Local layers are ~24x cheaper and handle spatial structure efficiently
- Alternating gives you both spatial coherence and temporal correspondence at every other layer
- Think of it as: local = "understand each frame", global = "connect frames together"

#### Why the aspect ratio token?

Videos are resized to 256x256 square for the encoder, destroying the original aspect ratio (e.g., a 1280x720 video gets squashed). The aspect ratio token:

1. Takes `[w_ratio, h_ratio]` (e.g., `[1.0, 0.5625]` for 16:9)
2. Projects through a linear layer to 768D
3. Gets prepended to the patch sequence
4. Participates in **global** attention only (excluded from local, since it's not spatial)
5. Gets removed before the output

This way, the model can learn to compensate for the spatial distortion caused by square resizing.

---

### 1.3 Query Embedding: The 5-Tuple Interface

**Files**: `models/embeddings.py`, `models/decoder.py` lines 214-255

A query `(u, v, t_src, t_tgt, t_cam)` becomes a 768-dimensional vector via the **sum** of 5 components:

| Component | What it encodes | How | Norm (untrained) |
|-----------|----------------|-----|-------------------|
| Fourier(u,v) | Spatial position | Multi-scale sinusoids -> Linear | ~6.6 |
| Timestep(t_src) | Source frame | Learned embedding table | ~0.3 |
| Timestep(t_tgt) | Target frame | Learned embedding table | ~0.3 |
| Timestep(t_cam) | Camera frame | Learned embedding table | ~0.3 |
| RGB Patch(9x9) | Local appearance | Extract 9x9 crop, MLP | ~1.7 |
| query_token | Learnable bias | nn.Parameter | ~0 |

Let's examine why each exists:

#### Fourier Embedding (`embeddings.py` lines 9-52)

**Problem**: A raw (u, v) coordinate is just 2 numbers. That's way too low-dimensional for attention to work with -- the model can't distinguish between points 1 pixel apart.

**Solution**: Sinusoidal expansion with 64 exponentially-spaced frequencies:

```python
freqs = 2.0 ** torch.linspace(0, 63, 64)  # [1, 2, 4, 8, ..., ~10^19]
# For each coordinate:
#   sin(freq_0 * 2pi * u), cos(freq_0 * 2pi * u),
#   sin(freq_1 * 2pi * u), cos(freq_1 * 2pi * u),
#   ...
# Total: 2 coords x 2 (sin/cos) x 64 freqs = 256 dimensions
# Then: Linear(256 -> 768)
```

**Intuition**: Low frequencies (freq=1) capture coarse position ("left half vs right half of image"). High frequencies (freq=2^63) capture sub-pixel position. The learned linear projection lets the model decide which frequencies matter for which decoder layers.

This is the same idea as positional encoding in the original Transformer (Vaswani et al. 2017), extended to 2D continuous coordinates.

#### Three Separate Timestep Embeddings (`embeddings.py` lines 55-97)

```python
self.src_embedding = nn.Embedding(128, 768)  # "which frame was the point in?"
self.tgt_embedding = nn.Embedding(128, 768)  # "what time do we want the position at?"
self.cam_embedding = nn.Embedding(128, 768)  # "whose camera coords?"
```

**Why three separate tables?** These have fundamentally different semantic roles:
- **t_src**: Selects which frame to look at for appearance (determines which RGB patch is extracted)
- **t_tgt**: Selects the moment in time for the 3D position (crucial for tracking -- same point, different times)
- **t_cam**: Selects the reference frame for coordinates (crucial for point clouds -- all points in one frame's coordinate system)

For a **depth query**: t_src = t_tgt = t_cam = t (all the same -- "where is this pixel right now, in its own camera")
For a **tracking query**: t_src = 5, t_tgt = 20, t_cam = 20 ("where is the point from frame 5, at time 20, in camera 20's coords")

If they shared one embedding table, the model couldn't distinguish "this is the source frame" from "this is the target frame."

#### The 9x9 RGB Patch (`embeddings.py` lines 213-300)

This is marked as **critical for performance** in the paper. Here's why:

The encoder processes the video in 16x16 spatial patches. A 16x16 patch averages out fine detail -- texture gradients, thin edges, small objects. But the decoder needs to answer queries at **pixel precision**.

The solution: for each query at pixel (u, v) in frame t_src, extract a 9x9 RGB neighborhood directly from the full-resolution frame and project it through an MLP:

```python
# Vectorized extraction via grid_sample (not nested loops!)
patches = F.grid_sample(
    query_frames,       # (B*N_q, 3, H, W) -- one frame per query
    grid,               # (B*N_q, 9, 9, 2)  -- sampling coordinates
    mode='bilinear',
    padding_mode='border'
)
# patches: (B*N_q, 3, 9, 9) -> flatten -> MLP -> (B, N_q, 768)
```

**The `PatchEmbeddingFast` vs `PatchEmbedding` distinction**: The original `PatchEmbedding` (lines 100-210) uses nested Python loops over B and N_q -- functional but slow. `PatchEmbeddingFast` (lines 213-300) uses `F.grid_sample` for a single vectorized GPU kernel. The codebase uses Fast by default. This is a common pattern: prototype with loops, then vectorize for production.

---

### 1.4 Decoder: Queries x Scene to 3D Predictions

**File**: `models/decoder.py`

The decoder is a stack of cross-attention blocks. Let's trace exactly what happens:

#### Architecture

```python
# decoder.py lines 183-189
self.blocks = nn.ModuleList([
    DecoderBlock(embed_dim=768, num_heads=12, mlp_ratio=4.0)
    for _ in range(8)  # 8 layers
])
```

Each `DecoderBlock` (`decoder.py` lines 103-144):
```python
def forward(self, query, encoder_features):
    # Cross-attention: query attends to encoder features
    query = query + self.cross_attn(
        self.norm1(query),           # (B, N_q, 768) -- queries
        self.norm_kv(encoder_features)  # (B, 6144, 768) -- keys/values
    )
    # Feed-forward
    query = query + self.mlp(self.norm2(query))
    return query  # (B, N_q, 768) -- refined queries
```

#### The Critical Design: No Self-Attention Between Queries

In architectures like DETR (object detection), decoded queries interact with each other via self-attention. D4RT deliberately omits this. Each query only talks to the encoder features via cross-attention.

**Why?**
1. **Independence**: Each query represents a different (u,v,t) point. There's no reason pixel (10,20) in frame 3 should need to know about pixel (100,50) in frame 7 to predict its depth.
2. **Parallelism**: At inference, you can decode any number of queries -- 1 or 1 million -- without changing output quality. No O(N_q^2) scaling.
3. **Flexibility**: During training, decode 2048 queries. During inference, decode a full 256x256 depth map (65,536 queries). Same model, same weights.

#### Cross-Attention Mechanics (`decoder.py` lines 11-72)

```python
# Q comes from the decoded query, K and V come from encoder features
q = self.q_proj(query).reshape(B, N_q, num_heads, head_dim).transpose(1, 2)
k = self.k_proj(key_value).reshape(B, N_kv, num_heads, head_dim).transpose(1, 2)
v = self.v_proj(key_value).reshape(B, N_kv, num_heads, head_dim).transpose(1, 2)

# PyTorch's optimized attention (FlashAttention when on GPU)
x = F.scaled_dot_product_attention(q, k, v)
```

In words: each query computes attention weights over all 6144 encoder tokens, then reads a weighted combination of their values. This is how the query "looks up" relevant scene information -- it might attend to the spatial patch matching its (u,v) and the temporal slot matching its t_src.

#### 6 Output Heads (`decoder.py` lines 191-197, 294-300)

After 8 cross-attention blocks, the refined query embeddings are projected through 6 independent linear heads:

```python
pos_3d = self.head_3d(query)                    # (B, N, 3) - raw XYZ position
pos_2d = self.head_2d(query)                    # (B, N, 2) - 2D reprojection
visibility = self.head_vis(query)               # (B, N, 1) - logit
displacement = self.head_disp(query)            # (B, N, 3) - motion vector
normal = F.normalize(self.head_normal(query), dim=-1)  # (B, N, 3) - unit vector
confidence = torch.sigmoid(self.head_conf(query))      # (B, N, 1) - in [0, 1]
```

Note: normals are L2-normalized to unit vectors, and confidence is sigmoided to [0,1]. All other heads output raw values.

---

### 1.5 Loss: 6 Weighted Components

**File**: `losses/losses.py`

```
L_total = 1.0 * L_3d + 0.1 * L_2d + 0.1 * L_vis + 0.1 * L_disp + 0.5 * L_normal + 0.2 * L_conf
```

#### L_3d: The Primary Loss (`losses.py` lines 41-76, 292-316)

This is the most complex loss. Three processing steps before computing L1:

**Step 1 -- Normalize by mean depth** (`losses.py` lines 9-24):
```python
mean_depth = points[..., 2].mean(dim=-1, keepdim=True)  # average Z across queries
normalized = points / (mean_depth + eps)
```
*Why?* Makes the loss scale-invariant. A 1-meter error at 100m depth is trivial, but 1m error at 1m depth is catastrophic. Normalizing by mean depth ensures both scenarios are weighted fairly.

**Step 2 -- Log transform** (`losses.py` lines 27-38):
```python
log_transformed = sign(x) * log(1 + |x|)
```
*Why?* Further dampens influence of very far points. Without this, a single outlier at 1000m depth could dominate the loss even after depth normalization.

**Step 3 -- Confidence weighting** (`losses.py` lines 304-314):
```python
point_loss = |pred_log - target_log|.mean(dim=-1)  # per-point L1: (B, N)
weighted_loss = confidence * point_loss              # high conf = high penalty
loss_3d = (weighted_loss * mask_3d).sum() / mask_3d.sum()
```
*Why?* The model simultaneously predicts 3D position and confidence. If it's unsure about a point (confidence=0.1), the 3D error is downweighted 10x. But L_conf (below) prevents the model from just setting all confidences to 0.

#### L_conf: Confidence Penalty (`losses.py` lines 187-213)

```python
loss = -log(confidence)  # Penalizes low confidence
```

This creates a productive tension with the confidence-weighted L_3d:
- **L_3d wants** confidence = 0 (to zero out errors)
- **L_conf wants** confidence = 1 (to minimize -log penalty)
- **Equilibrium**: confidence is high where predictions are accurate, low where they're uncertain

This is a simple but elegant self-supervised uncertainty mechanism. The model learns to say "I don't know" in the right places.

#### L_normal: Surface Normal Loss (`losses.py` lines 155-184)

```python
cos_sim = (normalize(pred) * normalize(target)).sum(dim=-1)
loss = 1 - cos_sim  # 0 when perfect, 2 when opposite
```

Note the high weight (0.5 vs 0.1 for other auxiliary losses). Surface normals encode local surface orientation -- crucial for mapping because they define how a surface is angled, not just where it is. A depth map can be metric-correct but geometrically noisy; normals penalize that noise.

#### The Mask System

Not every query has every type of ground truth:
- **mask_3d** `(B, N)`: 1 if we have valid depth -> 3D position. Gates L_3d, L_2d, L_vis, L_conf.
- **mask_disp** `(B, N)`: 1 if we have tracking displacement. Only set for tracking queries.
- **mask_normal** `(B, N)`: 1 if we have surface normal ground truth.

If a mask is not provided, the corresponding loss is set to 0. This allows training on datasets with partial annotations (e.g., depth-only datasets without tracks).

---

### 1.6 Query Sampling: How Training Data is Constructed

**File**: `data/dataset.py` -- `QuerySampler` class

Each training sample generates 2048 queries split across three task types:

| Task | % of Queries | Query Pattern | Target Source |
|------|-------------|---------------|---------------|
| Depth | ~50% | t_src = t_tgt = t_cam = t | Unproject depth map to 3D |
| Tracking | ~30% | Fixed (u,v,t_src), varying t_tgt=t_cam | 3D track annotations |
| Point Cloud | ~20% | Varying (u,v,t_src), fixed t_cam | Unproject + transform to ref frame |

#### Boundary-Aware Sampling (`dataset.py` lines 174-234)

30% of spatial samples are drawn from regions near depth boundaries (edges in the depth map). The boundary map is computed via Sobel filtering:

```python
# Sobel on depth -> gradient magnitude
sobel_x = conv2d(depth, sobel_kernel_x)
sobel_y = conv2d(depth, sobel_kernel_y)
boundary_map = sqrt(sobel_x^2 + sobel_y^2)
# 30% of samples: weighted by boundary_map (edges more likely)
# 70% of samples: uniform random
```

*Why?* Object boundaries are where depth changes rapidly -- these are the hardest regions to reconstruct and the most informative for learning fine geometry. Uniform random sampling would under-represent these critical areas.

#### Unprojection: Pixel + Depth -> 3D (`dataset.py` lines 236-271)

The fundamental operation that converts 2D observations to 3D supervision:

```python
# Given: pixel (u_px, v_px), depth d, intrinsics K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
X = (u_px - cx) * d / fx
Y = (v_px - cy) * d / fy
Z = d
point_3d = [X, Y, Z]  # in camera coordinates
```

This is the **pinhole camera model inversion**. The camera projects 3D -> 2D by dividing by Z:
```
u_px = fx * X/Z + cx
v_px = fy * Y/Z + cy
```
Unprojection reverses this given known depth Z = d.

#### How Each Query Type Constructs Targets

**Depth queries** (`dataset.py` lines 273-332):
1. Pick a random frame t and random (u,v)
2. Look up depth d = depth_map[t, v, u]
3. Unproject to 3D: target = [X, Y, Z]
4. Set t_src = t_tgt = t_cam = t
5. mask_3d = 1 if depth is valid, else 0

**Tracking queries** (`dataset.py` lines 334-423):
1. Pick a random track and source frame
2. Look up (u,v) from 2D track annotations
3. Pick a random target frame
4. Target 3D = track_3d[track_idx, tgt_frame], transformed to tgt camera coords
5. Displacement = target_3d - source_3d
6. mask_disp = 1 (tracking queries always have displacement)

**Point cloud queries** (`dataset.py` lines 425-499):
1. Pick a random reference frame (t_cam)
2. Pick random (u,v,t_src) from other frames
3. Unproject in source camera, then transform to reference camera via extrinsics:
   ```
   p_world = R_src @ p_cam + t_src   # source camera -> world
   p_ref = R_ref^T @ (p_world - t_ref)  # world -> reference camera
   ```

---

### 1.7 Camera Pose via Umeyama

**File**: `utils/camera.py`

D4RT doesn't predict camera poses directly. Instead, it recovers them from 3D point correspondences:

1. **Query a grid of points** (e.g., 8x8) in frame i, all in frame i's coordinates: `points_i = decode(coords, t_src=i, t_tgt=i, t_cam=i)`
2. **Query the same points** but in frame j's coordinates: `points_j = decode(coords, t_src=i, t_tgt=i, t_cam=j)`
3. **Umeyama alignment**: Find R, t, s such that `s * R @ points_j + t ≈ points_i`

The **Umeyama algorithm** (`camera.py` lines 82-134) solves this via SVD:

```python
# 1. Center the point clouds
mu_src = weighted_mean(source)
mu_tgt = weighted_mean(target)
source_c = source - mu_src
target_c = target - mu_tgt

# 2. Compute cross-covariance matrix
H = sum(w_i * target_c_i @ source_c_i^T)  # (3, 3)

# 3. SVD decomposition
U, S, Vh = svd(H)

# 4. Rotation (with reflection check)
R = U @ Vh
if det(R) < 0:  # reflection, not rotation
    Vh[-1] *= -1
    R = U @ Vh

# 5. Scale and translation
s = sum(S) / var(source)
t = mu_tgt - s * R @ mu_src
```

This is **the** classic algorithm for rigid body alignment, used throughout robotics, SLAM, and photogrammetry. Understanding it deeply (especially the SVD step) is fundamental.

*Why is this elegant?* The model only needs to predict consistent 3D positions for the same physical points in different coordinate frames. Camera geometry falls out as a byproduct. No camera pose head, no rotation representation issues (quaternion vs. rotation matrix), no gimbal lock.

---

### 1.8 Full Data Flow

```
                          VIDEO (B, 3, 48, 256, 256)
                                    |
                    +---------------+---------------+
                    v                               v
            +-------------+                  +-------------+
            |   ENCODER   |                  |  RGB Frames  |
            |  ViT + L/G  |                  |  (for patch  |
            |  attention   |                  |  extraction) |
            +------+------+                  +------+------+
                   |                                |
         F: (B, 6144, 768)                          |
                   |                                |
                   |        QUERY (u,v,t_s,t_t,t_c) |
                   |               |                |
                   |     +---------+---------+      |
                   |     | Fourier + Timestep |      |
                   |     | + RGB Patch + Token|<-----+
                   |     +---------+---------+
                   |               |
                   |      q: (B, N_q, 768)
                   |               |
                   v               v
            +-----------------------------+
            |         DECODER             |
            |  8x Cross-Attention blocks  |
            |  (q attends to F, not to q) |
            +-------------+---------------+
                          |
             +------------+------------+
             v            v            v
         pos_3d(3)    normal(3)   confidence(1)
         pos_2d(2)    disp(3)    visibility(1)
                          |
                          v
                    +-----+-----+
                    |   LOSS    |
                    | 6 terms   |
                    | weighted  |
                    +-----------+
```

**Training sample lifecycle:**
```
Raw sequence on disk (frames + depth + intrinsics + tracks)
  |  _load_sequence()
  v
Tensors: video (T, H, W, 3) + depth (T, H, W) + intrinsics (3,3) + ...
  |  temporal subsampling (pick 48 of T frames)
  v
Subsampled: video (48, H, W, 3) + aligned depth/tracks
  |  resize to 256x256, scale intrinsics
  v
Resized: video (48, 256, 256, 3)
  |  augmentation (crop + flip + color jitter, consistent across frames)
  v
Augmented video + adjusted targets
  |  QuerySampler.sample() -- 2048 queries with 3D targets
  v
{video, coords, t_src, t_tgt, t_cam, aspect_ratio, targets}
  |  collate_fn() -- stack into batch
  v
Batch tensors -> model.forward() -> criterion() -> loss.backward()
```

---

## Part 2: What to Expect During Training

### Hardware and Time Estimates

| Setup | Config File | Effective Batch | Est. Steps/sec | 4 hours | 7 hours | Full 500k |
|-------|-------------|----------------|-----------------|---------|---------|-----------|
| 1x A100 80GB | `d4rt_azure_a100.yaml` | 64 (bs=4, acc=16) | ~0.8 | ~11.5k | ~20k | ~7 days |
| 1x RTX 5090 32GB | `d4rt_rtx5090.yaml` | 64 (bs=2, acc=32) | ~0.4 | ~5.8k | ~10k | ~14 days |
| 8x A100 80GB | `d4rt_azure_nd96.yaml` | 64 (bs=4, acc=2) | ~5.5 | ~79k | ~139k | ~25 hrs |
| 64x A100 (paper) | -- | 64 (bs=1, gpus=64) | ~46 | ~662k | 500k | ~3 hrs |
| PACE A100 (1 GPU) | `train_pace.sh` | 64 | ~0.8 | ~11.5k | ~20k | ~7 days |

### Loss Trajectory: What You Should See

#### Steps 0-100 (first minutes)
Total loss drops **sharply** from ~0.6-1.0 to ~0.3-0.4. The model is learning basic depth scale and visibility patterns. This is the fastest learning phase.

**If loss doesn't drop here**: Something is fundamentally broken -- check data loading, tensor shapes, video channel ordering.

#### Steps 100-2,500 (warmup phase)
LR is still ramping up linearly from 0 to 1e-4 (`warmup_steps: 2500` in config). Loss continues decreasing to ~0.15-0.25.

Key milestones:
- Confidence head moves from uniform ~0.5 to meaningful values
- L_vis drops as model learns basic visibility (sky=visible, occluded=not)
- L_2d decreases quickly (2D reprojection is "easier" than 3D)

#### Steps 2,500-20,000 (4-7 hours on 1x A100)
Post-warmup, cosine decay begins. This is the **rapid learning** phase.

| Loss Component | Expected Range | What It Means |
|---------------|---------------|---------------|
| L_3d | 0.15 -> 0.05 | Depth structure emerging. Large objects correct, edges fuzzy. |
| L_2d | < 0.08 | 2D reprojections mostly right. Quick convergence. |
| L_vis | 0.3 - 0.5 | Visibility partially learned. Obvious occlusions caught. |
| L_normal | 1.0 -> 0.5 | Surface orientations coarsely correct. Flat surfaces first. |
| L_disp | Varies | Temporal correspondence beginning if tracking data present. |
| L_conf | 1.0 -> 0.5 | Model learning to be confident. |
| **Total** | **0.08 - 0.15** | -- |

**What you can qualitatively see at 10-20k steps:**
- Depth maps capture major scene structure (ground plane, large objects) but miss fine details
- Point tracking follows large motions but drifts on small/occluded targets
- Camera pose estimates roughly correct (~5-10 deg rotation error) for nearby frames

#### Steps 20k-100k
Gradual refinement. This is where:
- Object boundaries sharpen (boundary-aware sampling paying off)
- Fine details emerge (thin structures, small objects)
- Temporal consistency improves (tracks don't drift as much)
- Normal predictions become smooth and geometrically plausible

#### Steps 100k-500k
Diminishing returns but continued improvement:
- Rare cases handled better (strong occlusions, fast motion, reflections)
- Confidence calibration improves
- Final total loss typically 0.02-0.05

### Red Flags

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss = NaN after a few steps | LR too high, or zero depths in log_transform | Check depth values > 0; reduce LR |
| Loss stuck at 0.5+ from start | Data pipeline broken | Check mask_3d.sum() > 0; print depth stats |
| L_conf -> 0, L_3d stays high | Confidence collapse | Increase lambda_conf (try 0.5) |
| L_3d oscillates wildly | Gradient explosion | Verify grad_clip=10.0; check for inf in depths |
| Loss doesn't decrease at all | Wrong video format | Verify BCTHW vs BTHWC permutation |
| Samples/sec degrades over time | Memory leak or data loader issue | Monitor GPU memory; check num_workers |

---

## Part 3: Sanity Check Roadmap

A structured approach to building understanding and verifying correctness, from simple to complex.

### Level 1: Data Pipeline (no GPU, 2 min)

```bash
# Run existing data tests
pytest tests/test_data.py -v
```

**Manual checks to add to your understanding:**
```python
from data import SyntheticDataset, collate_fn

ds = SyntheticDataset(num_samples=5, num_frames=8, img_size=64, num_queries=256)
sample = ds[0]

# 1. Frames should not be black
assert sample['video'].max() > 0.1, "Frames are too dark!"

# 2. Most queries should be valid
validity = sample['targets']['mask_3d'].sum() / sample['targets']['mask_3d'].numel()
print(f"Valid queries: {validity:.0%}")  # Should be 70-95%
assert validity > 0.5, "Too few valid queries!"

# 3. Coordinates should be normalized [0, 1]
assert sample['coords'].min() >= 0 and sample['coords'].max() <= 1

# 4. Timesteps should be within range
T = sample['video'].shape[0]
assert sample['t_src'].max() < T
assert sample['t_tgt'].max() < T
assert sample['t_cam'].max() < T

# 5. Batch collation works
batch = collate_fn([ds[0], ds[1]])
assert batch['video'].shape[0] == 2  # batch dim
```

### Level 2: Model Components (CPU, 1 min)

```bash
pytest tests/test_model.py -v
```

**Manual checks:**
```python
from models import D4RT
from models.encoder import D4RTEncoder
from models.decoder import D4RTDecoder
import torch

# Small model for testing
enc = D4RTEncoder(img_size=64, temporal_size=8, patch_size=(2,8,8),
                  embed_dim=256, depth=4, num_heads=4)
dec = D4RTDecoder(embed_dim=256, depth=4, num_heads=4,
                  max_timesteps=16, patch_size=5)

# 1. Encoder output shape
video = torch.randn(1, 3, 8, 64, 64)
features = enc(video)
print(f"Encoder: {video.shape} -> {features.shape}")
# Should be (1, N_patches, 256)

# 2. Decoder outputs are reasonable ranges
coords = torch.rand(1, 32, 2)
t = torch.zeros(1, 32, dtype=torch.long)
frames = video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
outputs = dec(features, frames, coords, t, t, t)

print(f"pos_3d range: [{outputs['pos_3d'].min():.2f}, {outputs['pos_3d'].max():.2f}]")
print(f"confidence mean: {outputs['confidence'].mean():.2f}")  # ~0.5 untrained
assert not torch.isnan(outputs['pos_3d']).any(), "NaN in outputs!"

# 3. Gradient flow
loss = outputs['pos_3d'].sum()
loss.backward()
assert enc.patch_embed.proj.weight.grad is not None, "No gradient to encoder!"
assert enc.patch_embed.proj.weight.grad.abs().sum() > 0, "Zero gradient!"
```

### Level 3: Overfit One Sample (GPU or CPU, 5 min)

**The single most important sanity check.** If the model can't memorize one training sample, something is architecturally wrong.

```python
# Overfit a single sample
from data import SyntheticDataset, collate_fn
from losses import D4RTLoss

ds = SyntheticDataset(num_samples=1, num_frames=8, img_size=64, num_queries=256)
sample = ds[0]
batch = collate_fn([sample])

model = ...  # your small model
criterion = D4RTLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(500):
    video_input = batch['video'].permute(0, 4, 1, 2, 3)
    preds = model(video_input, batch['coords'], batch['t_src'],
                  batch['t_tgt'], batch['t_cam'], batch['aspect_ratio'])
    losses = criterion(preds, batch['targets'])

    optimizer.zero_grad()
    losses['loss'].backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: loss={losses['loss'].item():.4f}")

# After 500 steps on ONE sample, loss should be < 0.01
assert losses['loss'].item() < 0.05, f"Can't overfit! Loss={losses['loss'].item()}"
```

### Level 4: Short Training Run (GPU, 30 min)

```bash
python train.py --config configs/d4rt_test.yaml --dataset synthetic --steps 1000
```

Check:
- Loss trajectory decreases monotonically (with noise)
- No NaN/Inf
- TensorBoard shows LR ramp + cosine decay
- Memory usage stable (no leak)
- Checkpoints saved at configured frequency

### Level 5: Real Data (GPU, 1+ hour)

```bash
python train.py --config configs/d4rt_test.yaml --data_root data_samples --dataset video --steps 5000
```

Check:
- Loss decreases (slower than synthetic, expected)
- Depth predictions show real structure after ~1000 steps
- Compare MiDaS depth (pseudo-GT) vs. model prediction qualitatively

### Level 6: Evaluation Protocol

At checkpoints (5k, 20k, 50k, 100k, 500k steps), run:

1. **Depth maps**: `model.predict_depth(video)` -- do boundaries sharpen over training?
2. **Point tracks**: `model.predict_point_tracks(video, points, frames)` -- temporally smooth?
3. **Camera pose**: `estimate_camera_pose(model, ...)` -- rotation error decreasing?
4. **Point cloud**: `model.predict_point_cloud(video)` -- coherent 3D structure?

---

## Part 4: 2026 New-Grad MLE Roadmap

### The Skill Tree

```
                    FOUNDATIONS
                        |
          +-------------+-------------+
          |             |             |
    Linear Algebra   Probability   Optimization
    (projections,    (Bayes,       (SGD, Adam,
     SVD, rigid      MLE,          LR schedules,
     transforms)     filtering)    loss landscapes)
          |             |             |
          +-------------+-------------+
                        |
                  COMPUTER VISION
                        |
          +-------------+-------------+
          |             |             |
    Camera Models    Feature        Neural Nets
    (pinhole,        Matching       (CNNs, ViTs,
     intrinsics,     (SIFT ->       attention,
     extrinsics,     SuperPoint ->  transformers)
     distortion)     LightGlue)
          |             |             |
          +-------------+-------------+
                        |
                 3D RECONSTRUCTION
                        |
          +-------------+-------------+
          |             |             |
    Classical SfM    Depth Est.    Neural Scene
    (COLMAP,         (monocular:   Representations
     bundle adj,     MiDaS, DPT,  (NeRF -> 3DGS ->
     Umeyama,        Depth Any-   feed-forward
     PnP, RANSAC)    thing)       models like D4RT)
          |             |             |
          +-------------+-------------+
                        |
                   MAPPING & SLAM
                        |
          +-------------+-------------+
          |             |             |
    Visual SLAM      Semantic       Autonomous
    (ORB-SLAM3,      Understanding  Systems
     DROID-SLAM,     (segmentation, (sensor fusion,
     Gaussian-SLAM)  object det,    planning,
                     panoptic)      real-time)
```

### What You Already Know (from this codebase)

By working through D4RT, you've gained hands-on experience with:

| Concept | Where in D4RT | Industry relevance |
|---------|--------------|-------------------|
| **ViT architecture** | `models/encoder.py` | Foundation of modern vision (DINOv2, SAM, Depth Anything) |
| **Fourier features** | `models/embeddings.py` | Used in NeRF, 3DGS, any coordinate-based network |
| **Cross-attention** | `models/decoder.py` | Core of DETR, Perceiver, Stable Diffusion, all query-based models |
| **Camera geometry** | `utils/camera.py`, `data/dataset.py` | Fundamental to ANY 3D/mapping system |
| **Umeyama/Procrustes** | `utils/camera.py` | Standard pose estimation in SLAM and SfM |
| **Mixed precision + grad accumulation** | `train.py` | Required for any large-scale training job |
| **Multi-task loss balancing** | `losses/losses.py` | Common in autonomous driving, robotics |
| **Boundary-aware sampling** | `data/dataset.py` | Active learning and hard-example mining |

### What to Study Next

**Priority 1 -- The Modern Depth Stack:**
- **Depth Anything V2**: Successor to MiDaS. State-of-art monocular depth. Understanding how it trains (large-scale pseudo-labeling) is key.
- **UniDepth**: Metric depth estimation (absolute scale, not just relative). Critical for mapping where you need real-world measurements.

**Priority 2 -- The 3D Reconstruction Frontier:**
- **DUSt3R / MASt3R** (Naver Labs, 2024): Pairwise 3D reconstruction without camera calibration. Very similar to D4RT's query paradigm. The hottest papers in the space.
- **3D Gaussian Splatting**: The dominant real-time rendering representation since 2023. Learn gsplat library. Every mapping/XR company is using this.
- **COLMAP**: Classical Structure-from-Motion. Still the gold standard for ground truth. Run it on your city aerial video to understand what "traditional" looks like.

**Priority 3 -- Systems Knowledge:**
- **FlashAttention**: D4RT uses it via `F.scaled_dot_product_attention`. Understanding memory-efficient attention is essential for scaling.
- **Distributed training (DDP, FSDP)**: `train.py` supports multi-GPU via DDP. Understanding this is required for any production training job.
- **ONNX / TensorRT export**: The gap between "model trains" and "model deploys" is where engineers are needed most.

### Hands-On Projects

1. **Run COLMAP on the city aerial video**
   - Extract sparse SfM. Compare poses to what D4RT predicts.
   - This grounds your understanding: classical methods are slow but geometrically rigorous; learned methods are fast but approximate.
   - Deliverable: Side-by-side pose comparison plot.

2. **Fine-tune D4RT on your own video**
   - Record a 10-second video of your room. Run MiDaS for depth. Train D4RT for 10k steps.
   - Study the failure modes: where does the model get confused? Reflections? Textureless walls? Thin objects?
   - Deliverable: Before/after depth maps at checkpoints.

3. **D4RT Point Cloud -> Gaussian Splatting**
   - Use D4RT's `predict_point_cloud()` to get colored 3D points.
   - Initialize 3D Gaussians from those points. Optimize with gsplat.
   - This bridges feed-forward prediction (D4RT) with optimization-based rendering (3DGS).
   - Deliverable: Novel view synthesis from a video.

4. **Real-time depth demo**
   - Export D4RT to ONNX. Run on webcam at 10+ FPS.
   - This teaches you the engineering gap: batch normalization in eval mode, input preprocessing pipelines, GPU memory management, async inference.
   - Deliverable: Live depth overlay on webcam feed.

### The 2026 Job Landscape

**What's shifted**: Feed-forward models (D4RT, DUSt3R) are replacing iterative optimization (COLMAP, NeRF test-time fitting) for many tasks. The trend: train once on massive data, amortize at inference. This means ML engineering skills matter more than ever for deploying these systems, but geometry understanding is essential for debugging and evaluation.

**Who's hiring and what they want:**

| Sector | Companies | They want you to know |
|--------|----------|----------------------|
| **Autonomous driving** | Waymo, Aurora, Cruise, Zoox | Real-time perception, LiDAR-camera fusion, safety guarantees |
| **Mapping** | Niantic, Google, Apple Maps | Large-scale 3D reconstruction, localization, visual quality |
| **Robotics** | Figure, 1X, Boston Dynamics | Spatial understanding, sim2real, generalization |
| **AR/VR** | Meta Reality Labs, Apple Vision | Real-time depth, SLAM, consumer hardware constraints |
| **Geospatial** | Planet, Palantir, various startups | Satellite/aerial photogrammetry, GIS integration |

**Key interview topics:**
1. Camera models: pinhole + radial distortion. Be able to derive projection/unprojection.
2. Rigid body transforms: SE(3), rotation representations (matrix vs quaternion vs axis-angle), composition.
3. Epipolar geometry: fundamental matrix, essential matrix, triangulation. Conceptual understanding.
4. RANSAC: why it exists, how it works, when it fails.
5. Transformer attention: complexity analysis, FlashAttention, KV-cache.
6. Training at scale: mixed precision, gradient accumulation, distributed training, learning rate schedules.
7. Evaluation metrics: depth (AbsRel, RMSE), tracking (survival rate, median error), pose (ATE, RPE).

---

## Quick Reference: Key Files

| File | What to study | Lines |
|------|--------------|-------|
| `models/encoder.py` | ViT, local/global attention, patch embedding | PatchEmbed3D: 27-68, LocalAttn: 122-146, D4RTEncoder: 216-389 |
| `models/decoder.py` | Cross-attention, output heads, query building | CrossAttention: 11-72, DecoderBlock: 103-144, build_query: 214-255 |
| `models/embeddings.py` | Fourier, timestep, RGB patch embeddings | Fourier: 9-52, Timestep: 55-97, PatchFast: 213-300 |
| `models/d4rt.py` | Full model, inference methods | predict_depth: 148-205, predict_tracks: 207-262 |
| `losses/losses.py` | All 6 loss functions, confidence weighting | normalize: 9-24, log_transform: 27-38, D4RTLoss.forward: 245-374 |
| `data/dataset.py` | Query sampling, boundary detection, unprojection | QuerySampler: 16-499, boundary: 174-234, unproject: 236-271 |
| `utils/camera.py` | Umeyama, pose estimation, intrinsics recovery | umeyama: 82-134, estimate_pose: 178-255 |
| `train.py` | Training loop, LR schedule, mixed precision | forward_backward: 290-349, optimizer_step: 352-375 |

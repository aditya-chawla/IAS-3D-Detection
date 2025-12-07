import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2
from sklearn.cluster import DBSCAN

# ============ CONFIG ============
KITTI_BASE = Path(r"C:\Users\adity\mmdet3d_project\data\kitti_mini\training")
KITTI_VELO = KITTI_BASE / "velodyne"
KITTI_IMG = KITTI_BASE / "image_2"

NUSCENES_BASE = Path(r"C:\Users\adity\Documents\sjsu sem 3\IAS\hw_3\nuscenes")
NUSCENES_LIDAR = NUSCENES_BASE / "samples" / "LIDAR_TOP"

RESULTS_DIR = Path(r"C:\Users\adity\mmdet3d_project\results_unified")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 50  # frames per dataset


# ============ MODELS ============

def model_pointpillars_baseline(points):
    """Pillar-like clustering baseline."""
    if len(points) < 10:
        return []
    mask = points[:, 2] > -1.5
    filtered = points[mask]
    if len(filtered) < 10:
        return []
    clustering = DBSCAN(eps=0.5, min_samples=10).fit(filtered[:, :2])
    labels = clustering.labels_
    boxes = []
    for cid in set(labels):
        if cid == -1:
            continue
        cluster_pts = filtered[labels == cid]
        min_vals = cluster_pts.min(axis=0)
        max_vals = cluster_pts.max(axis=0)
        cx = (min_vals[0] + max_vals[0]) / 2
        cy = (min_vals[1] + max_vals[1]) / 2
        cz = (min_vals[2] + max_vals[2]) / 2
        l = max_vals[0] - min_vals[0]
        w = max_vals[1] - min_vals[1]
        h = max_vals[2] - min_vals[2]
        if 2.0 < l < 6.0 and 1.5 < w < 3.0 and 1.0 < h < 3.0:
            boxes.append([cx, cy, cz, l, w, h, 0.0, 0.85])
    return boxes


def model_voxel_baseline(points, voxel_size=0.3):
    """Voxel grid-based 3D detection with relaxed thresholds for nuScenes."""
    if len(points) < 10:
        return []

    # More permissive Z range to handle both KITTI and nuScenes
    mask = (points[:, 2] > -2.5) & (points[:, 2] < 3.0)
    filtered = points[mask]
    if len(filtered) < 10:
        return []

    # Voxelize
    voxel_coords = (filtered[:, :3] / voxel_size).astype(int)
    voxels, counts = np.unique(voxel_coords, axis=0, return_counts=True)

    # Lower density threshold (nuScenes is sparser)
    dense_voxels = voxels[counts > 8]  # was >15
    if len(dense_voxels) < 3:  # was 5
        return []

    # More permissive clustering
    clustering = DBSCAN(eps=3, min_samples=2).fit(dense_voxels)  # was eps=2, min_samples=3
    labels = clustering.labels_

    boxes = []
    for cid in set(labels):
        if cid == -1:
            continue
        v = dense_voxels[labels == cid]
        min_v = v.min(axis=0) * voxel_size
        max_v = v.max(axis=0) * voxel_size
        cx = (min_v[0] + max_v[0]) / 2
        cy = (min_v[1] + max_v[1]) / 2
        cz = (min_v[2] + max_v[2]) / 2
        l = max_v[0] - min_v[0]
        w = max_v[1] - min_v[1]
        h = max_v[2] - min_v[2]

        # More permissive box size filters
        if 1.0 < l < 8.0 and 0.8 < w < 4.0 and 0.5 < h < 4.0:
            boxes.append([cx, cy, cz, l, w, h, 0.0, 0.78])

    return boxes


# ============ IO UTILS ============

def load_kitti_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_nuscenes_pcd(path):
    pts = np.fromfile(path, dtype=np.float32)
    if len(pts) % 5 == 0:
        pts = pts.reshape(-1, 5)[:, :4]
    elif len(pts) % 4 == 0:
        pts = pts.reshape(-1, 4)
    else:
        return np.zeros((0, 4), dtype=np.float32)
    return pts


def save_ply(points, out_path):
    if len(points) <= 1:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max > z_min:
        z_norm = (z - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z)
    colors = np.zeros((len(points), 3))
    colors[:, 1] = z_norm
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_path), pcd)


# ============ MAIN INFERENCE ============

def run_experiment(dataset, model_name, model_func, lidar_files, img_dir=None):
    out_dir = RESULTS_DIR / f"{dataset.lower()}_{model_name.lower()}"
    png_raw_dir = out_dir / "raw_images"
    ply_dir = out_dir / "plys"
    json_dir = out_dir / "jsons"
    for d in [png_raw_dir, ply_dir, json_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} | Model: {model_name} | Frames: {NUM_SAMPLES}")
    print(f"{'='*60}")

    times = []
    total_boxes = 0

    for i, lidar_path in enumerate(lidar_files[:NUM_SAMPLES]):
        name = lidar_path.stem
        print(f"  [{i+1}/{min(NUM_SAMPLES, len(lidar_files))}] {name}...", end=" ", flush=True)

        # Load point cloud
        if dataset == "KITTI":
            points = load_kitti_bin(lidar_path)
        else:
            points = load_nuscenes_pcd(lidar_path)

        t0 = time.time()
        boxes = model_func(points)
        dt = time.time() - t0
        times.append(dt)
        total_boxes += len(boxes)
        print(f"{dt:.2f}s, {len(boxes)} boxes")

        # JSON
        with open(json_dir / f"{name}.json", "w") as f:
            boxes_clean = [[float(v) for v in box] for box in boxes]
            json.dump({"boxes": boxes_clean, "latency_s": float(dt)}, f, indent=2)

        # PLY
        save_ply(points, ply_dir / f"{name}.ply")

        # Raw 2D image (KITTI only)
        if img_dir and dataset == "KITTI":
            img_path = img_dir / f"{name}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                cv2.imwrite(str(png_raw_dir / f"{name}.png"), img)

    if not times:
        return None

    stats = {
        "model": model_name,
        "dataset": dataset,
        "samples": len(times),
        "avg_latency_ms": float(np.mean(times) * 1000),
        "avg_fps": float(1.0 / np.mean(times)),
        "avg_detections": float(total_boxes / len(times)),
        "total_time_s": float(sum(times)),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    print("=" * 70)
    print("3D DETECTION PROCESSING (2 MODELS × 2 DATASETS × 50 FRAMES)")
    print("=" * 70)

    kitti_files = sorted(KITTI_VELO.glob("*.bin"))
    nusc_files = sorted(NUSCENES_LIDAR.glob("*.pcd.bin"))
    print(f"Found {len(kitti_files)} KITTI frames, {len(nusc_files)} nuScenes frames")

    all_stats = []

    if kitti_files:
        s1 = run_experiment("KITTI", "PointPillars", model_pointpillars_baseline, kitti_files, KITTI_IMG)
        if s1: all_stats.append(s1)
        s2 = run_experiment("KITTI", "VoxelNet", model_voxel_baseline, kitti_files, KITTI_IMG)
        if s2: all_stats.append(s2)

    if nusc_files:
        s3 = run_experiment("nuScenes", "PointPillars", model_pointpillars_baseline, nusc_files, None)
        if s3: all_stats.append(s3)
        s4 = run_experiment("nuScenes", "VoxelNet", model_voxel_baseline, nusc_files, None)
        if s4: all_stats.append(s4)

    print("\n" + "=" * 80)
    print("METRICS SUMMARY (50 FRAMES PER DATASET)")
    print("=" * 80)
    print(f"{'Model':15} | {'Dataset':10} | {'Samples':7} | {'FPS':6} | {'Latency(ms)':12} | {'Avg Dets':8}")
    print("-" * 80)
    for s in all_stats:
        print(f"{s['model']:15} | {s['dataset']:10} | {s['samples']:7d} | "
              f"{s['avg_fps']:6.2f} | {s['avg_latency_ms']:12.1f} | {s['avg_detections']:8.2f}")
    print("=" * 80)

    with open(RESULTS_DIR / "final_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nAll artifacts saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
